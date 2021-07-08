""" Make simulations of UDGs for SED fitting with paintbox. """
import os

import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.io import fits
from astropy.table import Table, join
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from scipy.ndimage.filters import gaussian_filter
from photutils import EllipticalAperture, EllipticalAnnulus
from photutils import aperture_photometry
from astropy.stats import sigma_clipped_stats
from astropy.modeling.models import Sersic2D
import scipy.ndimage

import context
from estimate_sn_simulations import get_zps
from test_photometry import SersicPars

def smudges_morph_properties(nsim=1000, plot=False):
    """ Read photometric table from SMUDGEs to set simulation properties. """
    table = Table.read(os.path.join(context.tables_dir,
                             "DESI_UDGs_sorted_nodupsPA.csv"))
    table = table[table["mu_0_g"] > 20]
    table = table[table["Rearcsec"] < 30]
    columns = ["Rearcsec", "AR", "n", "mu_0_g", "PA"]
    gs = gridspec.GridSpec(2, 6)
    iy = [0, 0, 0, 1, 1]
    ix = [0, 2, 4, 1, 3]
    morph_table = Table()
    morph_table["ID"] = [f"{_:05d}" for _ in np.arange(nsim) + 1]
    for i, j, column in zip(ix, iy, columns):
        sample = table[column]
        morph_table[column] = np.random.choice(sample, nsim)
        if plot:
            ax = plt.subplot(gs[j, i:i+2])
            ax.hist(sample)
    if plot:
        plt.show()
    return morph_table

def make_sim_table(Nsim, N_pregrid, Nparam):
    outtable = os.path.join(context.tables_dir,
            f"morph_sed_Nsim{Nsim}_Npregrid{N_pregrid}_Nparam{Nparam}.fits")
    if os.path.exists(outtable):
        t = Table.read(outtable)
        return t
    # Read SED simulations
    sim_table = os.path.join(context.tables_dir,
                             f"simulations_dense_basis_N{Nparam}_Nrepgri"
                             f"d{N_pregrid}_Nsim{Nsim}.fits")
    sed_table = Table.read(sim_table)
    morph_table = smudges_morph_properties(Nsim)
    sim_table = join(morph_table, sed_table)
    sim_table.write(outtable, overwrite=True)
    return sim_table

def make_simulations(Nsim, N_pregrid, Nparam, redo=False):
    flamerr = np.loadtxt("assets/noise_50.dat", usecols=(1,), skiprows=1) * \
        context.flam_unit
    wave = np.array([context.wave_eff[band] for band in context.bands]) * \
           u.Angstrom
    fnuerr = flamerr / const.c * wave**2
    fnuerr = fnuerr.to(context.fnu_unit).value
    seeing = 1.7
    exptimes = 3 * np.array([context.expsingle[band] for band in context.bands])
    gain = context.gain
    effgain = gain * exptimes
    PS = context.PS
    sim_table = make_sim_table(Nsim, N_pregrid, Nparam)
    simname = f"sim{Nsim}_grid{N_pregrid}_Npar{Nparam}"
    cubes_outdir = os.path.join(context.home_dir, "simulations", simname)
    zptable = get_zps()
    zps = np.array([np.median(zptable[band]) for band in context.bands])
    f0 = np.power(10, -0.4 * (48.6 + zps))
    idx = context.bands.index("G")
    if not os.path.exists(cubes_outdir):
        os.mkdir(cubes_outdir)
    nre = 4
    desc = "Producing simulated cubes."
    for sim in tqdm(sim_table, desc=desc):
        output = os.path.join(cubes_outdir, f"sim_{sim['ID']}.fits")
        if os.path.exists(output) and not redo:
            continue
        ########################################################################
        # Parameters for Sersic profile in simulation
        mu0 = sim["mu_0_g"]
        re = sim["Rearcsec"]
        re_pix = re / PS
        ar = sim["AR"]
        ellip = 1 - ar
        n = sim["n"]
        PA = sim["PA"]
        ang = np.deg2rad(-PA)
        mu0_obs = mu0 - 2.5 * np.log10(PS**2)
        sersic = SersicPars(mu0_obs, re_pix, n, ar)
        Ie = np.power(10, -0.4 * (sersic.mue + 48.6))
        ########################################################################
        # Create 2D arrays
        size = np.ceil(re_pix).astype(np.int)
        rs = np.linspace(-nre * size, nre * size, nre * 2 * size+ 1)
        x2D, y2D = np.meshgrid(rs, rs)
        model = Sersic2D(amplitude=Ie, r_eff=re_pix, n=n, x_0=0, y_0=0,
                         ellip=ellip, theta=ang)
        fnu2D = model(x2D, y2D)
        ########################################################################
        # Producing model for all bands
        fnu_model = np.array([sim[band] for band in context.bands])
        fnunorm = fnu_model[idx]
        fnu_model /= fnunorm
        fnu3D = fnu2D[None, :, :] * fnu_model[:, None, None]
        ########################################################################
        # Adding noise to the model
        counts = fnu3D / f0[:, None, None] * exptimes[:,None,None]
        ne = counts * gain
        gal_e = np.random.poisson(ne)
        gal_c = gal_e / gain / exptimes[:, None, None]
        gal = f0[:, None, None] * gal_c
        skynoise = np.array([np.random.normal(0, f, x2D.shape) for f in fnuerr])
        simcube = skynoise + gal
        ########################################################################
        # Performing photometry
        x0 = y0 = 0.5 * len(rs) - 0.5
        aperture = EllipticalAperture((x0, y0), a=re_pix, b=re_pix * ar,
                                      theta=ang)
        annulus_aperture = EllipticalAnnulus((x0, y0), a_in=2.5 * re_pix,
            a_out=4 * re_pix, b_out=4 * re_pix * ar, b_in=2.5 * re_pix * ar,
            theta=ang)
        fnu_phot = np.zeros(12)
        fnu_re_model = np.zeros(12)
        fnuerr_phot = np.zeros_like(fnu_phot)
        for i, band in enumerate(context.bands):
            # Calculating expected mag at Re
            color = -2.5 * np.log10(fnu_model[i] / fnu_model[idx])
            mre_model= sersic.mre + color
            fnu_re_model[i] = np.power(10, -0.4 * (mre_model + 48.6))
            data = simcube[i, :, :]
            # Getting background noise
            annulus_masks = annulus_aperture.to_mask(method='center')
            annulus_data = annulus_masks.multiply(data)
            mask = annulus_masks.data
            annulus_data_1d = annulus_data[mask > 0]
            bkgmean, bkgmed, bkgstd = sigma_clipped_stats(annulus_data_1d)
            # Performing photometry on galaxy
            phot = aperture_photometry(data, aperture)
            fnu_phot[i] = phot["aperture_sum"].data[0]
            # Uncertainties
            bgnoise = np.sqrt(aperture.area * bkgstd**2)
            shotnoise = np.clip(fnu_phot[i] / effgain[i], 0, np.infty)
            fnuerr_phot[i] = np.sqrt(bgnoise**2 + shotnoise**2)
            # im = plt.imshow(data)
            # aperture.plot(color='r', lw=2)
            # annulus_aperture.plot(color="r", linestyle="--")
            # plt.colorbar(im)
            # plt.show()
        t = Table([context.bands, wave, fnu_re_model, fnu_phot, fnuerr_phot],
                  names=["bands", "wave", "fnu_model", "fnu_sim", "fnuerr_sim"])
        # Saving results
        hdu1 = fits.ImageHDU(simcube)
        hdu2 = fits.BinTableHDU(t)
        hdu3 = fits.BinTableHDU(Table(sim))
        hdulist = fits.HDUList([fits.PrimaryHDU(), hdu1, hdu2, hdu3])
        hdulist.writeto(output, overwrite=True)
        # plt.errorbar(t["wave"], t["fnu_sim"], yerr=t["fnuerr_sim"],  marker="o",
        #              ls="none", ecolor="0.8")
        # plt.plot(t["wave"], t["fnu_model"])
        # # plt.ylim(plt.ylim()[::-1])
        # plt.show()


if __name__ == "__main__":
    # Simulation parameters
    Nsim = 1000
    N_pregrid = 20000
    Nparam = 3
    make_simulations(Nsim, N_pregrid, Nparam)
    # Read Dense Basis model
    atlas_dir = os.path.join(context.home_dir, "dense_basis_models")
    atlas_file = os.path.join(atlas_dir,
                          f"dbgrid_splus_{N_pregrid}_Nparam_{Nparam}.dbatlas")
    if not os.path.exists(atlas_file):
        raise Exception("Dense basis models with given parameters is not set.")



