""" Make simulations of UDGs for SED fitting with paintbox. """
import os

import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.io import fits
from scipy.ndimage import gaussian_filter
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

import context
from estimate_sn_simulations import get_zps

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

def sersic(x, y, amplitude, r_eff, n, x_0, y_0, ellip, theta):
    """Two dimensional Sersic profile function."""
    a, b = r_eff, (1 - ellip) * r_eff
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    x_maj = (x - x_0) * cos_theta + (y - y_0) * sin_theta
    x_min = -(x - x_0) * sin_theta + (y - y_0) * cos_theta
    sma = np.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)
    return amplitude * np.exp(-((sma / r_eff) ** (1 / n)))



def make_cubes(Nsim, N_pregrid, Nparam):
    flamerr = np.loadtxt("assets/noise_50.dat", usecols=(1,), skiprows=1) * \
        context.flam_unit
    wave = np.array([context.wave_eff[band] for band in context.bands]) * \
           u.Angstrom
    fnuerr = flamerr / const.c * wave**2
    fnuerr = fnuerr.to(context.fnu_unit).value
    seeing = 1.7
    exptimes = 3 * np.array([context.expsingle[band] for band in context.bands])
    gain = context.gain
    PS = context.PS
    sim_table = make_sim_table(Nsim, N_pregrid, Nparam)
    simname = f"sim_Nsim{Nsim}_Npregrid{N_pregrid}_Nparam{Nparam}"
    outdir = os.path.join(context.home_dir, "smudges", simname)
    zptable = get_zps()
    zps = np.array([np.median(zptable[band]) for band in context.bands])
    f0 = np.power(10, -0.4 * (48.6 + zps))
    idx = context.bands.index("G")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    nre = 4
    desc = "Producing simulated cubes."
    for sim in tqdm(sim_table, desc=desc):
        output = os.path.join(outdir, f"sim_{sim['ID']}.fits")
        if os.path.exists(output):
            continue
        mag0 = sim["mu_0_g"] - 2.5 * np.log10(PS**2)
        fnu0 = np.power(10, -0.4 * (mag0 + 48.6))
        re = np.ceil(sim["Rearcsec"] / PS).astype(np.int)
        r = np.linspace(-nre * re, nre * re, nre * 2 * re + 1)
        x, y = np.meshgrid(r, r)
        fnug = sersic(x, y, fnu0, re, sim["n"], 0, 0, 1 - sim["AR"],
                   np.deg2rad(sim["PA"]))
        fmodel = np.array([sim[band] for band in context.bands])
        fmodel /= fmodel[idx]
        fnu = fnug[None, :, :] * fmodel[:, None, None]
        counts = fnu / f0[:, None, None] * exptimes[:,None,None]
        ne = counts * gain
        gal_e = np.random.poisson(ne)
        gal_c = gal_e / gain / exptimes[:, None, None]
        gal = f0[:, None, None] * gal_c
        skynoise = np.array([np.random.normal(0, f, x.shape) for f in fnuerr])
        cube = skynoise + gal
        hdu1 = fits.ImageHDU(cube)
        hdulist = fits.HDUList([fits.PrimaryHDU(), hdu1])
        hdulist.writeto(output, overwrite=True)

def make_photometry(Nsim, N_pregrid, Nparam):
    """ Forced photometry of cubes. """
    wave = np.array([context.wave_eff[band] for band in context.bands]) * \
           u.Angstrom
    sim_table = make_sim_table(Nsim, N_pregrid, Nparam)
    simname = f"sim{Nsim}_grid{N_pregrid}_Npar{Nparam}"
    wdir = os.path.join(context.home_dir, "smudges", simname)
    exptimes = 3 * np.array([context.expsingle[band] for band in context.bands])
    gain = context.gain * exptimes
    for sim in sim_table:
        re = sim["Rearcsec"] / context.PS
        a = re
        b = re * sim["AR"]
        pa = sim["PA"]
        ang = np.deg2rad(pa)
        cube = os.path.join(wdir, f"sim_{sim['ID']}.fits")
        cube = fits.getdata(cube, hdu=1)
        zdim, ydim, xdim = cube.shape
        x0 = 0.5 * xdim
        y0 = 0.5 * ydim
        aperture = EllipticalAperture((x0, y0), a=a, b=b, theta=ang)
        annulus_aperture = EllipticalAnnulus((x0, y0), a_in=2.5 * a,
                                             a_out=4 * a, b_out=4 * b,
                                             theta=ang)
        fnu = fnuerr = np.zeros(12)
        for i, band in enumerate(context.bands):
            data = cube[i,:,:]
            annulus_masks = annulus_aperture.to_mask(method='center')
            annulus_data = annulus_masks.multiply(data)
            mask = annulus_masks.data
            annulus_data_1d = annulus_data[mask > 0]
            bkgmean, bkgmed, bkgstd = sigma_clipped_stats(annulus_data_1d)
            fnu[i] = aperture_photometry(data,
                                       aperture)["aperture_sum"].data[0]
            if fnu[i] < 0:
                fnuerr[i] = np.sqrt(aperture.area * bkgstd**2)
            else:
                fnuerr[i] = np.sqrt(aperture.area * bkgstd ** 2 + fnu[i] /
                                    gain[i])
            vmin = np.percentile(data, 10)
            vmax = np.percentile(data, 99)
            plt.imshow(data, vmin=vmin, vmax=vmax, cmap="viridis")
            aperture.plot(color='r', lw=2)
            annulus_aperture.plot(color="r", linestyle="--")
            plt.show()
            fmodel = np.array([sim[band] for band in context.bands])
        plt.errorbar(wave.value, fnu, yerr=fnuerr, marker="o", ls="none",
                     ecolor="0.8")
        norm = np.median(fnu / fmodel)
        plt.plot(wave, norm * fmodel)
        plt.axhline(y=0, c="k", ls="--")
        plt.show()

        # stack = gaussian_filter(np.sum(cube, axis=(0,)), seeing / 2.355 / PS)
        # vmin = np.percentile(stack, 50)
        # vmax = np.percentile(stack, 99)
        # plt.imshow(stack, origin="lower", vmin=vmin, vmax=vmax)
        # # plt.colorbar()
        # plt.show()


if __name__ == "__main__":

    # Simulation parameters
    Nsim = 1000
    N_pregrid = 20000
    Nparam = 3

    make_cubes(Nsim, N_pregrid, Nparam)
    make_photometry(Nsim, N_pregrid, Nparam)
    # Read Dense Basis model
    atlas_dir = os.path.join(context.home_dir, "dense_basis_models")
    atlas_file = os.path.join(atlas_dir,
                          f"dbgrid_splus_{N_pregrid}_Nparam_{Nparam}.dbatlas")
    if not os.path.exists(atlas_file):
        raise Exception("Dense basis models with given parameters is not set.")



