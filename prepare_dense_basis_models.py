""" Testing the results of the dense basis with S-PLUS. """
import specutils.io.default_loaders.sixdfgs_reader

import context

import os

import numpy as np

import fsps
import astropy.units as u
import astropy.constants as const
from astropy.table import Table, vstack, hstack
from tqdm import tqdm
import dense_basis as db

import seaborn as sns

import matplotlib.pyplot as plt
sns.set()
plt.rcParams["text.usetex"] = True

def set_priors(Nparam):
    """ Set priors for dense basis models. """
    priors = db.Priors()
    priors.tx_alpha = 3.0
    priors.mass_max = 10
    priors.mass_min = 5
    priors.Nparam = Nparam
    priors.z_min = 0
    priors.z_max = 0.05
    priors.Z_min = -2.5
    priors.Z_max = 0.50
    return priors

def prepare_models(N_pregrid=20000, Nparam=3):
    # Setting up dense basis
    filt_dir = os.path.join(os.getcwd(), "filter_curves-master")
    filter_list = "filter_list_splus.dat"
    fname = "dbgrid_splus"
    models_dir = os.path.join(context.home_dir, "models")
    path = models_dir + "/"
    model_name = f"{fname}_{N_pregrid}_Nparam_{Nparam}.fits"
    dbfile = os.path.join(models_dir, model_name)
    priors = set_priors(Nparam)
    # priors.plot_prior_distributions()
    atlas = db.generate_atlas(N_pregrid=N_pregrid, priors=priors,
                      store=False, filter_list=filter_list,
                      filt_dir=filt_dir)
    model_params = np.vstack((atlas['mstar'], atlas['sfr'],
                              atlas['sfh_tuple'][0:, -3:].T,
                              atlas['met'].ravel(), atlas['dust'].ravel(),
                              atlas['zval'].ravel())).T
    if Nparam == 3:
        colnames = ["Mstar", "SFR", "t25", "t50", "t75", "metal", "Av", "z"]
    else:
        raise(NotImplementedError("Column names are not defined for this "
                                  "model"))
    params = Table(model_params, names=colnames)
    seds =  Table(atlas['sed'], names=context.bands)
    table = hstack([params, seds])
    table.write(dbfile, overwrite=True)
    return

def make_simulations(N_pregrid=20000, Nparam=3, Nsim=1000):
    # Determination of error in fnu in a single pixel
    flamerr = np.loadtxt("assets/noise_90.dat", usecols=(1,), skiprows=1) * \
        context.flam_unit
    wave = np.array([context.wave_eff[band] for band in context.bands]) * \
           u.Angstrom
    idx = context.bands.index("R")
    fnuerr = flamerr / const.c * wave**2
    fnuerr = fnuerr.to(u.mJy)
    # Performing simulations
    seed = np.random.randint(10 * Nsim, size = Nsim)
    filename = os.path.join(context.tables_dir,
                         f"simulations_dense_basis_N{Nparam}_Nrepgri"
                         f"d{N_pregrid}_Nsim{Nsim}.fits")
    if os.path.exists(filename):
        table = Table.read(filename)
        nready = len(table)
    else:
        table = None
        nready = 0

    for n in tqdm(range(Nsim)):
        if n + 1 < nready:
            continue
        nid = "{:05d}".format(n+1)
        rand_sfh_tuple, rand_Z, rand_Av, rand_z = priors.sample_all_params(
            random_seed=seed[n])
        specdetails = [rand_sfh_tuple, rand_Av, rand_Z, rand_z]
        # generate an SFH corresponding to the SFH-tuple and see how it looks:
        rand_sfh, rand_time = db.tuple_to_sfh(rand_sfh_tuple, zval=rand_z)
        # fig = db.plot_sfh(rand_time, rand_sfh, lookback=True)
        # plt.tight_layout()
        # plt.show()
        sfh_truths = [rand_time, rand_sfh]
        # generate a corresponding spectrum and multiply by filter curves to get the SED:
        obs_sed = db.makespec(specdetails, priors, db.mocksp, db.cosmo,
                              filter_list=filter_list, filt_dir=filt_dir,
                              input_sfh=False)
        obs_sed *= u.Jy
        obs_sed = obs_sed.to(context.fnu_unit).value
        # store the true stellar mass and SFR
        mstar_true = np.log10(db.mocksp.stellar_mass)
        sfr_true = np.log10(db.mocksp.sfr)

        sed_truths = (mstar_true, sfr_true, rand_sfh_tuple[3:], rand_Z,
                      rand_Av, rand_z, obs_sed)
        sed_truths = np.atleast_2d(np.hstack(sed_truths))
        t = hstack([Table([[nid]], names=["ID"]),
                    Table(sed_truths, names=colnames)])
        if n == 0:
            table = t
        else:
            table = vstack([table, t])
        table.write(filename, overwrite=True)
        # sed_truths = np.hstack(sed_truths)
        # sedfit = db.SedFit(obs_sed.value, obs_err, atlas, fit_mask=[])
        # sedfit.evaluate_likelihood()
        # sedfit.evaluate_posterior_percentiles()


if __name__ == "__main__":
    N_pregrid = 200000
    Nparam = 3
    Nsim = 1000
    prepare_models(N_pregrid=N_pregrid, Nparam=Nparam)
    # make_simulations(N_pregrid=N_pregrid, Nparam=Nparam, Nsim=Nsim)