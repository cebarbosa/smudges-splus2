""" Use S-PLUS data to estimate the typical S/N of observations for
simulations. """

import os
import shutil
import getpass

import numpy as np
import astropy.units as u
from astropy.table import Table, vstack
from scipy.interpolate import RectBivariateSpline
from astropy.io import fits
import astropy.constants as const
from tqdm import tqdm
import matplotlib.pyplot as plt

import context

import splusdata # To access the S-PLUS database


def get_zps(zpref="idr3_n4"):
    """ Load all tables with zero points for iDR3. """
    zp_dir = os.path.join(os.getcwd(), f"assets/{zpref}/zps")
    tables = []
    for fname in os.listdir(zp_dir):
        filename = os.path.join(zp_dir, fname)
        data = np.genfromtxt(filename, dtype=None, encoding=None)
        with open(filename) as f:
            h = f.readline().replace("#", "").replace("SPLUS_", "").split()
        table = Table(data, names=h)
        tables.append(table)
    zptable = vstack(tables)
    return zptable


def get_zp_correction():
    """ Get corrections of zero points for location in the field. """
    x0, x1, nbins = 0, 9200, 16
    xgrid = np.linspace(x0, x1, nbins + 1)
    zpcorr = {}
    for band in context.bands:
        corrfile = os.path.join(os.getcwd(),
                                f"assets/zpcorr_idr3/SPLUS_{band}_offsets_grid.npy")
        corr = np.load(corrfile)
        zpcorr[band] = RectBivariateSpline(xgrid, xgrid, corr)
    return zpcorr

def get_small_stamps():
    # Connect with S-PLUS
    username = "kadu"  # Change to your S-PLUS username
    password = getpass.getpass(f"Password for {username}:")
    conn = splusdata.connect(username, password)
    flam_unit = u.erg / u.cm / u.cm / u.s / u.AA
    fnu_unit = u.erg / u.s / u.cm / u.cm / u.Hz
    zps = get_zps()
    zpcorrs = get_zp_correction()
    tiles = Table.read("assets/tiles_new_status.csv")
    # Using only observed tiles
    tnames = tiles["NAME"].data
    fields = [_.replace("_", "-") for _ in zps["FIELD"]]
    idx = np.where(np.isin(tnames, fields))[0]
    tiles = tiles[idx]
    size = 2 / np.sqrt(2) / 2
    n_per_field = 5
    csize = 5
    outdir = os.path.join(context.data_dir, "sim_cutouts")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for tile in tqdm(tiles):
        ra0 = tile["RA_d"]
        dec0 = tile["DEC_d"]
        ras = np.random.uniform(ra0 - size, ra0 + size, n_per_field)
        decs = np.random.uniform(dec0 - size, dec0 + size,  n_per_field)
        tilename = tile["NAME"].replace("_", "-")
        outtable = os.path.join(outdir, f"{tilename}.fits")
        if os.path.exists(outtable):
            continue
        tabtile = []
        for i, (ra, dec) in enumerate(zip(ras, decs)):
            tab = Table([[tilename], [ra], [dec]], names=["TILE", "RA", "DEC"])
            for band in context.bands:
                # try:
                wave = context.wave_eff[band] * u.Angstrom
                name = f"{tile['NAME']}_{i+1:03d}_{band}"
                outname = os.path.join(outdir, f"{name}.fits")
                if os.path.exists(outname):
                    continue
                hdu = conn.get_cut_weight(ra, dec, csize, band)
                hdu1 = conn.get_cut(ra, dec, csize, band)
                h = hdu1[1].header
                weights = hdu[1].data
                # Getting zero-point calibration and location correction
                zp0 = zps[fields.index(tilename)][band]
                x0 = h["X0TILE"]
                y0 = h["Y0TILE"]
                tab["X0"] = x0
                tab["Y0"] = y0
                tab["GAIN"] = h["GAIN"]
                zpcorr = float(zpcorrs[band](x0, y0))
                zp = zp0 + zpcorr
                f0 = np.power(10, -0.4 * (48.6 + zp))
                dataerr = np.sqrt(1 / weights)
                fnuerr = dataerr * f0 * fnu_unit
                flamerr = fnuerr * const.c / wave ** 2
                flamerr = np.median(flamerr.to(flam_unit)).value
                tab[band] = [flamerr]
                # except:
                #     continue
            tabtile.append(tab)
        t = vstack(tabtile)
        t.write(outtable, format="fits", overwrite=True)

def noise_analysis():
    weights_dir = os.path.join(context.data_dir, "sim_cutouts_bk")
    filenames = os.listdir(weights_dir)
    data = []
    for filename in filenames:
        t = Table.read(os.path.join(weights_dir, filename))
        try:
            sstd = [np.median(t[band]) for band in context.bands]
        except:
            continue
        data.append(sstd)
    data = np.array(data)
    x = np.tile(np.arange(12), len(data))
    plt.hist2d(x, data.ravel(), bins=(12, 30), cmap="binary")
    noise_90 = np.percentile(data, 50, axis=0)
    wave = np.array([context.wave_eff[band] for band in context.bands]) * \
           u.Angstrom
    idx = context.bands.index("R")
    fnuerr = noise_90 * context.flam_unit / const.c * wave**2
    fnuerr = fnuerr.to(u.mJy).data
    t = Table([context.bands, noise_90], names=["FILTER", "NOISE_90"])
    t.write("assets/noise_50.dat", overwrite=True, format="ascii")
    plt.subplot(1,2,1)
    plt.plot(context.bands, np.percentile(data, 90, axis=0))
    plt.subplot(1,2,2)
    plt.plot(context.bands, fnuerr)
    plt.show()





if __name__ == "__main__":
    # get_small_stamps()
    noise_analysis()