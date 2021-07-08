""" Test photometry offset between SMUDGES and S-PLUS in their respective g
bands. """

import os

import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.table import Table
from astropy.io import fits
from photutils.aperture import CircularAperture, aperture_photometry, \
    EllipticalAperture, EllipticalAnnulus
import matplotlib.pyplot as plt
from scipy.special import gamma, gammaincinv
from scipy.ndimage.filters import gaussian_filter
from astropy.stats import sigma_clipped_stats

import context

class SersicPars:
    def __init__(self, mu0, re, n, ar):
        self.mu0 = mu0
        self.re = re
        self.n = n
        self.ar = ar
        self.bn = gammaincinv(2 * self.n, 0.5)
        self.fn = self.n * np.exp(self.bn) / np.power(self.bn, 2 * n) \
                  * gamma(2 * self.n)
        self.mue = mu0 + 2.5 * self.bn / np.log(10)
        self.mean_mue = self.mue - 2.5 * np.log10(self.fn)
        self.mre = self.mean_mue - 2.5 * np.log10(np.pi * self.re**2 * self.ar)
        self.mtot = self.mean_mue - 2.5 * np.log10(2 * np.pi * self.re**2 * \
                                                    self.ar)

if __name__ == "__main__":
    tabfile = os.path.join(context.tables_dir, "DESI_UDGs_sorted_nodupsPA.csv")
    table = Table.read(tabfile)
    cubes_dir = os.path.join(context.data_dir, "cubes_dr2")
    cubenames = [_ for _ in os.listdir(cubes_dir) if _.endswith(".fits")]
    galaxies = [_.split("_")[0] for _ in cubenames]
    idx = np.array([np.where(table["Filename_corr"] == _)[0][0] for _ in
                    galaxies])
    table = table[idx]
    ref_band = "G"
    idx = context.bands.index(ref_band)
    wave = context.wave_eff[ref_band] * u.Angstrom
    results = []
    for i, (galaxy, cubename) in enumerate(zip(galaxies, cubenames)):
        # Expected magnitudes inside Re for SMUDGES
        t = table[i]
        pa = t["PA"]
        re =  t["Rearcsec"]
        ar = t["AR"]
        comp = SersicPars(t[f"mu_0_{ref_band.lower()}"], re, t["n"], ar)
        m_within_re = comp.mre
        # Performin photometry on SPLUS data
        cube = fits.getdata(os.path.join(cubes_dir, cubename), hdu=1)
        stack = cube.sum(axis=(0,))
        data = cube[idx]
        ydim, xdim = data.shape
        a = re / context.PS
        b = a * ar
        ang = np.deg2rad(-t["PA"])
        x0 = 0.5 * xdim
        y0 = 0.5 * ydim
        aperture = EllipticalAperture((x0, y0), a=a, b=b, theta=ang)
        annulus_aperture = EllipticalAnnulus((x0, y0), a_in=2.5 * a,
                                             a_out=4 * a, b_out=4 * b,
                                             theta=ang)
        annulus_masks = annulus_aperture.to_mask(method='center')
        annulus_data = annulus_masks.multiply(data)
        mask = annulus_masks.data
        annulus_data_1d = annulus_data[mask > 0]
        bkgmean, bkgmed, bkgstd = sigma_clipped_stats(annulus_data_1d)
        flam = aperture_photometry(data - bkgmed, aperture)["aperture_sum"].data[0]
        fnu = flam * context.flam_unit / const.c * wave**2
        fnu = fnu.to(context.fnu_unit).value
        mag = -2.5 * np.log10(fnu) - 48.6
        results.append([m_within_re, mag])
        smdata = gaussian_filter(data - bkgmed, 2)
        vmin = 0
        vmax = bkgstd
        # plt.imshow(smdata, vmin=vmin, vmax=vmax, cmap="viridis")
        # aperture.plot(color='r', lw=2)
        # plt.show()
    results = np.array(results)
    print(results)
    diff = np.diff(results, axis=1)
    print(np.nanmedian(diff))
    plt.plot(results[:, 0], diff, "o")
    plt.axhline(y=0, c="k", ls="--")
    plt.show()
