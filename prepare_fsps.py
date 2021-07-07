""" Prepare templates for SED fitting. """
import os

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table, vstack
import speclite.filters
from speclite.redshift import redshift

import context

def load_java_system(bands=None, version="dr1"):
    """ Use speclite to load Javalambre's SPLUS filters for convolution. """
    bands = context.bands if bands is None else bands
    wdir = os.path.join(os.path.split(__file__)[0], "tables")
    filters_dir = os.path.join(wdir, "filter_curves", version)
    ftable = ascii.read(os.path.join(wdir, "filter_lam_filename.txt"))
    filternames = []
    for f in ftable:
        if f["filter"] not in bands:
            continue
        filtname = f["filter"]
        fname = os.path.join(filters_dir, f["filename"])
        fdata = np.loadtxt(fname)
        fdata = np.vstack(((fdata[0,0]-1, 0), fdata, (fdata[-1,0]+1, 0)))
        w = fdata[:,0] * u.AA
        response = np.clip(fdata[:,1], 0., 1.)
        speclite.filters.FilterResponse(wavelength=w,
                  response=response, meta=dict(group_name="java",
                  band_name=filtname))
        filternames.append("java-{}".format(filtname))
    java = speclite.filters.load_filters(*filternames)
    return java

def prepare_FSPS(emission=True):
    """ Prepare FSPS models for SED fitting with S-PLUS. """
    fsps_dir = "/home/kadu/Dropbox/SSPs/FSPS"
    templates_dir = os.path.join(fsps_dir, "varydoublex")
    if emission is True:
        templates_dir = f"{templates_dir}_with_lines"
    # Fixing slope of IMF to a Kroupa IMF
    x1 = "imf1p1.30"
    x2 = "imf2p2.30"
    filenames = [_ for _ in os.listdir(templates_dir)
                 if _.endswith(f"{x1}_{x2}.fits")]
    data = []
    table = []
    for filename in filenames:
        fpath = os.path.join(templates_dir, filename)
        d = fits.getdata(fpath)
        t = Table.read(fpath, hdu=1)
        wave = Table.read(fpath, hdu=2)
        data.append(d)
        table.append(t)
    table = vstack(table)
    data = np.vstack(data)
    for z in np.linspace(0, 0.1, 101):
        rules = [dict(name='wave', exponent=+1, array_in=wave["wave"].data),
                 dict(name='flux', exponent=-1, array_in=data[0])]
        result = redshift(z_in=0, z_out=z, rules=rules)
        print(result)
        input()



if __name__ == "__main__":
    prepare_FSPS(emission=True)