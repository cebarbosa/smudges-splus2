""" Get data for SMUDGES galaxies. """
import os
import getpass

import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from tqdm import tqdm
import splusdata
from splus_ifusci import SCube

import context

def get_splus_data():
    tabfile = os.path.join(context.tables_dir, "DESI_UDGs_sorted_nodupsPA.csv")
    table = Table.read(tabfile)
    outdir = os.path.join(context.data_dir, "cubes_dr2")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    # Remove galaxies with cubes already finished from the table
    galaxies = [_.split("_")[0] for _ in os.listdir(outdir) if
                  _.endswith(".fits")]
    idx = np.max([np.where(table["Filename_corr"] == _)[0][0] for _ in
                  galaxies])
    table = table[idx:]
    # Connect with S-PLUS
    username = getpass.getuser()  # Change to your S-PLUS username
    password = getpass.getpass(f"Password for {username}:")
    conn = splusdata.connect(username, password)
    for gal in tqdm(table, total=len(table)):
        name = gal["Filename_corr"]
        coords = SkyCoord(gal["RA_corr"], gal["dec_corr"], unit=u.degree)
        size = np.ceil(2 * 4.1 * gal["Rearcsec"] * u.arcsec /
                       context.ps).astype(np.int).value
        scube = SCube(name, coords, size, conn=conn,
                      coord_unit=(u.hourangle, u.degree), wdir=outdir)
        try:
            scube.download_stamps(redo=False)
        except:
            continue
        scube.make_cube(redo=False)


if __name__ == "__main__":
    get_splus_data()