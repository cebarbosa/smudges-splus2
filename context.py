# -*- coding: utf-8 -*-
"""

Created on 22/08/19

Author : Carlos Eduardo Barbosa

"""
from __future__ import print_function, division

import os
import platform
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
from dustmaps.config import config
from dustmaps import sfd

project_name = "smudges2"
home = str(Path.home())
if platform.node() == "kadu-Inspiron-5557":
    home_dir = f"/home/kadu/Dropbox/{project_name}"
elif platform.node() in ["uv100", "alphacrucis"]:
    home_dir = f"/sto/home/cebarbosa/{project_name}"
elif home == "/home/u11/cebarbosa":
    home_dir = os.path.join(home, projec_tname)

tables_dir = os.path.join(home_dir, "tables")
data_dir = os.path.join(home_dir, "data")
plots_dir = os.path.join(home_dir, "plots")

config['data_dir'] = os.path.join(data_dir, "dustmaps")
if not os.path.exists(config["data_dir"]): # Just to run once in my example
    sfd.fetch() # Specific for Schlafy and Finkbeiner (2011), which is an
    # updated version of the popular Schlegel, Finkbeiner & Davis (1998) maps

bands = ["F378", "F395", "F410", "F430", "F515", "F660", "F861", "G",
         "I", "R", "U", "Z"]

ps = 0.55 * u.arcsec / u.pixel
PS = 0.55
gain = 0.95
expsingle = {"U": 227, "F378": 220, "F395": 118, "F410": 59, "F430": 57,
             "G": 33, "F515": 61, "R": 40, "F660": 290, "I": 46,
             "F861": 80, "Z": 56}


bands = ['U', 'F378', 'F395', 'F410', 'F430', 'G', 'F515', 'R', 'F660', 'I',
         'F861', 'Z']

narrow_bands = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']

broad_band = ['U', 'G', 'R', 'I', 'Z']

bands_names = {'U' : "$u$", 'F378': "$J378$", 'F395' : "$J395$",
               'F410' : "$J410$", 'F430' : "$J430$", 'G' : "$g$",
               'F515' : "$J515$", 'R' : "$r$", 'F660' : "$J660$",
               'I' : "$i$", 'F861' : "$J861$", 'Z' : "$z$"}

wave_eff = {"F378": 3773.4, "F395": 3940.8, "F410": 4095.4,
            "F430": 4292.5, "F515": 5133.5, "F660": 6614.0, "F861": 8608.1,
            "G": 4647.8, "I": 7683.8, "R": 6266.6, "U": 3536.5,
            "Z": 8679.5}




# Matplotlib settings
plt.style.context("seaborn-paper")
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.serif'] = 'Computer Modern'
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# set tick width
width = 0.5
majsize = 4
minsize = 2
plt.rcParams['xtick.major.size'] = majsize
plt.rcParams['xtick.major.width'] = width
plt.rcParams['xtick.minor.size'] = minsize
plt.rcParams['xtick.minor.width'] = width
plt.rcParams['ytick.major.size'] = majsize
plt.rcParams['ytick.major.width'] = width
plt.rcParams['ytick.minor.size'] = minsize
plt.rcParams['ytick.minor.width'] = width
plt.rcParams['axes.linewidth'] = width

fig_width = 3.35 # inches

flam_unit = u.erg / u.cm / u.cm / u.s / u.AA
fnu_unit = u.erg / u.s / u.cm / u.cm / u.Hz