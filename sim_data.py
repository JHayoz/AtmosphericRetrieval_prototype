import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import os

os.environ["pRT_input_data_path"] = "/home/ipa/quanz/shared/petitRADTRANS/input_data"
sys.path.append("/home/ipa/quanz/shared/petitRADTRANS/")
from petitRADTRANS import radtrans as rt
from petitRADTRANS import nat_cst as nc

from doubleRetrieval.forward_model import ForwardModel
from doubleRetrieval.plotting import plot_data, plot_profiles
from doubleRetrieval.util import convert_units, save_photometry, save_spectrum, trim_spectrum, calc_cov_matrix, \
    save_spectra, save_lines
from doubleRetrieval.rebin import rebin_to_CC, rebin, doppler_shift, rebin_to_RES, rebin_to_PHOT, only_gaussian_filter
from doubleRetrieval.data2 import Data
from doubleRetrieval.rotbroad_utils import add_rot_broad, cut_spectrum
from config_sim import *

import csv
import pickle
import os

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

import shutil

cwd = os.getcwd()
source = cwd + '/config_sim.py'
destination = OUTPUT_DIR + '/config_sim.py'
shutil.copyfile(source, destination)
print('Config file copied')

# redesign how to call data object for simulation
# idea: import real data, and copy format

resolution = RESOLUTION
wlen_borders = WLEN_BORDERS
stepsize = (wlen_borders[0]+wlen_borders[1])/2/resolution
N_points = int((wlen_borders[1]-wlen_borders[0])/stepsize)
wlen_data = np.linspace(wlen_borders[0],wlen_borders[1],N_points)

instruments = {}
instruments['SINFONI'] = wlen_data

forwardmodel_lbl = ForwardModel(
    wlen_borders=wlen_borders,
    max_wlen_stepsize=stepsize,
    mode='lbl',
    line_opacities=ABUNDANCES,
    chem_model=MODEL,
    max_RV=0,
    max_winlen=WINDOW_LENGTH_LBL
)
forwardmodel_lbl.calc_rt_obj(lbl_sampling=LBL_SAMPLING)

wlen_lbl, flux_lbl, pressures, temperatures, abundances = forwardmodel_lbl.calc_spectrum(
    ab_metals=AB_METALS,
    temp_params=TEMP_PARAMS,
    clouds_params=CLOUDS_PARAMS,
    external_pt_profile=None,
    return_profiles=True)

plot_profiles(
    pressures,
    temperatures=temperatures,
    abundances=abundances,
    output_dir=OUTPUT_DIR,
    fontsize=20)

CC_wlen_shifted, CC_flux_shifted = doppler_shift(wlen_lbl, flux_lbl, RV)
CC_wlen_removed, CC_flux_removed, CC_wlen_rebin, CC_flux_rebin, sgfilter = {}, {}, {}, {}, {}

if CONVERT_SINFONI_UNITS:
    CC_wlen_shifted, CC_flux_shifted = convert_units(CC_wlen_shifted, CC_flux_shifted, TEMP_PARAMS['log_R'], DISTANCE)
else:
    CC_wlen_shifted = 1e4 * CC_wlen_shifted

for key in instruments.keys():
    # rebinning
    CC_wlen_rebin[key], CC_flux_rebin[key] = rebin(CC_wlen_shifted, CC_flux_shifted, instruments[key])

    if PHYSICAL_PARAMS['rot_vel'] != 0:
        CC_wlen_rebin[key], CC_flux_rebin[key] = add_rot_broad(CC_wlen_rebin[key], CC_flux_rebin[key],
                                                               PHYSICAL_PARAMS['rot_vel'], method='fast',
                                                               edgeHandling='cut')

    # remove continuum with filter
    wlen_after = None
    if OUTPUT_FORMAT == 'datalike':
        wlen_after = instruments[key]
    CC_wlen_removed[key], CC_flux_removed[key], sgfilter[key] = only_gaussian_filter(CC_wlen_rebin[key],
                                                                                     CC_flux_rebin[key],
                                                                                     sigma=WINDOW_LENGTH_LBL,
                                                                                     wlen_after=wlen_after)

save_spectra(CC_wlen_removed, CC_flux_removed, save_dir=OUTPUT_DIR + '/CC_spectrum', save_name='')
plot_data(config=CONFIG_DICT,
              CC_wlen = CC_wlen_removed,
              CC_flux = CC_flux_removed,
              CC_wlen_w_cont = CC_wlen_rebin,
              CC_flux_w_cont = CC_flux_rebin,
              sgfilter = sgfilter,
              inset_plot = False,
              output_file = OUTPUT_DIR,
              plot_name='plot',
              title = 'Spectrum',
              fontsize=15)