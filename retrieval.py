# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 15:32:01 2021

@author: jeanh
"""
# Main code to run an atmospheric retrieval

# import all relevant Python libraries
print('IMPORTING LIBRARIES')
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ["pRT_input_data_path"] = "/home/ipa/quanz/shared/petitRADTRANS/input_data"

from os import path

sys.path.append("/home/ipa/quanz/shared/petitRADTRANS/")
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc
import pickle
import json
import scipy.stats, scipy

# import all modules
from doubleRetrieval.util import *
from doubleRetrieval.priors import Prior
from doubleRetrieval.data2 import Data
from doubleRetrieval.retrievalClass3 import Retrieval
from doubleRetrieval.plotting import *
from config_retrieval import *

print('    DONE')

import pymultinest

if not os.path.exists(OUTPUT_DIR):
    try:
        os.mkdir(OUTPUT_DIR)
    except FileExistsError:
        print('Avoided error')

import shutil

cwd = os.getcwd()
source = cwd + '/config3.py'
destination = OUTPUT_DIR + 'config3_copy.py'
shutil.copyfile(source, destination)
print('Config file copied')

with open(OUTPUT_DIR + 'NOTES.txt', 'w') as f:
    for line in RETRIEVAL_NOTES:
        f.write(line + '\n')

# create data, prior, and retrieval class

data_obj = Data(data_dir=None,
                use_sim_files=USE_SIM_DATA,
                CC_data_dir=CC_DATA_FILE)

data_obj.plot(CONFIG_DICT, OUTPUT_DIR + 'data')

# Check that the retrieval does what I want it to do
if 'CC' in USE_SIM_DATA:
    assert (data_obj.CCinDATA())
    wlen_data_temp, flux_data_temp = data_obj.getCCSpectrum()
    assert (len(wlen_data_temp.keys()) == 1 or len(wlen_data_temp.keys()) > 5)


print('Check passed')

prior_obj = Prior(RANGE, LOG_PRIORS, CUBE_PRIORS)
prior_obj.plot(CONFIG_DICT, OUTPUT_DIR)

print(MODEL)
retrieval = Retrieval(
    data_obj,
    prior_obj,
    config=CONFIG_DICT,
    model=MODEL,  # free or chem_equ
    retrieval_name=RETRIEVAL_NAME,
    output_path=OUTPUT_DIR,
    plotting=PLOTTING,
    printing=PRINTING,
    timing=TIMING)

print('Starting Bayesian inference')

n_params = len(PARAMS_NAMES)

pymultinest.run(retrieval.lnprob_pymultinest,
                retrieval.Prior,
                n_params,
                outputfiles_basename=OUTPUT_DIR,
                resume=False,
                verbose=True)
# save positions
json.dump(PARAMS_NAMES, open(OUTPUT_DIR + 'params.json', 'w'))

# create analyzer object
a = pymultinest.Analyzer(n_params, outputfiles_basename=OUTPUT_DIR)

stats = a.get_stats()
bestfit_params = a.get_best_fit()
samples = np.array(a.get_equal_weighted_posterior())[:, :-1]

f = open(OUTPUT_DIR + 'SAMPLESpos.pickle', 'wb')
pickle.dump(samples, f)
f.close()

nb_positions = len(samples)
percent_considered = 1.

if not path.exists(OUTPUT_DIR + 'cornerplot'):
    plot_corner(CONFIG_DICT,
                samples,
                param_range=None,
                percent_considered=percent_considered,
                output_file=OUTPUT_DIR,
                title='Corner plot of ' + RETRIEVAL_NAME + ' ' + VERSION
                )

if not path.exists(OUTPUT_DIR + 'plot_retrieved_spectrum'):
    try:
        wlen_CC, flux_CC, wlen_RES, flux_RES, photometry = plot_retrieved_spectra_FM_dico(
            retrieval,
            samples,
            output_file=OUTPUT_DIR,
            title='Retrieved spectrum for ' + RETRIEVAL_NAME + ' ' + VERSION,
            show_random=None,
            output_results=True)
    except:
        pass

if not path.exists(OUTPUT_DIR + 'CC_function'):
    if data_obj.CCinDATA():
        CC_wlen_data, CC_flux_data = data_obj.getCCSpectrum()
        try:
            plot_SNR(CONFIG_DICT, wlen_CC, flux_CC, CC_wlen_data, CC_flux_data, output_file=OUTPUT_DIR,
                     title='C-C function for ' + RETRIEVAL_NAME + ' ' + VERSION, printing=True)
        except:
            pass

