# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:08:49 2021

@author: jeanh
"""

import numpy as np
from doubleRetrieval.util import a_b_range, uniform_prior, gaussian_prior, log_gauss, poor_mans_abunds_lbl, \
    poor_mans_abunds_ck, open_spectrum

# method used in this retrieval
# method and mode should be determined from data chosen

# MACHINE = 'rainbow'
MACHINE = 'sunray'
# MACHINE = 'bluesky'
# MACHINE = 'guenther38'

MODE = 'lbl'
LBL_SAMPLING = 10

USE_FORECASTER = False

USE_PRIOR = None

CONVERT_SINFONI_UNITS = True

USE_CHEM_DISEQU = True

MODEL = 'chem_equ'  # 'free' or 'chem_equ'
CLOUD_MODEL = 'ackermann'

WINDOW_LENGTH_lbl = 101
WINDOW_LENGTH_lbl = 64  # now referring to the length of the window for the median filter
WINDOW_LENGTH_lbl = 600  # now referring to the length of the window for the median filter, too slow
WINDOW_LENGTH_lbl = 300  # now referring to the length of the window for the only gaussian filter
WINDOW_LENGTH_lbl = 60  # now referring to the length of the window for the only gaussian filter
# WINDOW_LENGTH_lbl = 20 # now referring to the length of the window for the only gaussian filter
WINDOW_LENGTH_ck = 41
RVMIN = -400.
RVMAX = 400.
DRV = 0.5

BAYESIAN_METHOD = 'pymultinest'  # pymultinest, ultranest or mcmc

# ULTRANEST
ULTRANEST_JOBS = 10

# BAYESIAN_METHOD = 'mcmc' # doesn't work with object-oriented programming because the method (calc log-likelihood in retrievalClass) can't be pickled and therefore passed between processes, but now it works
N_WALKERS = 20
STEPSIZE = 1.75
N_THREADS = 60
N_ITER = 400
PRE_BURN_ITER = int(N_ITER / 4)

# relevant if using mcmc
CLUSTER = True

RETRIEVAL_NOTES = [
    'TESTING THE CODE'
]

# name of retrieval run
USE_WEIGHTS = False

NUMBER = ''
RETRIEVAL_NAME_INPUT = 'SPHERE_GRAVITY_ONE_a_06_v01_NO_NOISE_v01'
VERSION = 'retrieval'
if MODEL == 'free':
    VERSION += '_free'
RETRIEVAL_NAME = RETRIEVAL_NAME_INPUT + '_CHEM_EQU_CROCO'
# configure the paths of the input and output files
IPAGATE_ROUTE = '/home/ipa/quanz/user_accounts/jhayoz/Projects/'
INPUT_DIR = IPAGATE_ROUTE + 'MT_paper/PAPER_RETRIEVALS/' + RETRIEVAL_NAME_INPUT

OUTPUT_DIR = '/scratch/'
if MACHINE == 'rainbow':
    OUTPUT_DIR += 'user/'
OUTPUT_DIR += 'jhayoz/RunningJobs/' + RETRIEVAL_NAME + '_' + VERSION + '/'

SIM_DATA_DIR = INPUT_DIR
CC_DATA_FILE = INPUT_DIR + '/CC_spectrum'
RES_DATA_FILE = INPUT_DIR + '/RES_spectrum'
RES_ERR_FILE = INPUT_DIR + '/RES_error'

USE_SIM_DATA = ['CC', 'RES', 'PHOT']  # ['CC','RES','PHOT']

# Name of files for respective data. If don't want to use one, write None

PHOT_DATA_FILE = IPAGATE_ROUTE + 'retrieval_input/PHOT_data'
if 'PHOT' not in USE_SIM_DATA:
    PHOT_DATA_FILE = None
PHOT_DATA_FILTER_FILE, PHOT_DATA_FLUX_FILE = None, None
if 'PHOT' in USE_SIM_DATA:
    PHOT_DATA_FILTER_FILE = PHOT_DATA_FILE + '/filter'
    PHOT_DATA_FLUX_FILE = PHOT_DATA_FILE + '/flux'
    PHOT_DATA_FLUX_FILE = INPUT_DIR

# Beta Pictoris b filepaths
# SIM_DATA_DIR = None
# CC_DATA_FILE = '/scratch/jhayoz/retrieval_data/spectrum/Sinfoni_spectrum_v3.txt'
# RES_DATA_FILE = '/scratch/jhayoz/retrieval_data/GRAVITY/GRAVITY_spectrum.txt'
# RES_ERR_FILE = '/scratch/jhayoz/retrieval_data/GRAVITY/GRAVITY_cov_matrix.txt'
# PHOT_DATA_FLUX_FILE = '/scratch/jhayoz/retrieval_data/photometric_data'
# PHOT_DATA_FILTER_FILE = '/scratch/jhayoz/retrieval_data/filter_transmission_function'

# diagnostic parameters

WRITE_THRESHOLD = 50

PRINTING = True
PLOTTING = False
TIMING = True
SHOW_REF_VALUES = True
PLOTTING_THRESHOLD = 50

# configure parameter space


MOL_ABUNDS_KEYS_CK = poor_mans_abunds_ck()
MOL_ABUNDS_KEYS_LBL = poor_mans_abunds_lbl()

# ABUNDANCES = ['H2O_main_iso','CO_main_iso','CH4_main_iso']
if MODEL == 'free':
    # ABUNDANCES = ['H2O_main_iso','CH4_main_iso', 'CO_main_iso', 'CO2_main_iso','H2S_main_iso']
    ABUNDANCES = ['H2O_main_iso', 'CH4_main_iso', 'CO_main_iso', 'CO2_main_iso', 'H2S_main_iso', 'FeH_main_iso',
                  'TiO_all_iso', 'K', 'VO']
    ABUNDANCES = ['H2O_main_iso', 'CO_main_iso', 'CH4_main_iso', 'CO2_main_iso', 'H2S_main_iso', 'FeH_main_iso',
                  'HCN_main_iso', 'TiO_all_iso', 'K', 'VO']
    # ABUNDANCES = ['H2O_main_iso','CH4_main_iso', 'CO_main_iso', 'CO2_main_iso','H2S_main_iso','FeH_main_iso','TiO_all_iso','K']
    # ABUNDANCES = ['H2O_main_iso', 'CO_main_iso','FeH_main_iso','VO']
    # ABUNDANCES = ['H2O_main_iso', 'CO_main_iso']
    # ABUNDANCES = [ 'H2O_main_iso','CO_main_iso','CH4_main_iso','CO2_main_iso','HCN_main_iso','FeH_main_iso','H2S_main_iso','NH3_main_iso','TiO_all_iso','VO']
    # ABUNDANCES = ['H2O_main_iso','CH4_main_iso', 'CO_main_iso']
if MODEL == 'chem_equ':
    ABUNDANCES = ['C/O', 'FeHs']

UNSEARCHED_ABUNDS = []

if USE_FORECASTER or USE_PRIOR is not None:
    ALL_TEMPS = ['log_gamma', 't_int', 't_equ', 'log_kappa_IR', 'log_R', 'P0', 'log_gravity']
    TEMP_PARAMS = ['t_equ', 'log_R', 'log_gravity'
                   ]  # Pick from ALL_TEMPS, and order is relevant: must be like in ALL_TEMPS
else:
    ALL_TEMPS = ['log_gamma', 't_int', 't_equ', 'log_gravity', 'log_kappa_IR', 'R', 'P0']
    TEMP_PARAMS = ['t_equ', 'log_gravity', 'R'
                   ]  # Pick from ALL_TEMPS, and order is relevant: must be like in ALL_TEMPS

UNSEARCHED_TEMPS = [item for item in ALL_TEMPS if not (item in TEMP_PARAMS)]

# CLOUDS_OPACITIES = ['MgSiO3(c)_cm']
# ALL_CLOUDS = ['kzz','fsed','sigma_lnorm','MgSiO3(c)']
# CLOUDS = ['kzz','fsed','MgSiO3(c)']
CLOUDS = []
UNSEARCHED_CLOUDS = []

PARAMS = [TEMP_PARAMS, ABUNDANCES, CLOUDS]
PARAMS_NAMES = TEMP_PARAMS + ABUNDANCES + CLOUDS
UNSEARCHED_PARAMS = [UNSEARCHED_TEMPS, UNSEARCHED_ABUNDS, UNSEARCHED_CLOUDS]
ALL_PARAMS = TEMP_PARAMS + ABUNDANCES + CLOUDS + UNSEARCHED_TEMPS + UNSEARCHED_ABUNDS + UNSEARCHED_CLOUDS

NEEDED_LINE_SPECIES = ABUNDANCES + UNSEARCHED_ABUNDS

if MODEL == 'chem_equ':
    NEEDED_LINE_SPECIES += MOL_ABUNDS_KEYS_LBL
    if 'C/O' in NEEDED_LINE_SPECIES:
        NEEDED_LINE_SPECIES.remove('C/O')
    if 'FeHs' in NEEDED_LINE_SPECIES:
        NEEDED_LINE_SPECIES.remove('FeHs')

# enter reference values here (if data was simulated, their true values)

pc_to_m = 3.086 * 1e16
distance_pc = 19.7538
DISTANCE = distance_pc * pc_to_m

RADIUS_J = 69911 * 1000
MASS_J = 1.898 * 1e27

DATA_PARAMS = {}

DATA_PARAMS['log_gamma'] = np.log10(0.4)
DATA_PARAMS['t_int'] = 200.
DATA_PARAMS['t_equ'] = 1742
DATA_PARAMS['log_kappa_IR'] = np.log10(0.01)
DATA_PARAMS['R'] = 1.36
# DATA_PARAMS['log_gravity']    = 4.35
DATA_PARAMS['log_gravity'] = 4.35
DATA_PARAMS['P0'] = 2

if USE_FORECASTER:
    from doubleRetrieval.util import sample_to_pdf, skew_gauss_pdf, fit_skewed_gauss, skew_gauss_ppf

    sample_prior = {}
    for name in ['mass', 'radius', 'log_g']:
        try:
            sample_prior[name] = open_spectrum(INPUT_DIR + '/' + name + '_sample_prior.txt')
        except IOError:
            print('WE DIDNT FIND NO SAMPLE TO CALCULATE PRIORS FROM')
        print(np.shape(sample_prior[name]))
    print(sample_prior.keys())

    popt_param = {}
    print('UPDATING DATA PARAMS')
    # mass
    DATA_PARAMS['log_M'] = np.log10(np.median(sample_prior['mass']))
    mass_pdf = sample_to_pdf(sample_prior['mass'])
    mass_pos = np.linspace(min(sample_prior['mass']), max(sample_prior['mass']), 1000)
    popt_param['log_M'], pcov = fit_skewed_gauss(mass_pos, mass_pdf(mass_pos))

    # radius
    DATA_PARAMS['R'] = np.log10(np.median(sample_prior['radius']))
    radius_pdf = sample_to_pdf(sample_prior['radius'])
    radius_pos = np.linspace(min(sample_prior['radius']), max(sample_prior['radius']), 1000)
    popt_param['R'], pcov = fit_skewed_gauss(radius_pos, radius_pdf(radius_pos))

    # log_g
    DATA_PARAMS['log_gravity'] = np.median(sample_prior['log_g'])
    log_g_pdf = sample_to_pdf(sample_prior['log_g'])
    log_g_pos = np.linspace(min(sample_prior['log_g']), max(sample_prior['log_g']), 1000)
    popt_param['log_gravity'], pcov = fit_skewed_gauss(log_g_pos, log_g_pdf(log_g_pos))
if USE_PRIOR is not None:
    print('USING PRIORS')
    from doubleRetrieval.util import sample_to_pdf, fit_gauss, gauss_pdf, gauss_ppf

    try:
        sample_prior = open_spectrum(INPUT_DIR + '/' + USE_PRIOR + '_sample_prior.txt')
    except IOError:
        print('WE DIDNT FIND NO SAMPLE TO CALCULATE PRIORS FROM')
    print(np.shape(sample_prior))
    print('UPDATING DATA PARAMS')

    # DATA_PARAMS[USE_PRIOR] = np.median(sample_prior)
    if USE_PRIOR == 'R':
        print(USE_PRIOR, ' = ', DATA_PARAMS['log_R'])
    quantity_pdf = sample_to_pdf(sample_prior)
    quantity_pos = np.linspace(min(sample_prior), max(sample_prior), 1000)
    popt_param, pcov = fit_gauss(quantity_pos, quantity_pdf(quantity_pos))

# DATA_PARAMS['H2O_main_iso']   = -2.51
# DATA_PARAMS['CO_main_iso']    = -2.82
# DATA_PARAMS['CH4_main_iso']   = -5.18
# DATA_PARAMS['H2O_main_iso']   = -3.1
# DATA_PARAMS['CO_main_iso']    = -3.3
# DATA_PARAMS['CH4_main_iso']   = -4.5
# DATA_PARAMS['CO2_main_iso']   = -4.2
# DATA_PARAMS['H2S_main_iso']   = -4.2
DATA_PARAMS['C/O'] = 0.44
DATA_PARAMS['FeHs'] = -1

# DATA_PARAMS['kzz'] = 7.5
# DATA_PARAMS['fsed'] = 2
# DATA_PARAMS['sigma_lnorm'] = 1.05
# DATA_PARAMS['cloud_abunds'] = {}
# DATA_PARAMS['cloud_abunds']['MgSiO3(c)'] = -4


# DATA_PARAMS['log_Pcloud']     = np.log10(0.5)

# configure prior distributions

# define range of uniform distributions

RANGE = {}
RANGE['log_gamma'] = [-4, 0]
RANGE['t_int'] = [0, 1000]
RANGE['t_equ'] = [0, 5000]
RANGE['log_gravity'] = [1, 8]  # [-2,10]
RANGE['log_kappa_IR'] = [-5, 0]
# RANGE['log_R']          = [DATA_PARAMS['log_R']-0.5,DATA_PARAMS['log_R']+0.5]
RANGE['R'] = [0.1, 10]
RANGE['P0'] = [-2, 2]
RANGE['log_Pcloud'] = [-3, 1.49]
RANGE['abundances'] = [-10, 0]
RANGE['C/O'] = [0.1, 1.6]
RANGE['FeHs'] = [-2, 3]

# true prior distribution, used as sanity check before calculating likelihood

LOG_PRIORS = {}
LOG_PRIORS['log_gamma'] = lambda x: a_b_range(x, RANGE['log_gamma'])
LOG_PRIORS['t_int'] = lambda x: a_b_range(x, RANGE['t_int'])
LOG_PRIORS['t_equ'] = lambda x: a_b_range(x, RANGE['t_equ'])
LOG_PRIORS['log_gravity'] = lambda x: a_b_range(x, RANGE['log_gravity'])
LOG_PRIORS['log_kappa_IR'] = lambda x: a_b_range(x, RANGE['log_kappa_IR'])
LOG_PRIORS['R'] = lambda x: a_b_range(x, RANGE['R'])
# LOG_PRIORS['R']          = lambda x: log_gauss(x,DATA_PARAMS['R'],0.5)
LOG_PRIORS['P0'] = lambda x: a_b_range(x, RANGE['P0'])
LOG_PRIORS['log_Pcloud'] = lambda x: a_b_range(x, RANGE['log_Pcloud'])
for name in ABUNDANCES:
    LOG_PRIORS[name] = lambda x: a_b_range(x, RANGE['abundances'])
LOG_PRIORS['C/O'] = lambda x: a_b_range(x, RANGE['C/O'])
LOG_PRIORS['FeHs'] = lambda x: a_b_range(x, RANGE['FeHs'])
# transformation of the unit cube to correspond to prior distribution

CUBE_PRIORS = {}
CUBE_PRIORS['log_gamma'] = lambda x: uniform_prior(x, RANGE['log_gamma'])
CUBE_PRIORS['t_int'] = lambda x: uniform_prior(x, RANGE['t_int'])
CUBE_PRIORS['t_equ'] = lambda x: uniform_prior(x, RANGE['t_equ'])
CUBE_PRIORS['log_gravity'] = lambda x: uniform_prior(x, RANGE['log_gravity'])
CUBE_PRIORS['log_kappa_IR'] = lambda x: uniform_prior(x, RANGE['log_kappa_IR'])
CUBE_PRIORS['R'] = lambda x: uniform_prior(x, RANGE['R'])
# CUBE_PRIORS['R']          = lambda x: gaussian_prior(x,DATA_PARAMS['R'],0.5)
CUBE_PRIORS['P0'] = lambda x: uniform_prior(x, RANGE['P0'])
CUBE_PRIORS['log_Pcloud'] = lambda x: uniform_prior(x, RANGE['log_Pcloud'])
for name in ABUNDANCES:
    CUBE_PRIORS[name] = lambda x: uniform_prior(x, RANGE['abundances'])
CUBE_PRIORS['C/O'] = lambda x: uniform_prior(x, RANGE['C/O'])
CUBE_PRIORS['FeHs'] = lambda x: uniform_prior(x, RANGE['FeHs'])

if USE_FORECASTER:
    print('Creating priors with Forecaster')
    RANGE['log_R'] = [-2, 2]
    RANGE['log_M'] = [-2, 2]

    LOG_PRIORS['log_R'] = lambda x: skew_gauss_pdf(10 ** x, *popt_param['log_R'])
    LOG_PRIORS['log_M'] = lambda x: skew_gauss_pdf(10 ** x, *popt_param['log_M'])

    CUBE_PRIORS['log_R'] = lambda x: np.log10(skew_gauss_ppf(x, *popt_param['log_R']))
    CUBE_PRIORS['log_M'] = lambda x: np.log10(skew_gauss_ppf(x, *popt_param['log_M']))
if USE_PRIOR is not None:
    print('Creating priors on Mass')
    if USE_PRIOR == 'M':
        RANGE['M'] = [1, 30]
        LOG_PRIORS['M'] = lambda x: gauss_pdf(x, *popt_param)
        CUBE_PRIORS['M'] = lambda x: gauss_ppf(x, *popt_param)
    elif USE_PRIOR == 'R':
        RANGE['log_R'] = [0.1, 3]
        LOG_PRIORS['log_R'] = lambda x: gauss_pdf(x, *popt_param)
        CUBE_PRIORS['log_R'] = lambda x: gauss_ppf(x, *popt_param)

CONFIG_DICT = {}
CONFIG_DICT['MODEL'] = MODEL
CONFIG_DICT['MODE'] = MODE
CONFIG_DICT['USE_FORECASTER'] = USE_FORECASTER
CONFIG_DICT['USE_PRIOR'] = USE_PRIOR
CONFIG_DICT['ALL_PARAMS'] = ALL_PARAMS
CONFIG_DICT['ABUNDANCES'] = ABUNDANCES
CONFIG_DICT['NEEDED_LINE_SPECIES'] = NEEDED_LINE_SPECIES
CONFIG_DICT['TEMPS'] = TEMP_PARAMS
CONFIG_DICT['CLOUDS'] = CLOUDS
CONFIG_DICT['UNSEARCHED_ABUNDANCES'] = UNSEARCHED_ABUNDS
CONFIG_DICT['UNSEARCHED_TEMPS'] = UNSEARCHED_TEMPS
CONFIG_DICT['UNSEARCHED_CLOUDS'] = UNSEARCHED_CLOUDS
CONFIG_DICT['PARAMS_NAMES'] = PARAMS_NAMES
CONFIG_DICT['UNSEARCHED_PARAMS'] = UNSEARCHED_PARAMS
CONFIG_DICT['RVMAX'] = RVMAX
CONFIG_DICT['RVMIN'] = RVMIN
CONFIG_DICT['DRV'] = DRV
CONFIG_DICT['DISTANCE'] = DISTANCE
CONFIG_DICT['WIN_LEN'] = WINDOW_LENGTH_lbl
CONFIG_DICT['LBL_SAMPLING'] = LBL_SAMPLING
CONFIG_DICT['CONVERT_SINFONI_UNITS'] = CONVERT_SINFONI_UNITS
CONFIG_DICT['WRITE_THRESHOLD'] = WRITE_THRESHOLD
CONFIG_DICT['PLOTTING_THRESHOLD'] = PLOTTING_THRESHOLD
CONFIG_DICT['DATA_PARAMS'] = DATA_PARAMS