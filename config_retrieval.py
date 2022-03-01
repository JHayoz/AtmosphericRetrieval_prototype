# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:08:49 2021

@author: jeanh
"""

import numpy as np
from doubleRetrieval.util import a_b_range, uniform_prior, gaussian_prior, log_gauss, poor_mans_abunds_lbl, \
    poor_mans_abunds_ck, open_spectrum


MODEL = 'free'  # 'free' or 'chem_equ'
MACHINE = 'bluesky'

NUMBER = ''
RETRIEVAL_NAME_INPUT = 'spectrum_v01'
VERSION = '01'
if MODEL == 'free':
    VERSION += '_free'
RETRIEVAL_NAME = 'first_retrieval'
# configure the paths of the input and output files

USER = 'jhayoz'
USER_FOLDER = '/Projects/student_project/fabian/'
IPAGATE_ROUTE = '/home/' + USER + USER_FOLDER
INPUT_DIR = IPAGATE_ROUTE + RETRIEVAL_NAME_INPUT

OUTPUT_DIR = '/scratch/'
OUTPUT_DIR += RETRIEVAL_NAME + '_' + VERSION + '/'

SIM_DATA_DIR = INPUT_DIR
CC_DATA_FILE = INPUT_DIR + '/CC_spectrum'

USE_SIM_DATA = ['CC']  # ['CC','RES','PHOT']

RETRIEVAL_NOTES = [
    'TESTING THE CODE'
]

# method used in this retrieval
# method and mode should be determined from data chosen


MODE = 'lbl'
LBL_SAMPLING = 10

CONVERT_SINFONI_UNITS = True

USE_CHEM_DISEQU = True

WINDOW_LENGTH_lbl = 60  # now referring to the length of the window for the only gaussian filter
WINDOW_LENGTH_ck = 41

RVMIN = -500.
RVMAX = 500.
DRV = 0.5

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

if MODEL == 'free':
    ABUNDANCES = ['H2O_main_iso']
if MODEL == 'chem_equ':
    ABUNDANCES = ['C/O', 'FeHs']

UNSEARCHED_ABUNDS = []


ALL_TEMPS = ['log_gamma', 't_int', 't_equ', 'log_gravity', 'log_kappa_IR', 'R', 'P0']
TEMP_PARAMS = ['t_equ',
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
DATA_PARAMS['log_gravity'] = 4.35
DATA_PARAMS['P0'] = 2

# DATA_PARAMS['H2O_main_iso']   = -2.51
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
CUBE_PRIORS['P0'] = lambda x: uniform_prior(x, RANGE['P0'])
CUBE_PRIORS['log_Pcloud'] = lambda x: uniform_prior(x, RANGE['log_Pcloud'])
for name in ABUNDANCES:
    CUBE_PRIORS[name] = lambda x: uniform_prior(x, RANGE['abundances'])
CUBE_PRIORS['C/O'] = lambda x: uniform_prior(x, RANGE['C/O'])
CUBE_PRIORS['FeHs'] = lambda x: uniform_prior(x, RANGE['FeHs'])

CONFIG_DICT = {}
CONFIG_DICT['MODEL'] = MODEL
CONFIG_DICT['MODE'] = MODE
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