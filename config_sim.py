# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 15:27:11 2021

@author: jeanh
"""

import numpy as np
import scipy.constants as cst
from doubleRetrieval.util import save_1d_data

NAME = 'spectrum'
VERSION = '02'
NAME += '_v' + VERSION

MODEL = 'free'  # 'free' or 'chem_equ'
OUTPUT_FORMAT = 'datalike'  # determines whether output has same bins as data ('datalike') or if it still contains its extensions
INSTRUMENT = 'SINFONI'

USE_PRIOR = None

USER = 'jhayoz'
USER_FOLDER = '/Projects/student_project/fabian/'
IPAGATE_ROUTE = '/home/' + USER + USER_FOLDER

OUTPUT_DIR = IPAGATE_ROUTE + '/Student_projects'
OUTPUT_DIR += NAME
OUTPUT_DIR = '/home/jhayoz/Projects/student_project/fabian/' + NAME

# where to get data from to make synthetic data
INPUT_FILE = IPAGATE_ROUTE + ''

CC_WVL_DIR = INPUT_FILE + 'CC_data/CC_SINFONI'

RES_WVL_DIR = INPUT_FILE + 'RES_data/Flux'
RES_COV_DIR = INPUT_FILE + 'RES_data/Error'

FILTER_DIR = INPUT_FILE + 'PHOT_data/filter/'
PHOT_DATA_FILE = INPUT_FILE + 'PHOT_data/flux/'

EXTERNAL_PROFILES = False

MODE = 'lbl'
LBL_SAMPLING = None
CONVERT_SINFONI_UNITS = True

RESOLUTION = 5000

#WLEN_BORDERS = [2.0880364682002703, 2.4506398060442036]
WLEN_BORDERS = [1, 3]
EXTENSIONS = None

ABUNDANCES = ['H2O_main_iso','CO_main_iso']
#ABUNDANCES = ['C/O', 'FeHs']

pc_to_m = 3.086 * 1e16
distance_pc = 19.7538
DISTANCE = distance_pc * pc_to_m

RADIUS_J = 69911 * 1000
MASS_J = 1.898 * 1e27

TEMP_PARAMS = {}

TEMP_PARAMS['log_gamma'] = np.log10(0.4)
TEMP_PARAMS['t_int'] = 200.
TEMP_PARAMS['t_equ'] = 1742
TEMP_PARAMS['log_R'] = np.log10(1.36)
TEMP_PARAMS['log_gravity'] = 4.35
TEMP_PARAMS['log_kappa_IR'] = np.log10(0.01)
TEMP_PARAMS['P0'] = 2

CLOUDS_PARAMS = {}
"""
CLOUDS_PARAMS['log_Pcloud']     = np.log10(0.5)
"""

AB_METALS = {}
# free
AB_METALS['H2O_main_iso']   = -2.4
AB_METALS['CO_main_iso']    = -3.3
# AB_METALS['CH4_main_iso']   = -4.5
# AB_METALS['CO2_main_iso']   = -4.2

# chem_equ
#AB_METALS['C/O'] = 1
#AB_METALS['FeHs'] = 0.66

PHYSICAL_PARAMS = {}
# PHYSICAL_PARAMS['rot_vel'] = 25
PHYSICAL_PARAMS['rot_vel'] = 0

DATA_PARAMS = {}
DATA_PARAMS.update(TEMP_PARAMS)
DATA_PARAMS.update(AB_METALS)
DATA_PARAMS.update(CLOUDS_PARAMS)
DATA_PARAMS.update(PHYSICAL_PARAMS)

WINDOW_LENGTH_LBL = 20  # now referring to the length of the window for the only gaussian filter, for sinfoni data
WINDOW_LENGTH_CK = 41

RV = 31
NOISE = 0.5
NOISE_RES = 0
SNR = 0

CONFIG_DICT = {}
CONFIG_DICT['ALL_PARAMS'] = ABUNDANCES
CONFIG_DICT['ABUNDANCES'] = ABUNDANCES
CONFIG_DICT['NEEDED_LINE_SPECIES'] = ABUNDANCES
CONFIG_DICT['TEMPS'] = ['log_gamma', 't_int', 't_equ', 'log_gravity', 'log_kappa_IR', 'log_R', 'P0']
CONFIG_DICT['CLOUDS'] = CLOUDS_PARAMS.keys()
# CONFIG_DICT['CLOUDS_OPACITIES'] = CLOUDS_OPACITIES
CONFIG_DICT['UNSEARCHED_ABUNDANCES'] = []
CONFIG_DICT['UNSEARCHED_TEMPS'] = []
CONFIG_DICT['UNSEARCHED_CLOUDS'] = []
CONFIG_DICT['PARAMS_NAMES'] = []
CONFIG_DICT['UNSEARCHED_PARAMS'] = []
CONFIG_DICT['RVMAX'] = 0
CONFIG_DICT['RVMIN'] = 0
CONFIG_DICT['DRV'] = 0
CONFIG_DICT['DISTANCE'] = DISTANCE
CONFIG_DICT['WIN_LEN'] = WINDOW_LENGTH_LBL
CONFIG_DICT['LBL_SAMPLING'] = LBL_SAMPLING
CONFIG_DICT['CONVERT_SINFONI_UNITS'] = CONVERT_SINFONI_UNITS
CONFIG_DICT['WRITE_THRESHOLD'] = 0
CONFIG_DICT['DATA_PARAMS'] = DATA_PARAMS