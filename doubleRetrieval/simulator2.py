# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 11:27:23 2021

@author: jeanh
"""

import sys

import os
os.environ["pRT_input_data_path"] = "/home/ipa/quanz/shared/petitRADTRANS/input_data"

from os import path
sys.path.append("/home/ipa/quanz/shared/petitRADTRANS/")

from petitRADTRANS import radtrans as rt
from petitRADTRANS import nat_cst as nc
import numpy as np
from numpy.linalg import inv
from scipy.interpolate import interp1d
from time import time
from PyAstronomy.pyasl import crosscorrRV

from doubleRetrieval.util import *
from doubleRetrieval.rebin import *
from doubleRetrieval.model import *
from sim_config import *


class Simulator2:
    def __init__(self,
                 line_species,
                 wvl_bords = None,
                 wvl_stepsize = None,
                 photometric_filters = None,
                 external_pt_profile = None,
                 mode = 'c-k',
                 lbl_sampling = None,
                 continuum_removed = False,
                 max_win_len = 501,
                 max_RV_shift = 1000.
                 ):
        
        self.line_species = line_species + ['H2_main_iso']
        self.wvl_bords = wvl_bords
        self.wvl_stepsize = wvl_stepsize
        self.phot_filters = photometric_filters
        self.external_pt_profile = external_pt_profile
        self.mode = mode
        
        self.continuum_removed = continuum_removed
        self.max_win_len = max_win_len
        
        extension = 0
        
        if self.wvl_stepsize is not None:
            extension += 4*self.wvl_stepsize
            
        
        if self.phot_filters is not None:
            lower_bord = min([min(self.phot_filters[instr][0]) for instr in self.phot_filters.keys()])
            higher_bord = max([max(self.phot_filters[instr][0]) for instr in self.phot_filters.keys()])
            self.wvl_bords = [lower_bord,higher_bord]
            if self.wvl_stepsize is None:
                self.wvl_stepsize = lower_bord/1000
        
        if self.continuum_removed:
            extension += self.wvl_stepsize*((self.max_win_len+3)/2)
            extension += max((self.wvl_bords[0])*(max_RV_shift)/nc.c,
                             (self.wvl_bords[1])*(max_RV_shift)/nc.c)*1e5
            
        self.wvl_bords[0] -= extension
        self.wvl_bords[-1] += extension
        
        if self.mode == 'c-k':
            self.line_species = convert_to_ck_names(self.line_species)
            lbl_sampling = None
        print('wlen borders: ',self.wvl_bords)
        print('Line species included: ',self.line_species)
        self.rt_obj = Radtrans(line_species = self.line_species,
                              rayleigh_species = ['H2','He'],
                              continuum_opacities = ['H2-H2','H2-He'],
                              mode = self.mode,
                              wlen_bords_micron = self.wvl_bords,
                              lbl_opacity_sampling = lbl_sampling)
        
        
        return
    
    def calc_spectrum(self,
                      metal_params,
                      temps_params,
                      clouds_params,
                      external_metal_profile = None,
                      converting_units = True
                      ):
        external_profile = None
        if self.external_pt_profile is not None:
            pressures,temperatures = self.external_pt_profile
            external_profile = [pressures,temperatures,external_metal_profile]
        else:
            pressures = np.logspace(-6, temps_params['P0'], 100)
            temperatures = nc.guillot_global(
                pressures,
                1e1**temps_params['log_kappa_IR'],
                1e1**temps_params['log_gamma'],
                1e1**temps_params['log_gravity'],
                temps_params['t_int'],
                temps_params['t_equ'])
        
        self.rt_obj.setup_opa_structure(pressures)
        
        
        wlen, flux = retrieval_model_initial(
                self.rt_obj,
                temps_params,
                metal_params,
                clouds_params,
                mode=self.mode,
                external_profile=external_profile)
        if converting_units:
            wlen_temp, flux_temp = convert_units(wlen, flux, temps_params['log_R'], distance = DISTANCE)
        else:
            wlen_temp = 1e4*wlen
            flux_temp = flux
        
        return wlen_temp, flux_temp
