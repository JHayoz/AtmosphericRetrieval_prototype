# -*- coding: utf-8 -*-
"""
Created on Fri May 28 09:45:22 2021

@author: jeanh
"""

import sys

import os
#os.environ["pRT_input_data_path"] = "/scratch/software/petitRADTRANS/petitRADTRANS/input_data"
os.environ["pRT_input_data_path"] = "/home/ipa/quanz/shared/petitRADTRANS/input_data"

from os import path
#sys.path.append("/scratch/software/petitRADTRANS/")
sys.path.append("/home/ipa/quanz/shared/petitRADTRANS/")

from petitRADTRANS import radtrans as rt
from petitRADTRANS import nat_cst as nc
import numpy as np
from numpy.linalg import inv
from scipy.interpolate import interp1d
from time import time
from PyAstronomy.pyasl import crosscorrRV
import matplotlib.pyplot as plt

from doubleRetrieval.util import *
from doubleRetrieval.rebin import *
from doubleRetrieval.model2 import *

class ForwardModel:
    def __init__(self,
                 wlen_borders,
                 max_wlen_stepsize,
                 mode,
                 line_opacities,
                 clouds = None,
                 cloud_species = [],
                 do_scat_emis = False,
                 model = 'free', # can be 'free' or 'chem_equ'
                 max_RV = 1000.,
                 max_winlen = 201,
                 include_H2 = True,
                 only_include = 'all'
                 ):
        
        self.wlen_borders = wlen_borders
        self.max_wlen_stepsize = max_wlen_stepsize
        self.mode = mode
        self.line_opacities = line_opacities.copy()
        self.only_include = only_include
        if include_H2:
            self.line_opacities += ['H2_main_iso']
        if clouds == 'grey_deck' or clouds is None:
            self.cloud_species =  []
            self.do_scat_emis = False
        else:
            # if clouds = 'ackermann'
            self.cloud_species = cloud_species
            self.do_scat_emis = True
        self.model = model
        self.max_RV = max_RV
        self.max_winlen = max_winlen
        
        if self.model == 'chem_equ':
            if self.mode == 'c-k':
                self.line_opacities = poor_mans_abunds_ck()
            else:
                self.line_opacities = poor_mans_abunds_lbl()
            if self.only_include != 'all':
                self.line_opacities = self.only_include
                if include_H2:
                    self.line_opacities += ['H2_main_iso']
        else:
            # 'free' model
            if self.mode == 'c-k':
                self.line_opacities = convert_to_ck_names(self.line_opacities)
        
        self.opa_struct_set = False
        self.rt_obj = None
        
        print('Forward model setup with {model} model and {mode} mode'.format(model=self.model,mode=self.mode))
        
        return
    
    def extend(self):
        extensions = 0.
        
        # Doppler shift
        extensions += max(abs((self.wlen_borders[0])*(self.max_RV)/nc.c),
                             abs((self.wlen_borders[1])*(self.max_RV)/nc.c))*1e5
        
        # rebin
        extensions += 10*self.max_wlen_stepsize
        
        # Continuum-removed
        extensions += 2*self.max_wlen_stepsize*(self.max_winlen+3)
        
        # adjust wvl range
        self.wlen_borders[0] -= extensions
        self.wlen_borders[1] += extensions
    
    def calc_rt_obj(self,
                    lbl_sampling = None):
        
        # adjust wvl range
        self.extend()
        
        if self.mode == 'c-k':
            lbl_sampling = None
        print('wlen borders: ',self.wlen_borders)
        print('Line species included: ',self.line_opacities)
        line_opacities_to_use = self.line_opacities
        if 'He' in line_opacities_to_use:
            line_opacities_to_use.remove('He')
        if 'H2' in line_opacities_to_use and self.mode == 'lbl':
            line_opacities_to_use.remove('H2')
            line_opacities_to_use.append('H2_main_iso')
        self.rt_obj = Radtrans(line_species = line_opacities_to_use,
                              rayleigh_species = ['H2','He'],
                              continuum_opacities = ['H2-H2','H2-He'],
                              mode = self.mode,
                              wlen_bords_micron = self.wlen_borders,
                              cloud_species = self.cloud_species,
                              do_scat_emis = self.do_scat_emis,
                              lbl_opacity_sampling = lbl_sampling)
        
    def calc_spectrum(self,
                      ab_metals,
                      temp_params,
                      clouds_params,
                      external_pt_profile = None,
                      return_profiles = False,
                      contribution = False,
                      set_mol_abund = None
                      ):
        
        # temperature-pressure profile
        
        if external_pt_profile is not None:
            pressures,temperatures = external_pt_profile
        else:
            pressures = np.logspace(-6, temp_params['P0'], 100)
            temperatures = nc.guillot_global(
                pressures,
                1e1**temp_params['log_kappa_IR'],
                1e1**temp_params['log_gamma'],
                1e1**temp_params['log_gravity'],
                temp_params['t_int'],
                temp_params['t_equ'])
        
        # setup up opacity structure if not yet done so
        if not(self.opa_struct_set):
            self.rt_obj.setup_opa_structure(pressures)
            self.opa_struct_set = True
        
        # translate ab_metals dictionary if we're using c-k mode
        if self.mode == 'c-k':
            ab_metals_ck = {name_ck(mol):ab_metals[mol] for mol in ab_metals.keys()}
            ab_metals = ab_metals_ck
        
        # calculate forward model depending on case: free or chemical equilibrium
        if self.model == 'free':
            wlen, flux, abundances = retrieval_model_initial(
                self.rt_obj,
                pressures,
                temperatures,
                temp_params,
                ab_metals,
                clouds_params,
                mode=self.mode,
                contribution = contribution)
        else:
            # 'chem_equ'
            wlen, flux, abundances = retrieval_model_chem_disequ(
                self.rt_obj,
                temp_params,
                ab_metals,
                clouds_params,
                mode=self.mode,
                contribution = contribution,
                only_include = self.only_include,
                set_mol_abund = set_mol_abund)
            
            if self.only_include != 'all':
                assert(len(abundances.keys())==len(self.only_include)+2)
        if return_profiles:
            return wlen,flux,pressures,temperatures,abundances
        else:
            return wlen,flux
    
    def calc_emission_contribution(self):
        
        return self.rt_obj.contr_em
    
    def calc_pressure_distribution(
        self,
        config,
        contr_em_fct,
        ab_metals,
        temp_params,
        wlen,
        flux,
        which_data_format = 'CC',
        which_em = 'molecules', # 'retrieved' or 'molecules' refering to whether we take em contr fct from retrieved spectrum or from each molecule
        which_abund = 'retr_abund', # 'high_abund' or 'retr_abund' refering to the amount of the molecules to include when calculating em contr fct and flux
        which_included = 'included', # 'excluded' or 'included' refering to whether we only include one molecule at a time, or if we only exclude one molecule at a time but include all others
        output_dir = '',
        plot_distr = True
        ):
        
        if which_em == 'retrieved':
            contribution = False
        else:
            contribution = True
        
        #wlen_lbl_ref = nc.c/self.rt_obj.freq*1e4
        
        wlen_mol,flux_mol,flux_diff,flux_diff_interped,pressure_distr = {},{},{},{},{}
        abundances_considered = ab_metals.keys()
        
        if plot_distr:
            figs,axs = plt.subplots(nrows=len(abundances_considered),ncols=2,figsize=(2*5,len(abundances_considered)*5))
        
        for mol_i,mol in enumerate(abundances_considered):
            print('Considering '+mol)
            mol_ab_metals = {}
            for key in ab_metals.keys():
                # copy retrieved abundances
                if which_included == 'excluded':
                    if which_abund == 'retr_abund':
                        mol_ab_metals[key] = ab_metals[key]
                    else:
                        # 'high_abund'
                        mol_ab_metals[key] = -3.5
                else:
                    # 'included'
                    mol_ab_metals[key] = -20
            
            # remove one molecule
            if which_included == 'excluded':
                mol_ab_metals[mol] = -20
            else:
                # 'included'
                if which_abund == 'retr_abund':
                    mol_ab_metals[mol] = ab_metals[mol]
                else:
                    # 'high_abund'
                    mol_ab_metals[mol] = -3.5
            print(mol_ab_metals)
            print(contribution)
            wlen_lbl_mol,flux_lbl_mol = self.calc_spectrum(
                    ab_metals = mol_ab_metals,
                    temp_params = temp_params,
                    clouds_params = {},
                    external_pt_profile = None,
                    return_profiles = False,
                    contribution = contribution)
            if which_em == 'molecules':
                contr_em_fct = self.calc_emission_contribution()
                print(np.shape(contr_em_fct))
            if which_data_format == 'CC':
                wlen_mol[mol],flux_mol[mol],calc_filter,wlen_rebin_datalike,flux_rebin_datalike = rebin_to_CC(wlen_lbl_mol,flux_lbl_mol,wlen,win_len=config['WIN_LEN'],method='datalike',filter_method = 'only_gaussian',nb_sigma=5,convert = True,log_R=temp_params['log_R'],distance=config['DISTANCE'])
            else:
                # 'RES'
                # need to take the spectrum difference between retrieved and spectrum where we exclude each molecule, but we still need to take the emission contribution function of the molecules
                mol_ab_metals = {key:ab_metals[key] for key in ab_metals.keys()}
                mol_ab_metals[mol] = -20
                wlen_lbl_mol_RES,flux_lbl_mol_RES = self.calc_spectrum(
                    ab_metals = mol_ab_metals,
                    temp_params = temp_params,
                    clouds_params = {},
                    external_pt_profile = None,
                    return_profiles = False,
                    contribution = False)
                
                wlen_mol[mol],flux_mol[mol] = rebin_to_RES(wlen_lbl_mol_RES,flux_lbl_mol_RES,wlen,log_R=temp_params['log_R'],distance=config['DISTANCE'])
            wlen_lbl_mol = 1e4*wlen_lbl_mol
            if which_included == 'excluded' or which_data_format == 'RES':
                flux_diff[mol] = np.abs(flux_mol[mol]-flux)
            else:
                # 'included'
                flux_diff[mol] = np.abs(flux_mol[mol])
            
            flux_diff_interp = interp1d(wlen_mol[mol],flux_diff[mol],bounds_error=False,fill_value=0)
            flux_diff_interped[mol] = flux_diff_interp(wlen_lbl_mol)
            
            pressure_distr[mol] = np.dot(contr_em_fct,flux_diff_interped[mol])/sum(flux_diff_interped[mol])
            
            if plot_distr:
                print(np.shape(contr_em_fct))
                pressures = np.logspace(-6,temp_params['P0'],100)
                X,Y = np.meshgrid(wlen_lbl_mol[::100], pressures)
                axs[mol_i,0].contourf(X, Y,contr_em_fct[:,::100],cmap=plt.cm.bone_r)
                axs[mol_i,0].set_xlim([np.min(wlen_lbl_mol),np.max(wlen_lbl_mol)])
                axs[mol_i,0].set_title('Contribution emission function',fontsize=12)
                axs[mol_i,0].set_yscale('log')
                axs[mol_i,0].set_ylim([1e2,1e-6])
                axs[mol_i,1].plot(pressure_distr[mol],pressures)
                axs[mol_i,1].set_yscale('log')
                axs[mol_i,1].set_ylim([1e2,1e-6])
                
                axs[mol_i,0].set_ylabel(mol)
        if plot_distr:
            figs.savefig(output_dir + 'em_contr_fct_VS_press_distr_'+which_em[:4] + '_AB' + which_abund[:4] + '_' + which_included[:4] +'.png',dpi=600)
        
        return pressure_distr,wlen_lbl_mol,flux_diff_interped,contr_em_fct
    
    def calc_em_contr_pressure_distr(
            self,
            config,
            samples,
            data_obj,
            contribution = True,
            which_em = 'molecules', # or 'molecules' refering to whether we take em contr fct from retrieved spectrum or from each molecule
            which_abund = 'retr_abund', # 'high_abund' or 'retr_abund' refering to the amount of the molecules to include when calculating em contr fct and flux
            which_included = 'included', # 'excluded' or 'included' refering to whether we only include one molecule at a time, or if we only exclude one molecule at a time but include all others
            output_dir = '',
            plot_distr = True
            ):
        
        
        ab_metals,temp_params = calc_retrieved_params(config,samples)
        
        wlen_lbl,flux_lbl = self.calc_spectrum(
                ab_metals = ab_metals,
                temp_params = temp_params,
                clouds_params = {},
                external_pt_profile = None,
                return_profiles = False,
                contribution=contribution)
        contr_em_fct = None
        if which_em == 'retrieved':
            contr_em_fct = np.array(self.calc_emission_contribution())
        wlen_included,flux_included = None,None
        which_data_format = None
        if data_obj.CCinDATA():
            CC_wlen_data_dic,CC_flux_data_dic = data_obj.getCCSpectrum()
            for key in CC_wlen_data_dic.keys():
                CC_wlen_data,CC_flux_data = CC_wlen_data_dic[key],CC_flux_data_dic[key]
            
            CC_wlen,CC_flux,calc_filter,wlen_rebin_datalike,flux_rebin_datalike = rebin_to_CC(wlen_lbl,flux_lbl,CC_wlen_data,win_len=config['WIN_LEN'],method='datalike',filter_method = 'only_gaussian',nb_sigma=5,convert = True,log_R=temp_params['log_R'],distance=config['DISTANCE'])
            wlen_included,flux_included=CC_wlen,CC_flux
            which_data_format = 'CC'
            
        if data_obj.RESinDATA():
            RES_data_wlen,RES_data_flux,RES_data_err,RES_inv_cov,RES_data_flux_err = data_obj.getRESSpectrum()
            for key in RES_data_wlen.keys():
                RES_wlen_data,RES_flux_data = RES_data_wlen[key],RES_data_flux[key]
            
            RES_wlen,RES_flux = rebin_to_RES(wlen_lbl,flux_lbl,RES_wlen_data,log_R=temp_params['log_R'],distance=config['DISTANCE'])
            if not data_obj.CCinDATA():
                wlen_included,flux_included=RES_wlen,RES_flux
                which_data_format = 'RES'
        
        pressure_distr,wlen_lbl_ref,CC_flux_diff_interped,contr_em_fct = self.calc_pressure_distribution(
                config,
                contr_em_fct,
                ab_metals,
                temp_params,
                wlen_included,
                flux_included,
                which_data_format = which_data_format,
                which_em = which_em, # or 'molecules' refering to whether we take em contr fct from retrieved spectrum or from each molecule
                which_abund = which_abund, # 'high_abund' or 'retr_abund' refering to the amount of the molecules to include when calculating em contr fct and flux
                which_included = which_included, # 'excluded' or 'included' refering to whether we only include one molecule at a time, or if we only exclude one molecule at a time but include all others
                output_dir = output_dir,
                plot_distr = False)
        
        return wlen_lbl_ref,contr_em_fct,CC_flux_diff_interped,pressure_distr
    
