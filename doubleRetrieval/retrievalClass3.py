# -*- coding: utf-8 -*-
"""
Created on Fri May 28 10:32:56 2021

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

import matplotlib.pyplot as plt

from time import time
from PyAstronomy.pyasl import crosscorrRV

from doubleRetrieval.util import *
from doubleRetrieval.rebin import *
from doubleRetrieval.plotting import *
from doubleRetrieval.forward_model import ForwardModel
from doubleRetrieval.rotbroad_utils import trim_spectrum

RADIUS_J = 69911*1000
MASS_J = 1.898*1e27
gravity_cgs_si = 100

# class to define an atmospheric retrieval

class Retrieval:
    def __init__(self,
                 data_obj,
                 prior_obj,
                 config,
                 model='free', # or chem_equ
                 retrieval_name = '',
                 output_path = '',
                 plotting=False,
                 printing=False,
                 timing=False,
                 for_analysis=False):
        
        self.data_obj = data_obj
        self.prior_obj = prior_obj
        self.config = config # dictionary, with keys from the config file
        self.model = model
        self.use_forecaster = config['USE_FORECASTER']
        self.use_prior = config['USE_PRIOR']
        self.retrieval_name = retrieval_name
        self.output_path = output_path
        self.plotting = plotting
        self.printing = printing
        self.timing = timing
        
        
        self.diag_file = output_path + 'diag.txt'
        self.forecaster_file = output_path + 'forecasted_param.txt'
        
        # Declare diagnostics
        self.start_time = time()
        self.function_calls = 0
        self.computed_spectra = 0
        self.NaN_spectra = 0
        self.NaN_spectRES = 0
        self.NaN_savgolay = 0
        self.NaN_crosscorrRV = 0
        self.NaN_photometry = 0
        self.nb_failed_DS = 0
        
        if not for_analysis:
            open(self.diag_file,'w').close()
            
            if self.plotting:
                if not os.path.exists(self.output_path+'model/'):
                    os.mkdir(self.output_path+'model/')
        
        
        self.data_obj.distribute_FMs()
        
        if self.data_obj.RESinDATA() or self.data_obj.CCinDATA():
            self.lbl_itvls = self.data_obj.disjoint_lbl_intervals
            self.CC_to_lbl_itvls = self.data_obj.CC_to_lbl_intervals
            self.RES_to_lbl_itvls = self.data_obj.RES_to_lbl_intervals
            self.lbl_intervals_max_step = self.data_obj.disjoint_lbl_intervals_max_stepsize
            
            print(self.lbl_itvls)
            print(self.CC_to_lbl_itvls)
            print(self.RES_to_lbl_itvls)
            print(self.lbl_intervals_max_step)
        
        
        self.forwardmodel_lbl = None
        self.forwardmodel_ck = None
        
        if self.data_obj.PHOTinDATA():
            print('CALLING FORWARD MODEL WITH C-K MODE')
            wlen_borders_ck = self.data_obj.ck_FM_interval
            
            self.forwardmodel_ck = ForwardModel(
                 wlen_borders = wlen_borders_ck,
                 max_wlen_stepsize = wlen_borders_ck[1]/1000,
                 mode = 'c-k',
                 line_opacities = self.config['ABUNDANCES'] + self.config['UNSEARCHED_ABUNDANCES'],
                 chem_model = self.model,
                 max_RV = 0,
                 max_winlen = 0
                 )
            self.forwardmodel_ck.calc_rt_obj(lbl_sampling = None)
            
        if self.data_obj.CCinDATA() or self.data_obj.RESinDATA():
            print('CALLING FORWARD MODEL WITH LBL MODE')
            self.forwardmodel_lbl = {}
            
            for interval_key in self.lbl_itvls.keys():
                wlen_borders_lbl = self.lbl_itvls[interval_key]
                max_wlen_stepsize = self.lbl_intervals_max_step[interval_key]
                
                self.forwardmodel_lbl[interval_key] = ForwardModel(
                     wlen_borders = wlen_borders_lbl,
                     max_wlen_stepsize = max_wlen_stepsize,
                     mode = 'lbl',
                     line_opacities = self.config['ABUNDANCES'] + self.config['UNSEARCHED_ABUNDANCES'],
                     chem_model = self.model,
                     max_RV = 2*max(abs(self.config['RVMIN']),self.config['RVMAX']),
                     max_winlen = int(1.5*self.config['WIN_LEN'])
                     )
                self.forwardmodel_lbl[interval_key].calc_rt_obj(lbl_sampling = self.config['LBL_SAMPLING'])
            
            print('THEY ARE {val} FMs USING LBL MODE'.format(val=len(list(self.forwardmodel_lbl.keys()))))
            """
            wlen_borders_lbl,max_wlen_stepsize = self.data_obj.wlen_details_lbl()
            max_wlen_stepsize = data_obj.max_stepsize(self.data_obj.getCCSpectrum()[0])
            
            self.forwardmodel_lbl = ForwardModel(
                 wlen_borders = wlen_borders_lbl,
                 max_wlen_stepsize = max_wlen_stepsize,
                 mode = 'lbl',
                 line_opacities = self.config['ABUNDANCES'] + self.config['UNSEARCHED_ABUNDANCES'],
                 model = self.model,
                 max_RV = 2*max(abs(self.config['RVMIN']),self.config['RVMAX']),
                 max_winlen = int(1.5*self.config['WIN_LEN'])
                 )
            self.forwardmodel_lbl.calc_rt_obj(lbl_sampling = self.config['LBL_SAMPLING'])
            """
        
        return
    
    def Prior(self,
              cube,
              ndim,
              nparam):
        for i,name in enumerate(self.config['PARAMS_NAMES']):
            cube[i] = self.prior_obj.log_cube_priors[name](cube[i])
    
    def ultranest_prior(self,
                        cube):
        params = cube.copy()
        for i in range(len(self.config['PARAMS_NAMES'])):
            params[i] = self.prior_obj.log_cube_priors[self.config['PARAMS_NAMES'][i]](cube[i])
        return params
    
    def lnprob_pymultinest(self,cube,ndim,nparams):
        params = []
        for i in range(ndim):
            params.append(cube[i])
        params = np.array(params)
        return self.calc_log_likelihood(params)
    
    def lnprob_mcmc(self,x):
        return self.calc_log_likelihood(x)
    
    def calc_log_L_CC(self,wlen,flux,temp_params,data_wlen,data_flux,data_N,data_sf2):
        
        log_L_CC = 0
        
        # get data
        wlen_data,flux_data = data_wlen,data_flux
        
        N = data_N
        s_f2 = data_sf2
        
        # cut what I don't need to improve speed
        #wlen_cut,flux_cut = trim_spectrum(wlen,flux,wlen_data,threshold=5000,keep=1000)
        
        wlen_removed,flux_removed,sgfilter,wlen_rebin,flux_rebin = rebin_to_CC(wlen,flux,wlen_data,win_len = self.config['WIN_LEN'],method='linear',filter_method = 'only_gaussian',convert = self.config['CONVERT_SINFONI_UNITS'],log_R=temp_params['log_R'],distance=self.config['DISTANCE'])
        
        if sum(np.isnan(flux_rebin))>0:
            self.NaN_spectRES += 1
            log_L_CC += -np.inf
        
        assert(len(wlen_removed) == len(flux_removed))
        
        if sum(np.isnan(flux_removed))>0:
            self.NaN_savgolay += 1
            log_L_CC += -np.inf
        # cross-correlation
        dRV,CC=crosscorrRV(wlen_data,
                           flux_data,
                           wlen_removed,
                           flux_removed,
                           rvmin=self.config['RVMIN'],
                           rvmax=self.config['RVMAX'],
                           drv=self.config['DRV'])
        
        if sum(np.isnan(CC)>0):
            self.NaN_crosscorrRV += 1
            log_L_CC += -np.inf
        
        CC=CC/N
        RV_max_i=np.argmax(CC)
        CC_max = CC[RV_max_i]
        
        if self.plotting:
            if (self.function_calls%self.config['PLOTTING_THRESHOLD'] == 0):
                plt.figure()
                plt.plot(dRV,CC)
                plt.axvline(dRV[RV_max_i],color='r',label='Max CC at RV={rv}'.format(rv=dRV[RV_max_i]))
                plt.legend()
                plt.title('C-C ' + str(int(self.function_calls/self.config['PLOTTING_THRESHOLD'])))
                plt.savefig(self.output_path+'model/CC_fct_'+str(int(self.function_calls/self.config['PLOTTING_THRESHOLD']))+'.png',dpi=100)
        
        # need to doppler shift the model to the argmax of the CC-function. For that, we need to go back to high-resolution spectrum out of petitRADTRAS
        
        wlen_removed,flux_removed = wlen,flux
        if abs(dRV[RV_max_i])<max(abs(self.config['RVMIN']),abs(self.config['RVMAX']))*0.75:
            wlen_removed,flux_removed = doppler_shift(wlen,flux,dRV[RV_max_i])
        else:
            print('Cant Dopplershift too much')
            self.nb_failed_DS += 1
        
        wlen_removed,flux_removed,sgfilter,wlen_rebin,flux_rebin = rebin_to_CC(wlen_removed,flux_removed,wlen_data,win_len = self.config['WIN_LEN'],method='datalike',filter_method = 'only_gaussian',convert = self.config['CONVERT_SINFONI_UNITS'],log_R=temp_params['log_R'],distance=self.config['DISTANCE'])
        
        assert(len(wlen_removed) == len(wlen_data))
        
        s_g2 = 1./len(flux_removed)*np.sum(flux_removed**2)
        
        if (s_f2-2*CC_max+s_g2) <= 0:
            self.NaN_crosscorrRV += 1
            log_L_CC += -np.inf
            print('Negative values inside logarithm')
        log_L_CC += -N*np.log(s_f2-2*CC_max+s_g2)/2
        
        return log_L_CC,wlen_removed,flux_removed,sgfilter
    
    def calc_log_L_PHOT(self,wlen,flux,temp_params):
        
        log_L_PHOT = 0.
        
        # get data
        PHOT_flux,PHOT_flux_err,filt,filt_func,filt_mid,filt_width = self.data_obj.getPhot() #dictionaries
        
        model_photometry,wlen_rebin,flux_rebin = rebin_to_PHOT(wlen,flux,filt_func = filt_func,log_R = temp_params['log_R'],distance = self.config['DISTANCE'])
        
        for instr in PHOT_flux.keys():
            
            if np.isnan(model_photometry[instr]):
                self.NaN_photometry += 1
                return -np.inf
            
            log_L_PHOT += -0.5*((model_photometry[instr]-PHOT_flux[instr])/PHOT_flux_err[instr])**2
        
        return log_L_PHOT,model_photometry,wlen_rebin,flux_rebin
    
    def calc_log_L_RES(self,wlen,flux,temp_params,wlen_data,flux_data,flux_err,inverse_cov,flux_data_std):
        
        log_L_RES = 0
        
        wlen_rebin,flux_rebin = rebin_to_RES(wlen,flux,wlen_data,log_R = temp_params['log_R'],distance = self.config['DISTANCE'])
        
        if sum(np.isnan(flux_rebin))>0:
            self.NaN_spectRES += 1
            log_L_RES += -np.inf
        
        #log_L_RES += -0.5*np.dot((flux_data-flux_temp),np.dot(inverse_cov,(flux_data-flux_temp)))
        log_L_RES += -0.5*sum(((flux_data-flux_rebin)/flux_data_std)**2)
        
        return log_L_RES,wlen_rebin,flux_rebin
    
    def calc_log_likelihood(self,params):
        ab_metals = {}
        temp_params = {}
        clouds_params = {}
        
        params_names = self.config['PARAMS_NAMES']
        all_params = self.config['PARAMS_NAMES'] + self.config['UNSEARCHED_ABUNDANCES'] + self.config['UNSEARCHED_TEMPS'] + self.config['UNSEARCHED_CLOUDS']
        
        for i,name in enumerate(params_names):
            if name in self.config['TEMPS']:
                temp_params[name] = params[i]
            if name in self.config['ABUNDANCES']:
                ab_metals[name] = params[i]
            if name in self.config['CLOUDS']:
                clouds_params[name] = params[i]
        for name in all_params:
            if name in self.config['UNSEARCHED_ABUNDANCES']:
                ab_metals[name] = self.config['DATA_PARAMS'][name]
            if name in self.config['UNSEARCHED_TEMPS']:
                temp_params[name] = self.config['DATA_PARAMS'][name]
            if name in self.config['UNSEARCHED_CLOUDS']:
                clouds_params[name] = self.config['DATA_PARAMS'][name]
        
        if self.use_forecaster or self.use_prior == 'M':
            temp_params['log_gravity'] = np.log10(cst.gravitational_constant) + np.log10(temp_params['M']) + np.log10(MASS_J) - 2*temp_params['log_R'] - 2*np.log10(RADIUS_J) + np.log10(gravity_cgs_si)
        
        if self.use_prior == 'R':
            temp_params['log_R'] = np.log10(temp_params['log_R'])
        
        if 'log_R' not in temp_params.keys() and 'R' in temp_params.keys():
            temp_params['log_R'] = np.log10(temp_params['R'])
        
        self.function_calls += 1
        
        if self.timing:
            t0 = time()
        
        """Prior calculation of all input parameters"""
        log_prior = 0.
        
        """Metal abundances: check that their summed mass fraction is below 1."""
        metal_sum = 0.
        for name in ab_metals.keys():
            if name in self.config['ABUNDANCES']:
                log_prior += self.prior_obj.log_priors[name](ab_metals[name])
            if name != 'C/O' and name != 'FeHs':
                metal_sum += 1e1**ab_metals[name]
        
        if metal_sum > 1.:
            log_prior += -np.inf
        
        """temperature parameters"""
        if len(self.config['TEMPS'])>0:
            for name in self.config['TEMPS']:
                log_prior += self.prior_obj.log_priors[name](temp_params[name])
        
        if len(self.config['CLOUDS'])>0:
            for name in self.config['CLOUDS']:
                log_prior += self.prior_obj.log_priors[name](clouds_params[name])
        
        """return -inf if parameters fall outside prior distribution"""
        
        if (log_prior == -np.inf):
            return -np.inf
        
        """Calculate the log-likelihood"""
        log_likelihood = 0.
        log_L_PHOT = 0.
        log_L_CC = 0.
        log_L_RES = 0.
        
        wlen_ck,flux_ck,wlen_lbl,flux_lbl = None,None,None,None
        
        if self.data_obj.RESinDATA():
            RES_data_wlen,RES_data_flux,flux_err,inverse_cov,flux_data_std = self.data_obj.getRESSpectrum()
        
        # evaluate log-likelihood for FM using c-k mode
        wlen_RES,flux_RES = {},{}
        wlen_ck,flux_ck=None,None
        wlen_PHOT,flux_PHOT=None,None
        model_photometry = {}
        if self.forwardmodel_ck is not None:
            wlen_ck,flux_ck = self.forwardmodel_ck.calc_spectrum(
                      ab_metals = ab_metals,
                      temp_params = temp_params,
                      clouds_params = clouds_params,
                      external_pt_profile = None)
            
            if sum(np.isnan(flux_ck))>0 or sum(np.isnan(wlen_ck))>0:
                self.NaN_spectra += 1
                return -np.inf
            
            if self.data_obj.PHOTinDATA():
                print('c-k photometry Log-L')
                log_L_PHOT,model_photometry,wlen_PHOT,flux_PHOT = self.calc_log_L_PHOT(
                    wlen_ck,
                    flux_ck,
                    temp_params)
            
            if self.data_obj.RES_data_with_ck:
                for instr in RES_data_wlen.keys():
                    if self.data_obj.RES_data_info[instr][0] == 'c-k':
                        print('c-k RES-Log-L for',instr)
                        log_L_RES_temp,wlen_RES[instr],flux_RES[instr] = self.calc_log_L_RES(
                            wlen_ck,
                            flux_ck,
                            temp_params,
                            RES_data_wlen[instr],RES_data_flux[instr],flux_err[instr],inverse_cov[instr],flux_data_std[instr])
                        log_L_RES += log_L_RES_temp
        
        if self.data_obj.CCinDATA():
            CC_data_wlen,CC_data_flux = self.data_obj.getCCSpectrum()
            data_N,data_sf2 = self.data_obj.CC_data_N,self.data_obj.CC_data_sf2
        
        # evaluate log-likelihood for FM using lbl mode
        wlen_CC,flux_CC,sgfilter={},{},{}
        if self.forwardmodel_lbl is not None:
            
            for interval_key in self.lbl_itvls.keys():
                wlen_lbl,flux_lbl = self.forwardmodel_lbl[interval_key].calc_spectrum(
                          ab_metals = ab_metals,
                          temp_params = temp_params,
                          clouds_params = clouds_params,
                          external_pt_profile = None)
                
                if sum(np.isnan(flux_lbl))>0 or sum(np.isnan(wlen_lbl))>0:
                    self.NaN_spectra += 1
                    return -np.inf
                
                if self.data_obj.CCinDATA():
                    for instr in CC_data_wlen.keys():
                        if self.CC_to_lbl_itvls[instr] == interval_key:
                            print('lbl CC-Log-L for',instr)
                            log_L_CC_temp,wlen_CC[instr],flux_CC[instr],sgfilter[instr] = self.calc_log_L_CC(
                                wlen_lbl,
                                flux_lbl,
                                temp_params,
                                CC_data_wlen[instr],CC_data_flux[instr],data_N[instr],data_sf2[instr]
                                )
                            log_L_CC += log_L_CC_temp
                
                if self.data_obj.RESinDATA():
                    for instr in RES_data_wlen.keys():
                        if self.data_obj.RES_data_info[instr][0] == 'lbl':
                            if self.RES_to_lbl_itvls[instr] == interval_key:
                                print('lbl RES-Log-L for',instr)
                                log_L_RES_temp,wlen_RES[instr],flux_RES[instr] = self.calc_log_L_RES(
                                    wlen_lbl,
                                    flux_lbl,
                                    temp_params,
                                    RES_data_wlen[instr],RES_data_flux[instr],flux_err[instr],inverse_cov[instr],flux_data_std[instr])
                                log_L_RES += log_L_RES_temp
        
        if self.timing:
            t1 = time()
            print('Forward Models and likelihood functions: ',t1-t0)
        
        self.computed_spectra += 1
        
        log_likelihood += log_L_CC + log_L_RES + log_L_PHOT
        print(log_prior + log_likelihood)
        print("--> ", self.function_calls, " --> ", self.computed_spectra)
        if self.printing:
            if (self.function_calls%self.config['WRITE_THRESHOLD'] == 0):
                hours = (time() - self.start_time)/3600.0
                info_list = [self.function_calls, self.computed_spectra,
                             log_L_CC,log_L_RES,log_L_PHOT,log_likelihood, hours, 
                             self.nb_failed_DS ,self.NaN_spectra, self.NaN_spectRES, self.NaN_savgolay, self.NaN_crosscorrRV, self.NaN_photometry]
                with open(self.diag_file,'a') as f:
                    for i in np.arange(len(info_list)):
                        if (i == len(info_list) - 1):
                            f.write(str(info_list[i]).ljust(15) + "\n")
                        else:
                            f.write(str(info_list[i]).ljust(15) + " ")
        
        if self.plotting:
            if (self.function_calls%self.config['PLOTTING_THRESHOLD'] == 0):
                if not path.exists(self.output_path+'model'+'/plot'+str(int(self.function_calls/self.config['PLOTTING_THRESHOLD']))):
                    CC_data_wlen,CC_data_flux,data_RES_wlen,data_RES_flux,data_RES_err,data_RES_inv_cov,data_RES_flux_std,data_sim_wlen,data_sim_flux,data_PHOT_flux,data_PHOT_err,data_PHOT_filter,data_PHOT_filter_function,data_PHOT_filter_midpoint,data_PHOT_filter_width=None,None,None,None,None,None,None,None,None,{},{},{},{},{},{}
                    if self.data_obj.CCinDATA():
                        data_CC_wlen,data_CC_flux = self.data_obj.getCCSpectrum()
                    if self.data_obj.RESinDATA():
                        data_RES_wlen,data_RES_flux,data_RES_err,data_RES_inv_cov,data_RES_flux_std = self.data_obj.getRESSpectrum()
                    if self.data_obj.PHOTinDATA():
                        data_sim_wlen,data_sim_flux = self.data_obj.getSimSpectrum()
                        data_PHOT_flux,data_PHOT_err,data_PHOT_filter,data_PHOT_filter_function,data_PHOT_filter_midpoint,data_PHOT_filter_width = self.data_obj.getPhot()
                    print('PLOTTING MODEL')
                    plot_data(
                            self.config,
                            CC_wlen = CC_data_wlen,
                            CC_flux = CC_data_flux,
                            model_CC_wlen = wlen_CC,
                            model_CC_flux = flux_CC,
                            sgfilter = None,
                            RES_wlen = data_RES_wlen,
                            RES_flux = data_RES_flux,
                            RES_flux_err = data_RES_flux_std,
                            model_RES_wlen = wlen_RES,
                            model_RES_flux = flux_RES,
                            PHOT_midpoint = data_PHOT_filter_midpoint,
                            PHOT_width = data_PHOT_filter_width,
                            PHOT_flux = data_PHOT_flux,
                            PHOT_flux_err = data_PHOT_err,
                            PHOT_filter = data_PHOT_filter,
                            PHOT_sim_wlen = wlen_PHOT,
                            PHOT_sim_flux = flux_PHOT,
                            model_PHOT_flux = model_photometry,
                            output_file = self.output_path+'model',
                            plot_name = 'plot'+str(int(self.function_calls/self.config['PLOTTING_THRESHOLD'])))
        if self.timing and self.plotting:
            t2 = time()
            print('Printing and plotting: ',t2-t1)
        
        if np.isnan(log_prior + log_likelihood):
            return -np.inf
        else:
            return log_prior + log_likelihood