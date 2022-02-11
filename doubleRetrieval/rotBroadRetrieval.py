# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 10:08:15 2021

@author: jeanh
"""

from .util import a_b_range,uniform_prior,calc_SNR
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import PyAstronomy.pyasl as pyasl
import scipy.constants as cst
from scipy.interpolate import interp1d
from time import time
from doubleRetrieval.rebin import *

class RotBroadRetrievalClass:
    def __init__(self,
                 CCF_dRV,CCF_CC,
                 template_wlen,template_flux,
                 data_wlen,
                 data_flux,
                 config,
                 output_dir = ''):
        
        self.CCF_dRV = CCF_dRV
        
        # normalise CCF
        self.CCF_CC = CCF_CC/max(CCF_CC)
        
        # calculate radial velocity of maximum of CCF
        self.dRV_max_i = np.argmax(self.CCF_CC)
        self.dRV_max = self.CCF_dRV[self.dRV_max_i]
        
        # calculate std outside of the peak
        self.SNR,self.std_CCF,RV_max_i,left_bord,right_bord,CC_noisy_function = calc_SNR(self.CCF_dRV, self.CCF_CC)
        if self.std_CCF < 0.3:
            self.std_CCF = 0.3
        
        self.CCF_dRV,self.CCF_CC = np.transpose([[self.CCF_dRV[i],self.CCF_CC[i]] for i in range(left_bord - int(len(self.CCF_dRV)/5),right_bord + int(len(self.CCF_dRV)/5))])
        #self.CCF_dRV,self.CCF_CC = np.transpose([[self.CCF_dRV[i],self.CCF_CC[i]] for i in range(left_bord,right_bord)])
        
        self.t_wvl = template_wlen
        self.t_flux = template_flux
        
        self.d_wvl = data_wlen
        self.d_flux = data_flux
        
        self.params_names = config['PARAMS_NAMES']
        self.prior_range = config['RANGE']
        
        self.rvmin=config['RVMIN']
        self.rvmax= config['RVMAX']
        self.drv= config['DRV']
        
        self.algo = config['ALGORITHM'] # 'fast' or 'slow'
        
        # how much to skip when calculating CCF
        self.skipping = {}
        for key in self.t_wvl.keys():
            wvl_stepsize = max([self.t_wvl[key][i+1]-self.t_wvl[key][i] for i in range(len(self.t_wvl[key])-1)])
            ds_max = max([
                abs(self.t_wvl[key][-1]*self.rvmin*1000/cst.c),
                abs(self.t_wvl[key][-1]*self.rvmax*1000/cst.c)
                ])*1.5
            self.skipping[key] = int(ds_max/wvl_stepsize)+1
            print(self.skipping[key])
            if self.skipping[key]/len(self.t_wvl[key]) >= 0.25:
                print('WARNING: NEED TO SKIP {p:.2f} % OF TEMPLATE SPECTRUM TO INVESTIGATE ALL DOPPLER-SHIFT DURING CROSS-CORRELATION'.format(p=self.skipping[key]/len(self.t_wvl[key])))
        
        self.skipedge = int(max([self.skipping[key] for key in self.skipping.keys()])*1.25)
        print(self.skipedge)
        
        self.function_calls = 0
        self.start_time = time()
        self.output_dir = output_dir
        self.write_threshold = config['WRITE_THRESHOLD']
        self.diag_file = self.output_dir + 'diag.txt'
        
        open(self.diag_file,'w').close
        
        if not os.path.exists(self.output_dir+'model/'):
            os.mkdir(self.output_dir+'model/')
        
        return
    
    def Prior(self,
              cube,
              ndim,
              nparam):
        for i,param in enumerate(self.params_names):
            cube[i] = uniform_prior(cube[i],self.prior_range[param])
    
    def log_prior(self,x,param):
        return a_b_range(x,self.prior_range[param])
    
    def lnprob_pymultinest(self,cube,ndim,nparams):
        params = []
        for i in range(ndim):
            params.append(cube[i])
        params = np.array(params)
        return self.calc_log_likelihood(params)
    
    def calc_log_likelihood(self,param):
        
        # param is spin velocity investigated
        spin_vel,radial_vel = param
        
        log_prior = 0.
        log_likelihood = 0.
        self.function_calls += 1
        
        # check priors
        log_prior += self.log_prior(spin_vel,'spin_vel')
        log_prior += self.log_prior(radial_vel,'radial_vel')
        
        if log_prior == -np.inf:
            return -np.inf
        
        wlen,flux={},{}
        for key in self.t_wvl.keys():
            wlen[key],flux[key] = self.t_wvl[key],self.t_flux[key]
        
        # Doppler shift
        if 'radial_vel' in self.params_names:
            for key in wlen.keys():
                wlen[key],flux[key] = doppler_shift(wlen[key],flux[key],radial_vel)
        
        # rot. broad. spectrum
        if 'spin_vel' in self.params_names:
            for key in self.t_wvl.keys():
                if self.algo == 'fast':
                    flux[key] = pyasl.fastRotBroad(wlen[key],flux[key],0,spin_vel)
                else:
                    # 'slow'
                    flux[key] = pyasl.rotBroad(wlen[key],flux[key],0,spin_vel)
        
        # make sure that we have same wvl coverage as data
        for key in wlen.keys():
            wlen[key],flux[key] = np.transpose([[wlen[key][i],flux[key][i]] for i in range(len(wlen[key])) if wlen[key][i] >= self.d_wvl[key][0] and wlen[key][i] <= self.d_wvl[key][-1]])
        
        # calculate CCF between template and rot. broadened template
        CC_range_i = {}
        for key in self.t_wvl.keys():
            dRV,CC_range_i[key] = pyasl.crosscorrRV(wlen[key],flux[key],
                                   self.t_wvl[key],self.t_flux[key],
                                   rvmin=self.rvmin,rvmax=self.rvmax,drv=self.drv, skipedge=self.skipedge
                                   )
        
        CC = np.array([sum([CC_range_i[key][drv_i] for key in CC_range_i.keys()]) for drv_i in range(len(dRV))])
        drv_max_i = np.argmax(CC)
        drv_max = dRV[drv_max_i]
        
        
        # normalise CCF
        CC_max = CC[drv_max_i]
        CC = CC/CC_max
        """
        # the problem of just shifting the CCF by some RV value is that is doesnt account for the fact that the doppler shift also stretches the spectrum
        CCF_new = CC
        if 'radial_vel' in self.params_names:
            CCF_f = interp1d(dRV + radial_vel,CC,fill_value = 0)
            CCF_new = CCF_f(self.CCF_dRV)
        """
        
        CC_final = [CC[i] for i in range(len(CC)) if dRV[i] >= self.CCF_dRV[0] and dRV[i] <= self.CCF_dRV[-1]]
        
        assert(len(CC_final)==len(self.CCF_CC))
        
        for rv_i in range(len(self.CCF_dRV)):
            log_likelihood += -0.5*((CC_final[rv_i]-self.CCF_CC[rv_i])/self.std_CCF)**2
        
        if self.function_calls%self.write_threshold == 0:
                hours = (time() - self.start_time)/3600.0
                info_list = [self.function_calls, log_likelihood, hours]
                with open(self.diag_file,'a') as f:
                    for i in np.arange(len(info_list)):
                        if (i == len(info_list) - 1):
                            f.write(str(info_list[i]).ljust(15) + "\n")
                        else:
                            f.write(str(info_list[i]).ljust(15) + " ")
        
        if self.function_calls%(self.write_threshold) == 0:
            plt.figure()
            plt.plot(self.CCF_dRV,self.CCF_CC,'r',label='CCF with data')
            plt.plot(dRV,CC,'k',label='CCF with template')
            plt.legend()
            plt.savefig(self.output_dir + 'model/CCF_'+str(self.function_calls)+'.png')
        
        print(log_prior + log_likelihood)
        print("--> ", self.function_calls)
        
        return log_prior + log_likelihood
            
            
            
            
            
    
    