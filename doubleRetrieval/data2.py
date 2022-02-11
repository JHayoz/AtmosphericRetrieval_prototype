# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 09:44:19 2021

@author: jeanh
"""

from .util import *
from scipy.interpolate import interp1d
import numpy as np
from numpy.linalg import inv
import os
from .plotting import plot_data
from itertools import combinations
from sklearn.neighbors import KernelDensity

class Data:
    def __init__(self,
                 data_dir = None,
                 use_sim_files = [],
                 PHOT_flux_format = 4,
                 PHOT_filter_dir = None,
                 PHOT_flux_dir = None,
                 CC_data_dir=None,
                 RES_data_dir=None,
                 RES_err_dir=None,
                 quick_load = False,
                 ):
        
        # continuum-removed spectrum, aka used with cross-correlation spectroscopy
        self.CC_data_wlen = {}
        self.CC_data_flux = {}
        self.CC_data_N    = {}
        self.CC_data_sf2  = {}
        # photometric data
        self.PHOT_data_flux = {}
        self.PHOT_data_err = {}
        self.PHOT_data_filter = {}
        self.PHOT_filter_function = {}
        self.PHOT_filter_midpoint = {}
        self.PHOT_filter_width = {}
        # simulated continuum-included spectrum for photometry
        self.PHOT_sim_spectrum_wlen = {}
        self.PHOT_sim_spectrum_flux = {}
        # continuum-included spectrum, aka used with method of residuals
        self.RES_data_wlen = {}
        self.RES_data_flux = {}
        self.RES_cov_err = {}
        self.RES_inv_cov = {}
        self.RES_data_flux_err = {}
        
        """
        if 'PHOT' in use_sim_files:
            self.PHOT_data_filter,self.PHOT_filter_function,self.PHOT_data_flux,self.PHOT_data_err,self.PHOT_filter_midpoint,self.PHOT_filter_width={},{},{},{},{},{}
        if 'CC' in use_sim_files:
            self.CC_data_wlen,self.CC_data_flux,self.CC_data_N,self.CC_data_sf2={},{},{},{}
        if 'RES' in use_sim_files:
            self.RES_data_wlen,self.RES_data_flux,self.RES_cov_err,self.RES_inv_cov,self.RES_data_flux_err={},{},{},{},{}
        """
        if data_dir is not None:
            for files in os.listdir(data_dir):
                if 'spectrum' in files:
                    if 'RES' in files and 'RES' in use_sim_files and not 'cksim' in files:
                        RES_data_dir = data_dir+'/'+files
                    if 'CC' in files and 'CC' in use_sim_files:
                        CC_data_dir = data_dir+'/'+files
                    if 'cksim' in files and 'PHOT' in use_sim_files:
                        print('importing spectrum used for simulated photometry')
                        self.PHOT_sim_spectrum_wlen,self.PHOT_sim_spectrum_flux = open_spectrum(data_dir+'/'+files)
                if 'cov' in files and 'RES' in use_sim_files:
                    RES_err_dir = data_dir+'/'+files
        
        if 'PHOT' in use_sim_files and PHOT_flux_dir is not None:
            print('IMPORTING PHOTOMETRY')
            self.PHOT_data_filter = open_filter_dir(file_dir=PHOT_filter_dir,true_dir = True)
            for instr in self.PHOT_data_filter.keys():
                self.PHOT_filter_function[instr] = interp1d(self.PHOT_data_filter[instr][0],self.PHOT_data_filter[instr][1],bounds_error=False,fill_value=0.)
                if PHOT_flux_format == 4:
                    self.PHOT_data_flux[instr],self.PHOT_data_err[instr],self.PHOT_filter_midpoint[instr],self.PHOT_filter_width[instr] = open_photometry(PHOT_flux_dir+'/'+instr + '.txt')
                elif PHOT_flux_format == 2:
                    self.PHOT_data_flux[instr],self.PHOT_data_err[instr] = open_photometry(PHOT_flux_dir+'/'+instr+'.txt')
                    self.PHOT_filter_midpoint[instr] = calc_median_filter(self.PHOT_filter_function[instr],N_points = 2000*(not quick_load) + 500*quick_load)
                    self.PHOT_filter_width[instr] = effective_width_filter(self.PHOT_filter_function[instr],N_points = 2000*(not quick_load) + 500*quick_load)
                else:
                    # -2 for data
                    mag,mag_err,self.PHOT_data_flux[instr],self.PHOT_data_err[instr] = open_photometry(PHOT_flux_dir+'/'+instr+'.txt')
                    self.PHOT_filter_midpoint[instr] = calc_median_filter(self.PHOT_filter_function[instr],N_points = 2000*(not quick_load) + 500*quick_load)
                    self.PHOT_filter_width[instr] = effective_width_filter(self.PHOT_filter_function[instr],N_points = 2000*(not quick_load) + 500*quick_load)
        
        
        if 'PHOT' in use_sim_files:
            if len(self.PHOT_filter_midpoint.keys())==0:
                for instr in self.PHOT_data_filter.keys():
                    self.PHOT_filter_function[instr] = interp1d(self.PHOT_data_filter[instr][0],self.PHOT_data_filter[instr][1],bounds_error=False,fill_value=0.)
                    if data_dir is None or PHOT_sim_files_format is not None:
                        self.PHOT_filter_midpoint[instr] = calc_median_filter(self.PHOT_filter_function[instr],N_points = 2000)
                        self.PHOT_filter_width[instr] = effective_width_filter(self.PHOT_filter_function[instr],N_points = 2000)
        
        if 'CC' in use_sim_files and CC_data_dir is not None:
            print('IMPORTING CONTINUUM-REMOVED SPECTRUM')
            if '.txt' in CC_data_dir:
                spectrum_name = CC_data_dir[index_last_slash(CC_data_dir):-4]
                self.CC_data_wlen[spectrum_name],self.CC_data_flux[spectrum_name] = open_spectrum(CC_data_dir)
            else:
                for file in os.listdir(CC_data_dir):
                    spectrum_name = file[:-4]
                    self.CC_data_wlen[spectrum_name],self.CC_data_flux[spectrum_name] = open_spectrum(CC_data_dir + '/'+file)
            
            for spectrum_name in self.CC_data_wlen.keys():
                self.CC_data_N[spectrum_name] = len(self.CC_data_wlen[spectrum_name])
                self.CC_data_sf2[spectrum_name] = np.sum(np.array(self.CC_data_flux[spectrum_name])**2)/self.CC_data_N[spectrum_name]
                print('Length of SINFONI data',len(self.CC_data_wlen[spectrum_name]))
        
        if 'RES' in use_sim_files and RES_data_dir is not None:
            print('IMPORTING CONTINUUM-INCLUDED SPECTRUM')
            if '.txt' in RES_data_dir:
                spectrum_name = RES_data_dir[index_last_slash(RES_data_dir):-4]
                data = open_spectrum(RES_data_dir)
                print(np.shape(data))
                print(data)
                self.RES_data_wlen[spectrum_name],self.RES_data_flux[spectrum_name] = data
            else:
                for file in os.listdir(RES_data_dir):
                    spectrum_name = file[:-4]
                    print(spectrum_name)
                    self.RES_data_wlen[spectrum_name],self.RES_data_flux[spectrum_name] = open_spectrum(RES_data_dir + '/'+file)
        
        # calculates the error on the continuum-included data
        if 'RES' in use_sim_files and RES_err_dir is not None:
            temp_err = {}
            if '.txt' in RES_err_dir:
                spectrum_name = RES_err_dir[index_last_slash(RES_err_dir):-4]
                assert(len(self.RES_data_wlen.keys()) == 1)
                if spectrum_name != list(self.RES_data_wlen.keys())[0]:
                    spectrum_name = list(self.RES_data_wlen.keys())[0]
                temp_err[spectrum_name] = open_spectrum(RES_err_dir)
            else:
                for file in os.listdir(RES_err_dir):
                    spectrum_name = file[:-4]
                    temp_err[spectrum_name] = open_spectrum(RES_err_dir+'/'+file)
            for spectrum_name in temp_err.keys():
                print('COV matrix shape ',np.shape(temp_err[spectrum_name]))
                if len(np.shape(temp_err[spectrum_name])) == 1:
                    dim = len(temp_err[spectrum_name])
                    result = np.zeros((dim,dim))
                    inverse = np.zeros((dim,dim))
                    for i in range(dim):
                        result[i,i] = temp_err[spectrum_name][i]**2
                        assert(temp_err[spectrum_name][i] != 0)
                        inverse[i,i] = 1./temp_err[spectrum_name][i]
                    self.RES_cov_err[spectrum_name] = result
                    self.RES_inv_cov[spectrum_name] = inverse
                else:
                    self.RES_cov_err[spectrum_name] = temp_err[spectrum_name]
                    self.RES_inv_cov[spectrum_name] = inv(temp_err[spectrum_name])
                self.RES_data_flux_err[spectrum_name] = np.array([np.sqrt(self.RES_cov_err[spectrum_name][i][i]) for i in range(len(self.RES_cov_err[spectrum_name]))])
        
        return
    
    def getCCSpectrum(self):
        return self.CC_data_wlen,self.CC_data_flux
    def getRESSpectrum(self):
        return self.RES_data_wlen,self.RES_data_flux,self.RES_cov_err,self.RES_inv_cov,self.RES_data_flux_err
    def getSimSpectrum(self):
        return self.PHOT_sim_spectrum_wlen,self.PHOT_sim_spectrum_flux
    def getPhot(self):
        return self.PHOT_data_flux,self.PHOT_data_err,self.PHOT_data_filter,self.PHOT_filter_function,self.PHOT_filter_midpoint,self.PHOT_filter_width
    
    def CCinDATA(self):
        if isinstance(self.CC_data_wlen,dict):
            return len(self.CC_data_wlen.keys())>0
        else:
            return self.CC_data_wlen is not None
    def RESinDATA(self):
        if isinstance(self.RES_data_wlen,dict):
            return len(self.RES_data_wlen.keys())>0
        else:
            return self.RES_data_wlen is not None
    def PHOTinDATA(self):
        if isinstance(self.PHOT_data_filter,dict):
            return len(self.PHOT_data_filter.keys())>0
        else:
            return self.PHOT_data_filter is not None
    
    # calculates necessary range for retrieval using lbl mode
    
    def wlen_details(self,wlen):
        wlen_range = [wlen[0],
                         wlen[-1]]
        wlen_stepsize = max([wlen[i+1]-wlen[i] for i in range(len(wlen)-1)])
        return wlen_range,wlen_stepsize
    
    def min_max_range(self,wlen_dict):
        return [
                min([wlen_dict[key][0] for key in wlen_dict.keys()]),
               max([wlen_dict[key][-1] for key in wlen_dict.keys()])]
    def max_stepsize(self,wlen_dict):
        return max([max([wlen_dict[key][i+1]-wlen_dict[key][i] for i in range(len(wlen_dict[key])-1)]) for key in wlen_dict.keys()])
    
    def wlen_details_lbl(self):
        wlen_range_CC,wlen_stepsize_CC = {},{}
        wlen_range_RES,wlen_stepsize_RES = {},{}
        if self.CCinDATA():
            for key in self.CC_data_wlen.keys():
                wlen_range_CC[key],wlen_stepsize_CC[key] = self.wlen_details(self.CC_data_wlen[key])
        if self.RESinDATA():
            for key in self.RES_data_wlen.keys():
                wlen_range_RES[key],wlen_stepsize_RES[key] = self.wlen_details(self.RES_data_wlen[key])
        if self.CCinDATA() and self.RESinDATA():
            outer_wlen_range_CC=self.min_max_range(wlen_range_CC)
            outer_wlen_range_RES=self.min_max_range(wlen_range_RES)
            outer_wlen_range = [min(outer_wlen_range_CC[0],outer_wlen_range_RES[0]),
                                max(outer_wlen_range_CC[1],outer_wlen_range_RES[1])]
            # actually only care about CC stepsize
            #larger_stepsize = max(self.max_stepsize(self.CC_data_wlen),self.max_stepsize(self.RES_data_wlen))
            larger_stepsize = self.max_stepsize(self.CC_data_wlen)
            return outer_wlen_range,larger_stepsize
        elif self.CCinDATA():
            return self.min_max_range(wlen_range_CC),self.max_stepsize(self.CC_data_wlen)
        elif self.RESinDATA():
            return self.min_max_range(wlen_range_RES),self.max_stepsize(self.RES_data_wlen)
        else:
            return None,None
    
    
    # calculates necessary range for retrieval using c-k mode
    
    def wlen_range_ck(self):
        wlen_range_PHOT = None
        if self.PHOTinDATA():
            MinWlen = min([self.PHOT_data_filter[instr][0][0] for instr in self.PHOT_data_filter.keys()])
            MaxWlen = max([self.PHOT_data_filter[instr][0][-1] for instr in self.PHOT_data_filter.keys()])
            wlen_range_PHOT = [MinWlen,MaxWlen]
        return wlen_range_PHOT
    
    def distribute_FMs(self):
        self.ck_FM_interval = None
        self.RES_data_with_ck = False
        self.RES_data_info = {}
        self.CC_data_info = {}
        
        self.disjoint_lbl_intervals = {}
        self.RES_to_lbl_intervals = {}
        self.CC_to_lbl_intervals = {}
        
        all_lbl_intervals = {}
        interval_naming = 0
        
        if self.PHOTinDATA():
            self.ck_FM_interval = self.wlen_range_ck()
        
        if not (self.CCinDATA() or self.RESinDATA()):
            return 
        
        # collect all intervals of data that need a forward model with lbl mode, and those that need c-k mode
        if self.CCinDATA():
            for instr in self.CC_data_wlen.keys():
                all_lbl_intervals[interval_naming] = ['CC',instr,[self.CC_data_wlen[instr][0],self.CC_data_wlen[instr][-1]]]
                self.CC_data_info[instr] = ['lbl',[self.CC_data_wlen[instr][0],self.CC_data_wlen[instr][-1]],max(self.CC_data_wlen[instr][1:] - self.CC_data_wlen[instr][:-1])]
                interval_naming += 1
        if self.RESinDATA():
            for instr in self.RES_data_wlen.keys():
                max_resolution = max(self.RES_data_wlen[instr][1:]/(self.RES_data_wlen[instr][1:] - self.RES_data_wlen[instr][:-1]))
                if max_resolution < 900:
                    self.RES_data_info[instr] = ['c-k',[self.RES_data_wlen[instr][0],self.RES_data_wlen[instr][-1]],max(self.RES_data_wlen[instr][1:] - self.RES_data_wlen[instr][:-1])]
                    self.RES_data_with_ck = True
                else:
                    self.RES_data_info[instr] = ['lbl',[self.RES_data_wlen[instr][0],self.RES_data_wlen[instr][-1]],max(self.RES_data_wlen[instr][1:] - self.RES_data_wlen[instr][:-1])]
                    all_lbl_intervals[interval_naming] = ['RES',instr,[self.RES_data_wlen[instr][0],self.RES_data_wlen[instr][-1]]]
                    interval_naming += 1
        
        # increase range of c-k FM
        if self.RESinDATA():
            if not self.PHOTinDATA():
                self.ck_FM_interval = [min([self.RES_data_wlen[key][0] for key in self.RES_data_wlen.keys()]),max([self.RES_data_wlen[key][-1] for key in self.RES_data_wlen.keys()])]
            else:
                print(self.ck_FM_interval)
                for key in self.RES_data_wlen.keys():
                    if self.RES_data_info[key][0] == 'c-k':
                        if self.RES_data_wlen[key][0] < self.ck_FM_interval[0]:
                            self.ck_FM_interval[0] = self.RES_data_wlen[key][0]
                        if self.RES_data_wlen[key][-1] > self.ck_FM_interval[1]:
                            self.ck_FM_interval[1] = self.RES_data_wlen[key][-1]
        
        
        final_intervals = {}
        for key in all_lbl_intervals.keys():
            final_intervals[key] = all_lbl_intervals[key][2]
        
        # merge intervals of data requiring lbl mode that overlap
        if len(final_intervals.keys()) > 1:
            working = True
            while working:
                for key_i,key_j in combinations(final_intervals.keys(),2):
                    if key_i == key_j:
                        continue
                    if do_arr_intersect(final_intervals[key_i],final_intervals[key_j]):
                        final_intervals[key_i] = [min(final_intervals[key_i][0],final_intervals[key_j][0]),max(final_intervals[key_i][-1],final_intervals[key_j][-1])]
                        final_intervals.pop(key_j)
                        break
                else:
                    working=False
        
        self.disjoint_lbl_intervals = final_intervals
        self.disjoint_lbl_intervals_max_stepsize = {key:0 for key in self.disjoint_lbl_intervals.keys()}
        
        if self.CCinDATA():
            for key in self.CC_data_wlen.keys():
                for interval_i in self.disjoint_lbl_intervals.keys():
                    if do_arr_intersect(self.CC_data_wlen[key],self.disjoint_lbl_intervals[interval_i]):
                        self.CC_to_lbl_intervals[key] = interval_i
                        self.disjoint_lbl_intervals_max_stepsize[interval_i] = max(self.disjoint_lbl_intervals_max_stepsize[interval_i],self.CC_data_info[key][2])
                        break
        
        if self.RESinDATA():
            for key in self.RES_data_wlen.keys():
                if self.RES_data_info[key][0] == 'lbl':
                    for interval_i in self.disjoint_lbl_intervals.keys():
                        if do_arr_intersect(self.RES_data_wlen[key],self.disjoint_lbl_intervals[interval_i]):
                            self.RES_to_lbl_intervals[key] = interval_i
                            self.disjoint_lbl_intervals_max_stepsize[interval_i] = max(self.disjoint_lbl_intervals_max_stepsize[interval_i],self.RES_data_info[key][2])
                            break
    
    def calculate_KDF_weights(self,h,plot_weights=True,output_dir=''):
        # fix dic keys to avoid mix up in case keys are not always output the same way
        RES_keys = list(self.RES_data_wlen.keys())
        #CC_keys = list(self.CC_data_wlen.keys())
        PHOT_keys = list(self.PHOT_filter_midpoint.keys())
        # stack data on top of each other, sorting according to wavelength doesn't matter
        data_put_together = np.hstack([[self.RES_data_wlen[key],self.RES_data_flux[key]] for key in RES_keys])
        data_put_together = np.hstack([data_put_together,np.transpose([[self.PHOT_filter_midpoint[key],self.PHOT_data_flux[key]] for key in PHOT_keys])])
        
        data = data_put_together[0][:,np.newaxis]
        
        kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(data)
        weights = np.exp(kde.score_samples(data))
        
        new_err_RES = {}
        new_cov_RES = {}
        new_cov_inv_RES = {}
        new_err_PHOT = {}
        
        for key_i,key in enumerate(RES_keys):
            if key_i == 0:
                pos = 0
            else:
                pos = sum([len(self.RES_data_wlen[subkey]) for subkey in RES_keys[:key_i-1]])
            weights_instr = weights[pos:pos+len(self.RES_data_wlen[key])]
            new_cov_RES[key] = np.dot(np.transpose(np.diag(weights_instr)),np.dot(self.RES_cov_err[key],np.diag(weights_instr)))
            new_cov_inv_RES[key] = inv(new_cov_RES[key])
            new_err_RES[key] = np.array([np.sqrt(new_cov_RES[key][i,i]) for i in range(len(new_cov_RES[key]))])
            
            self.RES_cov_err[key] = new_cov_RES[key]
            self.RES_inv_cov[key] = new_cov_inv_RES[key]
            self.RES_data_flux_err[key] = new_err_RES[key]
        
        len_RES_data = sum([len(self.RES_data_wlen[subkey]) for subkey in RES_keys])
        
        for instr_i,instr in enumerate(PHOT_keys):
            pos = len_RES_data+instr_i
            weight_instr = weights[pos]
            new_err_PHOT[instr] = weight_instr*self.PHOT_data_err[instr]
            
            self.PHOT_data_err[instr] = new_err_PHOT[instr]
        
        if plot_weights:
            x_plot = np.linspace(np.min(data),np.max(data),2000)[:, np.newaxis]
            
            log_likelihood = kde.score_samples(x_plot)
            plt.figure(figsize=(16,8))
            plt.subplot(2,1,1)
            
            plt.plot(data_put_together[0],data_put_together[1],'k.')
            plt.ylabel('Flux')
            plt.subplot(2,1,2)
            plt.plot(x_plot[:,0],np.exp(log_likelihood),'b')
            plt.xlabel('Wavelength [$\mu$m]')
            plt.ylabel('Kernel density')
            plt.savefig(output_dir + 'kernel_density.png',dpi=300)
        
        return
    
    def plot(self,config,
             output_dir='',
             plot_name = 'plot',
             title = 'Spectrum',
             inset_plot=True):
        
        plot_data(config,
                  CC_wlen       = self.CC_data_wlen,
                  CC_flux       = self.CC_data_flux,
                  RES_wlen      = self.RES_data_wlen,
                  RES_flux      = self.RES_data_flux,
                  RES_flux_err  = self.RES_cov_err,
                  PHOT_midpoint = self.PHOT_filter_midpoint,
                  PHOT_width    = self.PHOT_filter_width,
                  PHOT_flux     = self.PHOT_data_flux,
                  PHOT_flux_err = self.PHOT_data_err,
                  PHOT_filter   = self.PHOT_data_filter,
                  #PHOT_sim_wlen = self.PHOT_sim_spectrum_wlen,
                  #PHOT_sim_flux = self.PHOT_sim_spectrum_flux,
                  inset_plot    = inset_plot,
                  output_file   = output_dir,
                  title         = title,
                  plot_name     = plot_name)


def index_last_slash(string):
    if '/' not in string:
        return 0
    else:
        return len(string)-string[::-1].index('/')

def do_arr_intersect(arr1,arr2):
    # whether arrays arr1 and arr2 almost intersect (by 0.1 micron)
    a=arr1[0]
    b=arr1[-1]
    c=arr2[0]
    d=arr2[-1]
    return not (c > b + 0.1 or a > d + 0.1)
    