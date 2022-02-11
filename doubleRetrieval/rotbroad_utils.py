# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 15:42:22 2021

@author: jeanh
"""
import numpy as np
from PyAstronomy.pyasl import crosscorrRV,fastRotBroad,rotBroad
import scipy.constants as cst
import matplotlib.pyplot as plt

from doubleRetrieval.rebin import doppler_shift


def trim_spectrum(wlen,flux,wlen_data,threshold=5000,keep=1000):
    wvl_stepsize = np.mean([wlen_data[i+1]-wlen_data[i] for i in range(len(wlen_data)-1)])
    nb_bins_left = int(abs(wlen_data[0]-wlen[0])/wvl_stepsize)
    nb_bins_right = int(abs(wlen_data[-1]-wlen[-1])/wvl_stepsize)
    cut_right=False
    cut_left=False
    if nb_bins_left >= threshold:
        nb_bins_left -= keep
        cut_left=True
    if nb_bins_right >= threshold:
        nb_bins_right -= keep
        cut_right=True
    if cut_right or cut_left:
        CC_wlen_cut,CC_flux_cut = cut_spectrum(wlen,flux,nb_bins_left,nb_bins_right)
        return CC_wlen_cut,CC_flux_cut
    else:
        return wlen,flux
    
    

def cut_spectrum(wlen,flux,nb_bins_left,nb_bins_right):
    if nb_bins_left + nb_bins_right > len(wlen):
        return wlen,flux
    wlen_cut,flux_cut = np.transpose([[wlen[i],flux[i]] for i in range(len(wlen)) if i >= nb_bins_left and i <= len(wlen) - 1 - nb_bins_right])
    return wlen_cut,flux_cut

def add_rot_broad(wlen,flux,rot_vel,method='fast',edgeHandling = 'cut'):
    
    if method == 'fast':
        flux_broad = fastRotBroad(wlen,flux,0,rot_vel)
    else:
        # method == 'slow'
        flux_broad = rotBroad(wlen,flux,0,rot_vel)
    
    if edgeHandling=='cut':
        skipping = abs(wlen[-1]*rot_vel*1000/cst.c)
        wvl_stepsize=np.mean([wlen[i+1]-wlen[i] for i in range(len(wlen)-1)])
        skipped_bins = int(skipping/wvl_stepsize)
        #print('Bins skipped for rotational broadening {b}'.format(b=skipped_bins))
        wlen_cut,flux_cut = cut_spectrum(wlen,flux_broad,nb_bins_left = skipped_bins,nb_bins_right=skipped_bins)
        return wlen_cut,flux_cut
    else:
        # edgeHandling == 'keep'
        return wlen,flux_broad

def plot_retrieved_rotbroad(
        samples,
        dRV_data,
        CC_data,
        wlen,
        flux,
        config,
        output_dir = '',
        fontsize=15
        ):
    
    rvmin,rvmax,drv = min(dRV_data),max(dRV_data),abs(dRV_data[1]-dRV_data[0])
    skipping = {}
    for key in wlen.keys():
        wvl_stepsize = max([wlen[key][i+1]-wlen[key][i] for i in range(len(wlen[key])-1)])
        ds_max = max([
            abs(wlen[key][-1]*rvmin*1000/cst.c),
            abs(wlen[key][-1]*rvmax*1000/cst.c)
            ])
        skipping[key] = int(ds_max/wvl_stepsize)+1
        if skipping[key]/len(wlen[key]) >= 0.25:
            print('WARNING: NEED TO SKIP {p:.2f} % OF TEMPLATE SPECTRUM TO INVESTIGATE ALL DOPPLER-SHIFT DURING CROSS-CORRELATION'.format(p=skipping[key]/len(wlen[key])))
    
    skipedge = int(max([skipping[key] for key in skipping.keys()])*1.5)
    
    
    nb_positions = len(samples)
    nb_params = len(samples[0])
    assert(nb_params==1)
    quantiles = {}
    for param_i,param in config['PARAMS_NAMES']:
        quantiles[param] = np.quantile(samples[param_i][0],q=[0.16,0.5,0.84])
    
    wlen_temp,flux_temp = {},{}
    for key in wlen.keys():
        wlen_temp[key],flux_temp[key] = wlen[key],flux[key]
    plt.figure()
    for quant_i,quant in enumerate([0.16,0.5,0.84]):
        CC_range_i = {}
        for key_i,key in enumerate(wlen.keys()):
            print('Progress: {p:.2f} %'.format(p=int(100*(key_i+1)/len(wlen.keys()))))
            if 'radial_vel' in config['PARAMS_NAMES'] and quant == 0.5:
                wlen_temp[key],flux_temp[key] = doppler_shift(wlen_temp[key],flux_temp[key],quantiles['radial_vel'][quant_i])
            if 'spin_vel' in config['PARAMS_NAMES']:
                flux_temp[key] = fastRotBroad(wlen_temp[key],flux_temp[key],0,quantiles['spin_vel'][quant_i])
            dRV,CC_range_i[key] = crosscorrRV(wlen[key],flux[key],
                                   wlen_temp[key],flux_temp,
                                   rvmin=rvmin,rvmax=rvmax,drv=drv,skipedge=skipedge)
        CC = np.array([sum([CC_range_i[key][drv_i] for key in CC_range_i.keys()]) for drv_i in range(len(dRV))])
        CC = CC/max(CC)
        RV_max_i = np.argmax(CC_data)
        
        
        if quant_i == 1:
            median_str,q2_str,q1_str = {},{},{}
            for param in config['PARAMS_NAMES']:
                median_str[param] = '{median:.2f}'.format(median=quantiles[param][1])
                q2_str[param] = '{q2:.2f}'.format(q2=quantiles[param][2]-quantiles[param][1])
                q1_str[param] = '{q1:.2f}'.format(q1=quantiles[param][1]-quantiles[param][0])
            plt.plot(dRV,CC,'k',label='Retrieved v$_{spin}$ = '+median_str['spin_vel']+'$^{+'+q2_str['spin_vel']+'}_{-'+q1_str['spin_vel']+'}$ km$s^{-1}$')
            plt.axvline(quantiles['radial_vel'][1],color='g',label='Retrieved RV: '+median_str['radial_vel']+'$^{+'+q2_str['radial_vel']+'}_{-'+q1_str['radial_vel']+'}$ km$s^{-1}$')
        else:
            plt.plot(dRV,CC,'k--')
            plt.axvline(quantiles['radial_vel'][quant_i],color='g',ls='--')
            plt.axvline(quantiles['radial_vel'][quant_i],color='g',ls='--')
    CC_data = CC_data/max(CC_data)
    if 'spin_vel' in config['PARAMS_NAMES']:
        plt.plot(dRV_data,CC_data,'r',label='True v$_{spin}$: '+str(config['DATA_PARAMS']['spin_vel'])+' km$s^{-1}$')
    else:
        plt.plot(dRV_data,CC_data,'r')
    if 'radial_vel' in config['PARAMS_NAMES']:
        plt.axvline(config['DATA_PARAMS']['radial_vel'],color='r',ls='--')
    plt.legend(fontsize=fontsize)
    plt.xlabel('Radial velocity [kms$^{-1}$]')
    plt.ylabel('Normalised CCF')
    plt.savefig(output_dir+'retrieved_rot_broad.png',dpi=300)


def calc_SNR(dRV,CC):
    RV_max_i = np.argmax(CC)
    CC_max = CC[RV_max_i]
    
    left_bord,right_bord=RV_max_i-1,RV_max_i+1
    
    while left_bord > 0 and right_bord < len(dRV)-1:
        if CC[left_bord-1] > CC[left_bord] and CC[right_bord+1] > CC[right_bord]:
            break
        else:
            if CC[left_bord-1] < CC[left_bord]:
                left_bord -= 1
            if CC[right_bord+1] < CC[right_bord]:
                right_bord += 1
    nb_CC_bins = len(dRV)
    left_bord = RV_max_i - int(nb_CC_bins/10)
    right_bord = RV_max_i + int(nb_CC_bins/10)
    
    noisy_CC_function = [CC[i] for i in range(len(CC)) if i not in range(left_bord,right_bord)]
    std_CC = np.std(noisy_CC_function)
    SNR = CC_max/std_CC
    return SNR,std_CC,RV_max_i,left_bord,right_bord,noisy_CC_function