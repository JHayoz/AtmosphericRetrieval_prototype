# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 09:40:17 2021

@author: jeanh
"""

# help functions to plot the results of the retrieval

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,mark_inset)
from corner import corner
import numpy as np
from scipy.ndimage import gaussian_filter
import os
from os import path
from random import sample
from PyAstronomy.pyasl import crosscorrRV,fastRotBroad,rotBroad
import scipy.constants as cst
from seaborn import color_palette

from doubleRetrieval.util import *
from doubleRetrieval.model2 import *
from doubleRetrieval.rebin import *
from doubleRetrieval.plot_errorbars_abundances import Errorbars_plot,Posterior_Classification_Errorbars


def plot_data(config,
              CC_wlen = None,
              CC_flux = None,
              CC_wlen_w_cont = None,
              CC_flux_w_cont = None,
              model_CC_wlen = None,
              model_CC_flux = None,
              sgfilter = None,
              RES_wlen = None,
              RES_flux = None,
              RES_flux_err = None,
              model_RES_wlen = None,
              model_RES_flux = None,
              PHOT_midpoint = None,
              PHOT_width = None,
              PHOT_flux = None,
              PHOT_flux_err = None,
              PHOT_filter = None,
              PHOT_sim_wlen = None,
              PHOT_sim_flux = None,
              model_PHOT_flux = None,
              inset_plot = True,
              output_file = None,
              plot_name='plot',
              title = 'Spectrum',
              fontsize=15):
    
    wvl_label = 'Wavelength [$\mu$m]'
    filter_label = 'Filter trsm.'
    flux_label = 'Flux [Wm$^{-2}\mu$m$^{-1}$]'
    CC_flux_label = 'Residuals [Wm$^{-2}\mu$m$^{-1}$]'
    
    
    
    # change arguments to None if they are empty dictionaries
    if isinstance(CC_wlen,dict):
        if len(CC_wlen.keys()) == 0:
            CC_wlen,CC_flux,sgfilter,CC_wlen_w_cont,CC_flux_w_cont=None,None,None,None,None
    if isinstance(RES_wlen,dict):
        if len(RES_wlen.keys()) == 0:
            RES_wlen,RES_flux,RES_flux_err,model_RES_wlen,model_RES_flux=None,None,None,None,None
    if isinstance(PHOT_flux,dict):
        if len(PHOT_flux.keys()) == 0:
            PHOT_midpoint,PHOT_width,PHOT_flux,PHOT_flux_err,PHOT_filter,PHOT_sim_wlen,PHOT_sim_flux,model_PHOT_flux=None,None,None,None,None,None,None,None
    
    
    
    nb_plots = (PHOT_filter is not None) + 2*(PHOT_flux is not None or PHOT_sim_wlen is not None) + (CC_flux is not None) + (sgfilter is not None or CC_wlen_w_cont is not None) #+ 2*(RES_wlen is not None)
    
    fig = plt.figure(figsize=(10,2*nb_plots))
    
    plot_i = 1
    """ZERO-TH PLOT"""
    
    # determine order of filters wrt filter midpoint
    if PHOT_flux is not None:
        
        filter_pos = filter_position(PHOT_midpoint)
        # give photometric fluxes a nice color
        rgba = {}
        cmap = color_palette('colorblind',n_colors = len(PHOT_flux.keys()),as_cmap = True)
        
        for instr in PHOT_flux.keys():
            rgba[instr] = cmap[filter_pos[instr]%len(cmap)]
    
        
    
    """FILTER TRANSMISSION FUNCTIONS"""
    
    if PHOT_filter is not None and PHOT_flux is not None:
        
        #print('new plot',plot_i)
        ax = plt.subplot(nb_plots,1,plot_i) 
        ax.set_title(title,fontsize=fontsize)
        x_min = 100
        x_max = 0
        for instr in PHOT_filter.keys():
            ax.plot(PHOT_filter[instr][0],PHOT_filter[instr][1],color=rgba[instr])
            x_min = min(x_min,PHOT_filter[instr][0][0])
            x_max = max(x_max,PHOT_filter[instr][0][-1])
        
        
        ax.set_xlabel(wvl_label,fontsize=fontsize)
        ax.set_ylabel(filter_label,fontsize=fontsize)
        ax.tick_params(axis='both',which='both',labelsize=fontsize-2)
        ax.set_xlim((x_min-0.2,x_max+0.2))
        
        plot_i += 1
    
    
    """FIRST PLOT"""
    if RES_flux_err is not None:
        YERR = None
        if isinstance(RES_flux_err,dict):
            YERR = {}
            for key in RES_flux_err.keys():
                if len(np.shape(RES_flux_err[key])) >= 2:
                    # if it's a covariance matrix
                    YERR[key] = [np.sqrt(RES_flux_err[key][i][i]) for i in range(len(RES_flux_err[key]))]
                else:
                    # if it's a vector of error
                    YERR[key] = RES_flux_err[key]
        else:
            if len(np.shape(RES_flux_err)) >= 2:
                # if it's a covariance matrix
                YERR = [np.sqrt(RES_flux_err[i][i]) for i in range(len(RES_flux_err))]
            else:
                # if it's a vector of error
                YERR = RES_flux_err
    
    if PHOT_flux is not None or PHOT_sim_wlen is not None:
        
        ax = plt.subplot(nb_plots,1,(plot_i,plot_i+1))
        #print('new plot',plot_i)
        
        
        
        """RESIDUALS DATA"""
        if RES_flux_err is not None:
            ax = custom_errorbar(ax,RES_wlen,RES_flux,xerr=None,yerr = YERR,fmt='|',color='b',alpha=0.5,capsize=2, elinewidth=1, markeredgewidth=1,zorder=1)
        
        if RES_wlen is not None:
        #    ax = custom_plot(ax,RES_wlen,RES_flux,color='b',alpha=0.5,marker='+',label='GRAVITY spectrum',zorder=1)
            if model_RES_wlen is not None:
                ax = custom_plot(ax,model_RES_wlen,model_RES_flux,color='r',label='Retrieved GRAVITY spectrum',zorder=1)
        
        """PHOTOMETRIC DATA"""
        
        if PHOT_flux is not None:
            
            for instr in PHOT_flux.keys():
                yerr = None
                if PHOT_flux_err is not None:
                    yerr = PHOT_flux_err[instr]
                ax.errorbar(PHOT_midpoint[instr],PHOT_flux[instr],xerr = PHOT_width[instr]/2, yerr = yerr,color=rgba[instr],zorder=2)
                
            if model_PHOT_flux is not None:
                for instr in model_PHOT_flux.keys():
                    ax.errorbar(PHOT_midpoint[instr],model_PHOT_flux[instr],xerr = PHOT_width[instr]/2,color='r',zorder=2)
        
        """SIMULATED SPECTRUM FOR SIMULATED DATA"""
        
        if PHOT_sim_wlen is not None:
            ax = custom_plot(ax,PHOT_sim_wlen,PHOT_sim_flux,color='k',label='Simulated spectrum')
            
        if PHOT_flux is not None or PHOT_sim_wlen is not None:
            #ax.legend(fontsize=fontsize)
            ax.set_xlabel(wvl_label,fontsize=fontsize)
            ax.set_ylabel(flux_label,fontsize=fontsize)
            ax.tick_params(axis='both',which='both',labelsize=fontsize-2)
            ax.set_xlim((x_min-0.2,x_max+0.2))
            plot_i += 2
    
    
    if RES_wlen is not None:
        
        if inset_plot:
            """SECOND PLOT"""
            """NOW INSET IN FIRST PLOT"""
            ax2 = plt.axes([2.8,9,1,1])
            ip = InsetPosition(ax, [0.45,0.45,0.54,0.49])
            ax2.set_axes_locator(ip)
            mark_inset(ax, ax2, loc1=2, loc2=2, fc="none", ec='0')
            
            #ax = plt.subplot(nb_plots,1,(plot_i,plot_i+1))
            #print('new plot',plot_i)
            """RESIDUAL DATA"""
            if RES_flux_err is not None:
                ax2 = custom_errorbar(ax2,RES_wlen,RES_flux,xerr=None,yerr = YERR,color='b',fmt='|',alpha=0.5,capsize=2, elinewidth=1, markeredgewidth=1,zorder=1)
            
            #ax2 = custom_plot(ax2,RES_wlen,RES_flux,color='b',alpha=0.5,lw = 0.5,label='GRAVITY spectrum')
            
            if model_RES_wlen is not None:
                ax2 = custom_plot(ax2,model_RES_wlen,model_RES_flux,color='r',lw = 0.5,label='Retrieved GRAVITY spectrum',zorder=1)
            if PHOT_sim_wlen is not None:
                ax2 = custom_plot(ax2,PHOT_sim_wlen,PHOT_sim_flux,color='k',lw = 0.5,label='Retrieved spectrum',zorder=1)
            
            if PHOT_flux is not None:
                
                for instr in PHOT_flux.keys():
                    yerr = None
                    if PHOT_flux_err is not None:
                        yerr = PHOT_flux_err[instr]
                    ax2.errorbar(PHOT_midpoint[instr],PHOT_flux[instr],xerr = PHOT_width[instr]/2, yerr = yerr,color=rgba[instr],zorder=2)
            
            #ax2.legend(fontsize=fontsize)
            ax2.set_xlabel(wvl_label,fontsize=fontsize-4)
            ax2.set_ylabel('GRAVITY',fontsize=fontsize-4)
            ax2.tick_params(axis='both',which='both',labelsize=fontsize-4)
            if isinstance(RES_wlen,dict):
                ax2.set_xlim((min([RES_wlen[key][0] for key in RES_wlen.keys()]),max([RES_wlen[key][-1] for key in RES_wlen.keys()])))
                ax2.set_ylim((min([min(RES_flux[key])*0.975 for key in RES_wlen.keys()]),max([max(RES_flux[key])*1.025 for key in RES_wlen.keys()])))
            else:
                ax2.set_xlim((RES_wlen[0],RES_wlen[-1]))
                ax2.set_ylim((min(RES_flux)*0.9,max(RES_flux)*1.1))
            #plot_i += 2
    
    
    if sgfilter is not None and CC_wlen_w_cont is not None:
        
        """THIRD PLOT"""
        
        
        ax = plt.subplot(nb_plots,1,plot_i)
        #print('new plot',plot_i)
        ax = custom_plot(ax,CC_wlen_w_cont,CC_flux_w_cont,color='k',lw=0.5,label='SINFONI spectrum')
        ax = custom_plot(ax,CC_wlen,sgfilter,color='r',lw=0.5,label='Filter')
        
        if model_RES_wlen is not None:
            ax = custom_plot(ax,model_RES_wlen,model_RES_flux,color='blueviolet',lw = 0.5,label='Retrieved GRAVITY spectrum')
        
        if PHOT_sim_wlen is not None:
            ax = custom_plot(ax,PHOT_sim_wlen,PHOT_sim_flux,color='k',ls='--',lw = 0.5,label='Retrieved spectrum')
        
        xlim_min,xlim_max = min([CC_wlen[key][0] for key in CC_wlen.keys()]),max([CC_wlen[key][-1] for key in CC_wlen.keys()])
        ax.set_xlim((xlim_min,xlim_max))
        ax.legend(fontsize=fontsize)
        ax.set_xlabel(wvl_label,fontsize=fontsize)
        ax.set_ylabel(flux_label,fontsize=fontsize)
        ax.tick_params(axis='both',which='both',labelsize=fontsize-2)
        plot_i += 1
        
    
    
    if CC_flux is not None:
        
        """FOURTH PLOT"""
        
        
        ax = plt.subplot(nb_plots,1,plot_i)
        #print('new plot',plot_i)
        """CC DATA"""
        
        ax = custom_plot(ax,CC_wlen,CC_flux,color='k')#,label='SINFONI residuals')
        if model_CC_wlen is not None:
            ax = custom_plot(ax,model_CC_wlen,model_CC_flux,color='r',lw = 0.5,label='Retrieved SINFONI residuals')
        
        #ax.legend(fontsize=fontsize)
        ax.set_xlabel(wvl_label,fontsize=fontsize)
        ax.set_ylabel(CC_flux_label,fontsize=fontsize)
        ax.tick_params(axis='both',which='both',labelsize=fontsize-2)
    
    
    
    if not os.path.exists(output_file):
        try:
            os.mkdir(output_file)
        except FileExistsError:
            print('Error avoided')
    #fig.tight_layout()
    fig.savefig(output_file+'/'+plot_name+'.png',dpi=300,bbox_inches = 'tight',pad_inches = 0)
    fig.savefig(output_file+'/'+plot_name+'.pdf',dpi=600,bbox_inches = 'tight',pad_inches = 0)

def custom_errorbar(ax,x,y,xerr,yerr,**kwargs):
    kwargs['fmt'] = ' '
    if isinstance(x,dict):
        for key in x.keys():
            if xerr is None:
                xerr_temp = None
            else:
                xerr_temp = xerr[key]
            if yerr is None:
                yerr_temp = None
            else:
                yerr_temp = yerr[key]
            ax.errorbar(x[key],y[key],xerr = xerr_temp,yerr = yerr_temp,**kwargs)
            if 'label' in kwargs:
                del kwargs['label']
            
    else:
        ax.errorbar(x,y,xerr = xerr,yerr = yerr,**kwargs)
    return ax


def custom_plot(ax,x,y,**kwargs):
    
    if isinstance(x, dict):
        for key in x.keys():
            ax.plot(x[key],y[key],**kwargs)
            if 'label' in kwargs:
                del kwargs['label']
            
    else:
        ax.plot(x,y,**kwargs)
    return ax

def plot_SNR(config,
             wlen_CC,
             flux_CC,
             CC_wlen_data,
             CC_flux_data,
             output_file = ' ',
             title='C-C function',
             ax = None,
             fontsize=15,
             printing=False):
    
    if not isinstance(wlen_CC,dict):
        wlen_CC,flux_CC = {'data':wlen_CC},{'data':flux_CC}
        if isinstance(CC_wlen_data,dict):
            CC_wlen_data['data'],CC_flux_data['data'] = CC_wlen_data[[key for key in CC_wlen_data.keys()][0]],CC_flux_data[[key for key in CC_wlen_data.keys()][0]]
        else:
            CC_wlen_data,CC_flux_data = {'data':CC_wlen_data},{'data':CC_flux_data}
    else:
        if not isinstance(CC_wlen_data,dict):
            CC_wlen_data,CC_flux_data = {'data':CC_wlen_data},{'data':CC_flux_data}
    dRV_temp,CC_temp={},{}
    for key in CC_wlen_data.keys():
        
        dRV_temp[key],CC_temp[key]=crosscorrRV(
                        CC_wlen_data[key],
                        CC_flux_data[key],
                        wlen_CC[key],
                        flux_CC[key],
                        rvmin=config['RVMIN'],
                        rvmax=config['RVMAX'],
                        drv=config['DRV'])
    
    CC = np.array([sum([CC_temp[key][i] for key in CC_temp.keys()]) for i in range(len(CC_temp[min(CC_temp.keys())]))])
    dRV = dRV_temp[min(dRV_temp.keys())]
    
    CC=CC/len(wlen_CC)
    RV_max_i=np.argmax(CC)
    CC_max = CC[RV_max_i]
    
    SNR,std_CC,RV_max_i,left_bord,right_bord,noisy_CC_function = calc_SNR(dRV,CC)
    
    saving = False
    if ax is None:
        saving = True
        fig = plt.figure()
        ax = fig.gca()
    ax.plot(dRV,CC)
    ax.axvline(dRV[RV_max_i],color='r',label='Retrieved RV={rv}, SNR={snr:.2f}'.format(rv=dRV[RV_max_i],snr=SNR))
    ax.axvline(dRV[left_bord],color = 'b',alpha=0.4)
    ax.axvline(dRV[right_bord],color = 'b',alpha=0.4)
    
    ax.axhline(np.mean(noisy_CC_function) + std_CC,color = 'b',ls='--',alpha=0.4)
    ax.axhline(np.mean(noisy_CC_function) - std_CC,color = 'b',ls='--',alpha=0.4)
    
    ax.legend(fontsize=fontsize)
    ax.set_xlabel('Radial velocity [km$^{-1}$]',fontsize=fontsize)
    ax.set_ylabel('CCF',fontsize=fontsize)
    #plt.title(title)
    if printing:
        save_spectrum(dRV,CC,save_dir= output_file,save_name='/CCF')
    
    if saving:
        fig.savefig(output_file+'CC_function.png',dpi=300)
    else:
        return ax
        

def plot_profiles(pressures,
                  temperatures = None,
                  abundances = None,
                  output_dir='',
                  fontsize=15):
    if (temperatures is None) and (abundances is None):
        return None
    
    nb_plots = (temperatures is not None) + (abundances is not None)
    print('Plotting profiles')
    fig = plt.figure(figsize=(10,10 + (nb_plots-1)*5))
    plt.subplot(nb_plots,1,1)
    cmap = plt.get_cmap('gist_rainbow')
    rgba = {}
    if abundances is not None:
        # sort abundances according to their mean
        abunds_means = {mol:np.mean(abundances[mol]) for mol in abundances.keys()}
        
        if 'H2' in abundances.keys() and 'H2_main_iso' in abundances.keys():
            if abunds_means['H2'] < abunds_means['H2_main_iso']:
                del abunds_means['H2']
                del abundances['H2']
            else:
                del abunds_means['H2_main_iso']
                del abundances['H2_main_iso']
        
        sorted_abunds = sorted(abunds_means.items(),key=lambda kv: (kv[1],kv[0]))
        for i,(mol_name,mol_mean) in enumerate(sorted_abunds):
            rgba[mol_name] = cmap(i/len(abundances.keys()))
    if abundances is not None:
        for element in abundances.keys():
            plt.plot(abundances[element],pressures,color=rgba[element],label=nice_name(element))
        plt.yscale('log')
        plt.xscale('log')
        fig.gca().invert_yaxis()
        plt.xlabel('Abundances (MMR)',fontsize=fontsize)
        plt.ylabel('Pressure [bar]',fontsize=fontsize)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.legend(fontsize=fontsize)
    
    if (abundances is not None) and (temperatures is not None):
        plt.subplot(nb_plots,1,2)
    
    if temperatures is not None:
        plt.plot(temperatures,pressures)
        plt.yscale('log')
        fig.gca().invert_yaxis()
        plt.xlabel('Temperature [K]',fontsize=fontsize)
        plt.ylabel('Pressure [bar]',fontsize=fontsize)
        plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout()
    fig.savefig(output_dir+'/temp_abunds_plot.png',dpi=300)

def plot_corner(config,
                samples,
                param_range = None,
                percent_considered = 0.90,
                output_file = '',
                fontsize=12,
                include_abunds = True,
                title = 'Retrieval'):
    params_names = config['PARAMS_NAMES']
    abundances_names = config['ABUNDANCES']
    
    nb_iter = len(samples)
    index_consider = int(nb_iter*(1.-percent_considered))
    
    samples_cut = samples[index_consider:,:]
    
    if not include_abunds:
        params_names = [param for param in config['PARAMS_NAMES'] if not param in abundances_names]
        samples_cut = samples_cut[:,[index for index in range(len(config['PARAMS_NAMES'])) if config['PARAMS_NAMES'][index] not in abundances_names]]
    
    n_params = len(params_names)
    corner_range=None
    if param_range is not None:
        corner_range=list([(param_range[param][0],param_range[param][1]) if param in config['TEMPS'] + config['UNSEARCHED_TEMPS'] + config['CLOUDS'] else (param_range['abundances'][0],param_range['abundances'][1]) for param in params_names])
    fig = corner(samples_cut, quantiles = [0.16, 0.5, 0.84],show_titles=True,title_kwargs={"fontsize":fontsize},verbose=True,labels=[nice_param_name(param,config) for param in params_names],bins=20,range=corner_range)
    if title is not None:
        fig.suptitle(title,fontsize=12)
    
    axes = np.array(fig.axes).reshape((n_params, n_params))
    for i in range(n_params):
        ax = axes[i, i]
        ax.axvline(np.median(samples_cut, axis=0)[i], color='g')
        if params_names[i] in config['DATA_PARAMS'].keys():
            ax.axvline(config['DATA_PARAMS'][params_names[i]],color='r',label='True: {value:.2f}'.format(value=config['DATA_PARAMS'][params_names[i]]))
            ax.legend(fontsize=6)
    for yi in range(n_params):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(np.median(samples_cut, axis=0)[xi], color='g')
            ax.axhline(np.median(samples_cut, axis=0)[yi], color='g')
            ax.plot(np.median(samples_cut, axis=0)[xi], np.median(samples_cut, axis=0)[yi], 'sg')
            if params_names[xi] in config['DATA_PARAMS'].keys():
                ax.axvline(config['DATA_PARAMS'][params_names[xi]],color='r')
            if params_names[yi] in config['DATA_PARAMS'].keys():
                ax.axhline(config['DATA_PARAMS'][params_names[yi]],color='r')
    if include_abunds:
        fig.savefig(output_file+'full_cornerplot.png',dpi=300)
    else:
        fig.savefig(output_file+'partial_cornerplot.png',dpi=300)

def plot_mol_abunds(config,
                    samples,
                    title = 'Retrieved molecular profiles',
                    output_file = '',
                    ax = None,
                    fontsize = 15):
    # either retrieval was in free or chem_equ model, and same for data, and it's possible that data is unknown
    
    median_params = np.median(samples,axis=0)
    
    ab_metals,temps_params,clouds_params = fix_params(config, median_params)
    
    pressures = np.logspace(-6, temps_params['P0'], 100)
    temperatures = nc.guillot_global(pressures, 1e1**temps_params['log_kappa_IR'], 1e1**temps_params['log_gamma'], 1e1**temps_params['log_gravity'], temps_params['t_int'], temps_params['t_equ'])
    
    # results from retrieval
    
    
    rgba = {}
    cmap = plt.get_cmap('gist_rainbow')
    if config['MODEL'] == 'free':
        for i,mol_name in enumerate(config['ABUNDANCES']):
            rgba[mol_name] = cmap(i/len(config['ABUNDANCES']))
    else:
        if config['MODE'] == 'lbl':
            for i,mol_name in enumerate(poor_mans_abunds_lbl()):
                rgba[mol_name] = cmap(i/len(poor_mans_abunds_lbl()))
        else:
            for i,mol_name in enumerate(poor_mans_abunds_ck()):
                rgba[mol_name] = cmap(i/len(poor_mans_abunds_ck()))
    saving = False
    if ax is None:
        saving = True
        fig = plt.figure()
        ax = fig.gca()
    fontsize=fontsize
    if config['MODEL'] == 'free':
        for param_i,param in enumerate(config['PARAMS_NAMES']):
            if param in config['ABUNDANCES']:
                q1,q2,q3 = np.quantile(samples[:,param_i],q=[(1-0.6827)/2,0.5,1-(1-0.6827)/2])
                sigma_bot = q2-q1
                sigma_top = q3-q2
                ax.plot((10**q2)*np.ones_like(pressures),pressures,color=rgba[param],label=nice_name(param) + ': {q:.2f}'.format(q = q2) +'$^{+'+'{sigma:.2f}'.format(sigma=sigma_top) +'}_{-'+'{sigma:.2f}'.format(sigma=sigma_bot)+'}$')
                #ax.plot((10**q1)*np.ones_like(pressures),pressures,color=rgba[param],ls='--')
                #ax.plot((10**q3)*np.ones_like(pressures),pressures,color=rgba[param],ls='--')
                ax.fill_betweenx(pressures,(10**q1)*np.ones_like(pressures),(10**q3)*np.ones_like(pressures),color=rgba[param],alpha=0.3)
                
    else:
        # chem_equ model
        nb_positions = len(samples)
        nb_random_pos = int(nb_positions/10)
        considered_positions = sample(list(samples),k=nb_random_pos)
        profile_curves = {}
        for pos_i,param_i in enumerate(considered_positions):
            ab_metals,temps_params,clouds_params = fix_params(config, param_i)
            COs = ab_metals['C/O']*np.ones_like(pressures)
            FeHs = ab_metals['FeHs']*np.ones_like(pressures)
            mass_fractions = poor_mans_nonequ_chem.interpol_abundances(
                                    COs,
                                    FeHs,
                                    temperatures,
                                    pressures)
            profile_curves[pos_i] = filter_relevant_mass_fractions(mass_fractions,config['MODE'])
        mol_abunds_names = profile_curves[0].keys()
        for mol_name in mol_abunds_names:
            curve_q1,curve_q2,curve_q3 = np.quantile([profile_curves[i][mol_name] for i in range(len(profile_curves.keys()))],
                                                     q = [(1-0.6827)/2,0.5,1-(1-0.6827)/2],
                                                     axis=0)
            curve_q1_smooth = gaussian_filter(curve_q1,sigma=10)
            curve_q2_smooth = gaussian_filter(curve_q2,sigma=10)
            curve_q3_smooth = gaussian_filter(curve_q3,sigma=10)
            ax.plot(curve_q2_smooth,pressures,color=rgba[mol_name],label=nice_name(mol_name))
            #plt.plot(curve_q1_smooth,pressures,color=rgba[mol_name],ls='--')
            #plt.plot(curve_q3_smooth,pressures,color=rgba[mol_name],ls='--')
            ax.fill_betweenx(pressures,curve_q1_smooth,curve_q3_smooth,color=rgba[mol_name],alpha=0.3)
        
    # simulated data
    # free
    for param_i,param in enumerate(config['PARAMS_NAMES']):
        if param in config['ABUNDANCES']:
            if param in config['DATA_PARAMS'].keys():
                ax.plot((10**config['DATA_PARAMS'][param])*np.ones_like(pressures),pressures,color=rgba[mol_name],ls=':')
    # chem_equ
    if np.prod([el in config['DATA_PARAMS'].keys() for el in ['C/O','FeHs']]) == 1:
        COs = config['DATA_PARAMS']['C/O']*np.ones_like(pressures)
        FeHs = config['DATA_PARAMS']['FeHs']*np.ones_like(pressures)
        mass_fractions = poor_mans_nonequ_chem.interpol_abundances(
                                COs,
                                FeHs,
                                temperatures,
                                pressures)
        profile_curves = filter_relevant_mass_fractions(mass_fractions,config['MODE'])
        for mol_name in profile_curves.keys():
            if mol_name in config['ABUNDANCES']:
                ax.plot(profile_curves[mol_name],pressures,color=rgba[mol_name],ls=':')
                ax.set_xlim((1e-8,1))
            """
            else:
                plt.plot(profile_curves[mol_name],pressures,ls=':',alpha=0.3)
            """
            
    
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.invert_yaxis()
    ax.set_xlim((1e-8,1))
    ax.set_xlabel('Mass Fraction (dex)',fontsize=fontsize)
    ax.set_ylabel('Pressure [bar]',fontsize=fontsize)
    if title is not None:
        ax.set_title(title,fontsize=fontsize)
    ax.legend(fontsize=fontsize,loc='lower right')
    if saving:
        fig.savefig(output_file+'retrieved_abunds.png',dpi=300)
    else:
        return ax
            
        

def plot_temperature(config,
                     samples,
                     title,
                     output_file):
    nb_positions = len(samples)
    data_pressures = np.logspace(-6, config['DATA_PARAMS']['P0'], 100)
    temperature_data = nc.guillot_global(data_pressures, 1e1**config['DATA_PARAMS']['log_kappa_IR'], 1e1**config['DATA_PARAMS']['log_gamma'], 1e1**config['DATA_PARAMS']['log_gravity'], config['DATA_PARAMS']['t_int'], config['DATA_PARAMS']['t_equ'])
    nb_temperature_curves = min([len(samples),1000])/10
    to_pick_from = 0.5
    pick_sample_from = int(nb_positions*(1.-to_pick_from))
    considered_positions = sample(samples[pick_sample_from:],k=min([nb_temperature_curves,nb_positions]))
    temperature_curves = {}
    for i,param in enumerate(considered_positions):
        temp_param = {}
        for name in config['TEMPS']+config['UNSEARCHED_TEMPS']:
            if name in config['TEMPS']:
                temp_param[name] = param[config['PARAMS_NAMES'].index(name)]
            else:
                temp_param[name] = config['DATA_PARAMS'][name]
        temperature_curves[i] = nc.guillot_global(data_pressures, 1e1**temp_param['log_kappa_IR'], 1e1**temp_param['log_gamma'], 1e1**temp_param['log_gravity'], temp_param['t_int'], temp_param['t_equ'])
    temp_curves = np.array([temperature_curves[i] for i in range(len(considered_positions))])
    quantile_curves = np.quantile(temp_curves,[(1-0.6827)/2,0.5,1-(1-0.6827)/2],axis=0)
    quantile_curves_smooth = {}
    for curve_i in range(len(quantile_curves)):
        quantile_curves_smooth[curve_i] = gaussian_filter(quantile_curves[curve_i],sigma=10)
    median_param = {}
    params_median = np.median(samples,axis=0)
    for name in config['TEMPS']+config['UNSEARCHED_TEMPS']:
        if name in config['TEMPS']:
            median_param[name] = params_median[config['PARAMS_NAMES'].index(name)]
        else:
            median_param[name] = config['DATA_PARAMS'][name]
    median_curve = nc.guillot_global(data_pressures, 1e1**median_param['log_kappa_IR'], 1e1**median_param['log_gamma'], 1e1**median_param['log_gravity'], median_param['t_int'], median_param['t_equ'])
    fig=plt.figure()
    if True:
        plt.plot(temperature_data,data_pressures,'k',linewidth=0.3,label='True')
    plt.plot(median_curve,data_pressures,'b',linewidth=0.8,ls='--',label='Median parameters')
    #plt.plot(quantile_curves[0],data_pressures,'lightblue',linewidth=0.3,label='2 $\sigma$')
    plt.plot(quantile_curves_smooth[0],data_pressures,'r',linewidth=0.3,label='1 $\sigma$')
    plt.plot(quantile_curves_smooth[1],data_pressures,'b',linewidth=0.8,label='Median curve')
    plt.plot(quantile_curves_smooth[2],data_pressures,'r',linewidth=0.3)
    #plt.plot(quantile_curves[4],data_pressures,'lightblue',linewidth=0.3)
    plt.fill_betweenx(data_pressures,quantile_curves_smooth[0],quantile_curves_smooth[2],color='r',label='1 $\sigma$')
    #plt.fill_betweenx(data_pressures,quantile_curves[0],quantile_curves[1],color='lightblue')
    #plt.fill_betweenx(data_pressures,quantile_curves[3],quantile_curves[4],color='lightblue')
    if len(config['CLOUDS'])>0:
        log_Pcloud_values = np.quantile(samples[config['PARAMS_NAMES'].index('log_Pcloud')],[(1-0.6827)/2,0.5,1-(1-0.6827)/2])
        plt.axhline(10**log_Pcloud_values[0],color='lightgrey')
        plt.axhline(10**log_Pcloud_values[1],color='grey',label='Cloud deck')
        plt.axhline(10**log_Pcloud_values[2],color='lightgrey')
    plt.yscale('log')
    fig.gca().invert_yaxis()
    plt.xlabel('Temperature [K]',fontsize=8)
    plt.ylabel('Pressure [bar]',fontsize=8)
    plt.legend(fontsize=8)
    fig.savefig(output_file+'temperatureplot.png',dpi=300)

def fix_params(config,
               params):
    params_dico = {}
    for i,name in enumerate(config['PARAMS_NAMES']):
        params_dico[name] = params[i]
    abunds_dico = {}
    temps_dico = {}
    clouds_dico = {}
    for name in config['ALL_PARAMS']:
        if name in config['ABUNDANCES']:
            abunds_dico[name] = params_dico[name]
        if name in config['UNSEARCHED_ABUNDANCES']:
            abunds_dico[name] = config['DATA_PARAMS'][name]
        if name in config['TEMPS']:
            temps_dico[name] = params_dico[name]
        if name in config['UNSEARCHED_TEMPS']:
            print('FOUND',name)
            temps_dico[name] = config['DATA_PARAMS'][name]
        if name in config['CLOUDS']:
            clouds_dico[name] = params_dico[name]
        if name in config['UNSEARCHED_CLOUDS']:
            clouds_dico[name] = config['DATA_PARAMS'][name]
    
    if 'R' in config['TEMPS'] and 'log_R' not in config['TEMPS']:
        print('Changing R to logR in fix_params')
        temps_dico['log_R'] = np.log10(temps_dico['R'])
    
    if 'log_R' not in temps_dico.keys():
        if 'R' in config['ALL_PARAMS']:
            print('Adding logR in fix_params')
            temps_dico['log_R'] = np.log10(config['DATA_PARAMS']['R'])
    
    return abunds_dico,temps_dico,clouds_dico
    
def plot_retrieved_spectra_FM(
        config,  
        samples,
        output_file,
        data_obj,
        forwardmodel_lbl,
        forwardmodel_ck,
        title = 'Retrieval',
        show_random = None,
        saving = True,
        output_results = False):
    
    output_result_dir = output_file + 'retrieved_spectrum/'
    if saving:
        if not os.path.exists(output_result_dir):
            try:
                os.mkdir(output_result_dir)
            except FileExistsError:
                print('Error avoided')
        if not os.path.exists(output_result_dir+'photometry/'):
            try:
                os.mkdir(output_result_dir+'photometry/')
            except FileExistsError:
                print('Error avoided')
    
    CC_wvl_data,CC_flux_data,wlen_CC,flux_CC,sgfilter,RES_wvl_data,RES_flux_data,RES_cov_data,wlen_RES,flux_RES = None,None,None,None,None,None,None,None,None,None,
    PHOT_filter_midpoint,PHOT_filter_width,PHOT_data_flux,PHOT_data_err,filt,wlen_ck,flux_ck,photometry = None,None,None,None,None,None,None,None
    
    
    nb_positions = len(samples)
    med_par = np.median(samples,axis=0)
    
    median_abunds,median_temps,median_clouds = fix_params(config,med_par)
    
    if forwardmodel_lbl is not None:
        
        wlen_lbl,flux_lbl = forwardmodel_lbl.calc_spectrum(
                      ab_metals = median_abunds,
                      temp_params = median_temps,
                      clouds_params = median_clouds,
                      external_pt_profile = None)
        
        if data_obj.CCinDATA():
            CC_wvl_data,CC_flux_data = data_obj.getCCSpectrum()
            data_N,data_sf2 = data_obj.CC_data_N,data_obj.CC_data_sf2
            
            wlen_CC,flux_CC,sgfilter = {},{},{}
            
            if not isinstance(CC_wvl_data,dict):
                CC_wvl_data,CC_flux_data,data_N,data_sf2 = {'key':CC_wvl_data},{'key':CC_flux_data},{'key':data_obj.CC_data_N},{'key':data_obj.CC_data_sf2}
            for key in CC_wvl_data.keys():
                wlen_CC[key],flux_CC[key],sgfilter[key],wlen_rebin,flux_rebin = rebin_to_CC(wlen_lbl,flux_lbl,CC_wvl_data[key],
                                                                         config['WIN_LEN'],filter_method = 'only_gaussian',
                                                                         convert = config['CONVERT_SINFONI_UNITS'],log_R=median_temps['log_R'],distance=config['DISTANCE'])
            
            if saving:
                save_spectra(wlen_CC,flux_CC,save_dir= output_result_dir + 'CC_spectrum',save_name='')
            
        if data_obj.RESinDATA():
            
            RES_wvl_data,RES_flux_data,RES_cov_data,inverse_cov,flux_err_data = data_obj.getRESSpectrum()
            
            wlen_RES,flux_RES = {},{}
            
            if not isinstance(RES_wvl_data,dict):
                RES_wvl_data,RES_flux_data,RES_cov_data,inverse_cov,flux_err_data = {'key':RES_wvl_data},{'key':RES_flux_data},{'key':RES_cov_data},{'key':inverse_cov},{'key':flux_err_data}
            
            for key in RES_wvl_data.keys():
                wlen_RES[key],flux_RES[key] = rebin_to_RES(wlen_lbl,flux_lbl,RES_wvl_data[key],median_temps['log_R'],config['DISTANCE'])
            
            if saving:
                save_spectra(wlen_RES,flux_RES,save_dir= output_result_dir + 'RES_spectrum',save_name='')
    
    if forwardmodel_ck is not None:
        
        wlen_ck,flux_ck = forwardmodel_ck.calc_spectrum(
                      ab_metals = median_abunds,
                      temp_params = median_temps,
                      clouds_params = median_clouds,
                      external_pt_profile = None)
        if data_obj.PHOTinDATA():
            PHOT_data_flux,PHOT_data_err,filt,filt_func,PHOT_filter_midpoint,PHOT_filter_width = data_obj.getPhot()
            photometry,wlen_ck,flux_ck = rebin_to_PHOT(wlen_ck,flux_ck,filt_func,median_temps['log_R'],config['DISTANCE'])
            if saving:
                save_photometry(photometry,data_obj.PHOT_data_err,data_obj.PHOT_filter_midpoint,data_obj.PHOT_filter_width,save_dir=output_result_dir+'photometry')
                save_lines([wlen_ck,flux_ck],save_dir = output_result_dir+'ck_spectrum')
            
    
    plot_data(
            config,
            CC_wlen = CC_wvl_data,
            CC_flux = CC_flux_data,
            model_CC_wlen = wlen_CC,
            model_CC_flux = flux_CC,
            RES_wlen = RES_wvl_data,
            RES_flux = RES_flux_data,
            RES_flux_err = RES_cov_data,
            model_RES_wlen = wlen_RES,
            model_RES_flux = flux_RES,
            PHOT_midpoint = PHOT_filter_midpoint,
            PHOT_width = PHOT_filter_width,
            PHOT_flux = PHOT_data_flux,
            PHOT_flux_err = PHOT_data_err,
            PHOT_filter = filt,
            PHOT_sim_wlen = wlen_ck,
            PHOT_sim_flux = flux_ck,
            model_PHOT_flux = photometry,
            output_file = output_file,
            plot_name='plot_retrieved_spectrum')
    
    if output_results:
        return wlen_CC,flux_CC,wlen_RES,flux_RES,photometry

def plot_retrieved_spectra_FM_dico(
        retrieval,
        samples,
        output_file = '',
        title = 'Retrieved spectrum',
        show_random = None,
        saving = True,
        output_results = False):
    
    data_obj = retrieval.data_obj
    config = retrieval.config
    
    output_result_dir = output_file + 'retrieved_spectrum/'
    if saving:
        if not os.path.exists(output_result_dir):
            try:
                os.mkdir(output_result_dir)
            except FileExistsError:
                print('Error avoided')
        if not os.path.exists(output_result_dir+'photometry/'):
            try:
                os.mkdir(output_result_dir+'photometry/')
            except FileExistsError:
                print('Error avoided')
    
    CC_wvl_data,CC_flux_data,wlen_CC,flux_CC,sgfilter,RES_wvl_data,RES_flux_data,RES_cov_data,wlen_RES,flux_RES = {},{},{},{},{},{},{},{},{},{}
    PHOT_filter_midpoint,PHOT_filter_width,PHOT_data_flux,PHOT_data_err,filt,wlen_ck,flux_ck,photometry = {},{},{},{},{},{},{},{}
    
    if 'log_R' not in config['PARAMS_NAMES'] and 'R' in config['PARAMS_NAMES']:
        print('CHANGING R to logR')
        index_R = config['PARAMS_NAMES'].index('R')
        samples[:,index_R] = np.log10(samples[:,index_R])
        config = config.copy()
        config['PARAMS_NAMES'][index_R] = 'log_R'
    
    nb_positions = len(samples)
    med_par = np.median(samples,axis=0)
    
    median_abunds,median_temps,median_clouds = fix_params(config,med_par)
    
    
    if data_obj.RESinDATA():
        RES_wvl_data,RES_flux_data,RES_cov_data,inverse_cov,flux_err_data = data_obj.getRESSpectrum()
        
    if data_obj.CCinDATA():
        CC_wvl_data,CC_flux_data = data_obj.getCCSpectrum()
        data_N,data_sf2 = data_obj.CC_data_N,data_obj.CC_data_sf2
    
    if retrieval.forwardmodel_ck is not None:
        
        wlen_ck,flux_ck = retrieval.forwardmodel_ck.calc_spectrum(
                      ab_metals = median_abunds,
                      temp_params = median_temps,
                      clouds_params = median_clouds,
                      external_pt_profile = None)
        if data_obj.PHOTinDATA():
            PHOT_data_flux,PHOT_data_err,filt,filt_func,PHOT_filter_midpoint,PHOT_filter_width = data_obj.getPhot()
            photometry,wlen_ck,flux_ck = rebin_to_PHOT(wlen_ck,flux_ck,filt_func,median_temps['log_R'],config['DISTANCE'])
            if saving:
                save_photometry(photometry,data_obj.PHOT_data_err,data_obj.PHOT_filter_midpoint,data_obj.PHOT_filter_width,save_dir=output_result_dir+'photometry')
                save_lines([wlen_ck,flux_ck],save_dir = output_result_dir+'ck_spectrum')
        
        if data_obj.RES_data_with_ck:
            for key in RES_wvl_data.keys():
                if data_obj.RES_data_info[key][0] == 'c-k':
                    wlen_RES[key],flux_RES[key] = rebin_to_RES(wlen_ck,flux_ck,RES_wvl_data[key],median_temps['log_R'],config['DISTANCE'])
    
    if retrieval.forwardmodel_lbl is not None:
        wlen_lbl,flux_lbl={},{}
        for interval_key in retrieval.lbl_itvls.keys():
            wlen_lbl[interval_key],flux_lbl[interval_key] = retrieval.forwardmodel_lbl[interval_key].calc_spectrum(
                      ab_metals = median_abunds,
                      temp_params = median_temps,
                      clouds_params = median_clouds,
                      external_pt_profile = None)
            if data_obj.CCinDATA():
                for key in CC_wvl_data.keys():
                    if retrieval.CC_to_lbl_itvls[key] == interval_key:
                        wlen_CC[key],flux_CC[key],sgfilter[key],wlen_rebin,flux_rebin = rebin_to_CC(wlen_lbl[interval_key],flux_lbl[interval_key],CC_wvl_data[key],
                                                                             config['WIN_LEN'],filter_method = 'only_gaussian',
                                                                             convert = config['CONVERT_SINFONI_UNITS'],log_R=median_temps['log_R'],distance=config['DISTANCE'])
                if saving:
                    save_spectra(wlen_CC,flux_CC,save_dir= output_result_dir + 'CC_spectrum',save_name='')
            
            if data_obj.RESinDATA():
                for key in RES_wvl_data.keys():
                    if retrieval.data_obj.RES_data_info[key][0] == 'lbl':
                        if retrieval.RES_to_lbl_itvls[key] == interval_key:
                            wlen_RES[key],flux_RES[key] = rebin_to_RES(wlen_lbl[interval_key],flux_lbl[interval_key],RES_wvl_data[key],median_temps['log_R'],config['DISTANCE'])
    if data_obj.RESinDATA():
        if saving:
            save_spectra(wlen_RES,flux_RES,save_dir= output_result_dir + 'RES_spectrum',save_name='')
    
    plot_data(
            config,
            CC_wlen = CC_wvl_data,
            CC_flux = CC_flux_data,
            model_CC_wlen = wlen_CC,
            model_CC_flux = flux_CC,
            RES_wlen = RES_wvl_data,
            RES_flux = RES_flux_data,
            RES_flux_err = RES_cov_data,
            model_RES_wlen = wlen_RES,
            model_RES_flux = flux_RES,
            PHOT_midpoint = PHOT_filter_midpoint,
            PHOT_width = PHOT_filter_width,
            PHOT_flux = PHOT_data_flux,
            PHOT_flux_err = PHOT_data_err,
            PHOT_filter = filt,
            PHOT_sim_wlen = wlen_ck,
            PHOT_sim_flux = flux_ck,
            model_PHOT_flux = photometry,
            output_file = output_file,
            plot_name='plot_retrieved_spectrum')
    
    if output_results:
        return wlen_CC,flux_CC,wlen_RES,flux_RES,photometry


def plot_CO_ratio(
                config,
                samples,
                percent_considered = 1.,
                abundances_considered = 'all',
                output_file = '',
                fontsize = 10,
                lw = 0.5,
                figsize=(8,4),
                color = 'g',
                label='C/O$=$',
                include_quantiles = True,
                title='C/O ratio',
                ax = None):
    
    CO_ratio_samples = calc_CO_ratio(samples, 
                                 params_names = config['PARAMS_NAMES'], 
                                 abundances = config['ABUNDANCES'], 
                                 percent_considered = percent_considered,
                                 abundances_considered = abundances_considered,
                                 method = 'VMRs')
    saving = False
    if ax is None:
        saving = True
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
    quantiles = np.quantile(CO_ratio_samples,q=[(1-0.6827)/2,0.5,1-(1-0.6827)/2])
    ax.hist(CO_ratio_samples,bins = 40,color=color,density = True,alpha=0.5,label=label + quantiles_to_string(quantiles))
    ax.axvline(quantiles[1],color=color,lw=lw,ls='--')
    if include_quantiles:
        ax.axvline(quantiles[0],color=color,ls='--',lw=lw)
        ax.axvline(quantiles[2],color=color,ls='--',lw=lw)
    ax.set_xlabel('C/O ratio (MMR)',fontsize=fontsize)
    ax.set_ylabel('Probability distribution',fontsize=fontsize)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.legend(fontsize=fontsize)
    if title is not None:
        ax.set_title(title,fontsize=fontsize)
    if saving:
        fig.savefig(output_file + 'CO_ratio.png',dpi=300)
    else:
        return ax

def plot_FeH_ratio(
                config,
                samples,
                percent_considered = 1.,
                abundances_considered = 'all',
                output_file = '',
                fontsize = 10,
                lw = 0.5,
                figsize=(8,4),
                color = 'g',
                label='[Fe/H]$=$',
                include_quantiles = True,
                title='[Fe/H]',
                ax = None):
    
    FeH_ratio_samples = calc_FeH_ratio_from_samples(
                                samples, 
                                params_names = config['PARAMS_NAMES'], 
                                abundances = config['ABUNDANCES'], 
                                percent_considered = percent_considered,
                                abundances_considered = abundances_considered)
    saving = False
    if ax is None:
        saving = True
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
    quantiles = np.quantile(FeH_ratio_samples,q=[(1-0.6827)/2,0.5,1-(1-0.6827)/2])
    ax.hist(FeH_ratio_samples,bins = 40,color=color,density = True,alpha=0.5,label=label + quantiles_to_string(quantiles))
    ax.axvline(quantiles[1],color=color,lw=lw,ls='--')
    if include_quantiles:
        ax.axvline(quantiles[0],color=color,ls='--',lw=lw)
        ax.axvline(quantiles[2],color=color,ls='--',lw=lw)
    ax.set_xlabel('[Fe/H] (dex)',fontsize=fontsize)
    ax.set_ylabel('Probability distribution',fontsize=fontsize)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.legend(fontsize=fontsize)
    if title is not None:
        ax.set_title(title,fontsize=fontsize)
    if saving:
        fig.savefig(output_file + 'FeH_ratio.png',dpi=300)
    else:
        return ax

def plot_walkers(config,
                 samples,
                 preburn_samples,
                 prob,
                 quantiles = ((1-0.6827)/2,0.5,1-(1-0.6827)/2),
                 percent_considered = 0.8,
                 output_files = '',
                 title = 'Walkers of retrieval'
                 ):
    nb_params=len(config['PARAMS_NAMES'])
    nb_walkers = len(prob)
    nb_iter = int(len(samples)/nb_walkers)
    nb_preburn_iter = int(len(preburn_samples)/nb_walkers)
    samples_reshaped = samples.reshape(nb_walkers,nb_iter,nb_params)
    samples_preburn_reshaped=preburn_samples.reshape(nb_walkers,nb_preburn_iter,nb_params)
    index_consider = int(nb_iter*(1-percent_considered))
    samples_cut = samples_reshaped[:,index_consider:,:]
    
    quantiles_params = np.quantile(samples_cut,q=quantiles,axis=(0,1))
    
    fig,ax = plt.subplots(nb_params,1,figsize=(10,nb_params*7))
    if nb_params == 1:
        ax = [ax]
    for k in range(nb_params):
        for j in range(nb_walkers):
            ax[k].plot(range(len(samples_preburn_reshaped[j,:,k])),samples_preburn_reshaped[j,:,k],'r',linewidth=0.2)
            ax[k].plot(range(len(samples_preburn_reshaped[j,:,k]),len(samples_reshaped[j,:,k])+len(samples_preburn_reshaped[j,:,k])),samples_reshaped[j,:,k],'k',linewidth=0.2,)
        if config['DATA_PARAMS'] is not None:
            ax[k].axhline(config['DATA_PARAMS'][config['PARAMS_NAMES'][k]],color='r',label='True: ' +str(config['DATA_PARAMS'][config['PARAMS_NAMES'][k]]))
        for i in range(len(quantiles)):
            ax[k].axhline(quantiles_params[i][k],xmin=(nb_preburn_iter+(1-percent_considered)*nb_iter)/(nb_preburn_iter+nb_iter), xmax=1 ,color='g')
        median_str = '{median:.2f}'.format(median=quantiles_params[1][k])
        q2_str = '{q2:.2f}'.format(q2=quantiles_params[2][k]-quantiles_params[1][k])
        q1_str = '{q1:.2f}'.format(q1=quantiles_params[1][k]-quantiles_params[0][k])
        ax[k].axvline(nb_preburn_iter,color='r')
        ax[k].axvline(nb_preburn_iter+int((1-percent_considered)*nb_iter),color='g',label= 'Retrieved: '+median_str+'$^{+'+q2_str+'}_{-'+q1_str+'}$')
        if k == 0:
            ax[k].set_title(title,fontsize=15)
        ax[k].set_ylabel(config['PARAMS_NAMES'][k],fontsize=15)
        ax[k].set_xlim((0,nb_iter+nb_preburn_iter))
        ax[k].legend(fontsize=15)
        
    fig.tight_layout()
    #fig.subplots_adjust(hspace=0.5)
    fig.savefig(output_files+'walkers.png',dpi=300)

def plot_retrieved_rotbroad(
        samples,
        dRV_data,
        CC_data,
        wlen,
        flux,
        config,
        output_dir = '',
        percent_considered = 0.8,
        fontsize=11,
        lw=0.5
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
    
    samples_cut = samples[int((1-percent_considered)*nb_positions):]
    print('considered: {p:.2f}'.format(p=len(samples_cut)/len(samples)))
    quantiles = np.quantile(samples_cut,q=[(1-0.6827)/2,0.5,1-(1-0.6827)/2],axis=0)
    
    
    plt.figure(figsize=(8,5))
    CC_data = CC_data/max(CC_data)
    
    plt.plot(dRV_data,CC_data,'k',lw=lw,label='True CCF')
    
    wlen_temp,flux_temp = {},{}
    CC_range_i = {}
    for key_i,key in enumerate(wlen.keys()):
        wlen_temp[key],flux_temp[key]=wlen[key],flux[key]
        print('Progress: {p:.2f} %'.format(p=100*(key_i + 1*(key_i+1))/3/len(wlen.keys())))
        if 'radial_vel' in config['PARAMS_NAMES']:
            wlen_temp[key],flux_temp[key] = doppler_shift(wlen_temp[key],flux_temp[key],quantiles[1][config['PARAMS_NAMES'].index('radial_vel')])
        if 'spin_vel' in config['PARAMS_NAMES']:
            flux_temp[key] = fastRotBroad(wlen_temp[key],flux_temp[key],0,quantiles[1][config['PARAMS_NAMES'].index('spin_vel')])
        dRV,CC_range_i[key] = crosscorrRV(
                               wlen_temp[key],flux_temp[key],
                               wlen[key],flux[key],
                               rvmin=rvmin,rvmax=rvmax,drv=drv,skipedge=skipedge)
    CC = np.array([sum([CC_range_i[key][drv_i] for key in CC_range_i.keys()]) for drv_i in range(len(dRV))])
    CC = CC/max(CC)
    
    #plt.plot(dRV,CC,'k--')
    """
    CCF_new = CC
    if 'radial_vel' in config['PARAMS_NAMES']:
        CCF_f = interp1d(dRV + quantiles['radial_vel'][quant_i],CC,fill_value = 0)
        CCF_new = CCF_f(dRV)
    CC = CCF_new
    """
    median_str = ['{median:.1f}'.format(median=quantiles[1][param_i]) for param_i in range(len(config['PARAMS_NAMES']))]
    q2_str = ['{q2:.1f}'.format(q2=quantiles[2][param_i]-quantiles[1][param_i]) for param_i in range(len(config['PARAMS_NAMES']))]
    q1_str = ['{q1:.1f}'.format(q1=quantiles[1][param_i]-quantiles[0][param_i]) for param_i in range(len(config['PARAMS_NAMES']))]
    plt.plot(dRV,CC,'r',lw=lw,label='Best fit to CCF')
    
    plt.axvline(config['DATA_PARAMS']['radial_vel'],color='k',lw=lw,ls='--',label='True v$_{\mathrm{R}}=$' +str(config['DATA_PARAMS']['radial_vel'])+' kms$^{-1}$\nTrue v$_{\mathrm{S}}=$'+str(config['DATA_PARAMS']['spin_vel'])+' kms$^{-1}$')
    
    plt.axvline(quantiles[1][config['PARAMS_NAMES'].index('radial_vel')],color='r',lw=lw,ls='--',label='Meas. v$_{\mathrm{R}}=$'+median_str[config['PARAMS_NAMES'].index('radial_vel')]+'$^{+'+q2_str[config['PARAMS_NAMES'].index('radial_vel')]+'}_{-'+q1_str[config['PARAMS_NAMES'].index('radial_vel')]+'}$ kms$^{-1}$\nMeas. v$_{\mathrm{S}}=$'+median_str[config['PARAMS_NAMES'].index('spin_vel')]+'$^{+'+q2_str[config['PARAMS_NAMES'].index('spin_vel')]+'}_{-'+q1_str[config['PARAMS_NAMES'].index('spin_vel')]+'}$ kms$^{-1}$')
    
    #plt.axvline(quantiles[0][config['PARAMS_NAMES'].index('radial_vel')],color='g',ls='--')
    #plt.axvline(quantiles[2][config['PARAMS_NAMES'].index('radial_vel')],color='g',ls='--')
    
    plt.legend(fontsize=fontsize)
    plt.xlabel('Radial velocity [kms$^{-1}$]',fontsize=fontsize)
    plt.ylabel('CCF',fontsize=fontsize)
    plt.title('Measured radial and spin velocity',fontsize=fontsize+1)
    plt.tick_params(axis='both',labelsize=fontsize)
    plt.tight_layout()
    plt.savefig(output_dir+'retrieved_rot_broad.png',dpi=300)

def plot_retrieved_temperature_profile(
        config,
        samples,
        output_file,
        nb_stds = 3,
        fontsize = 10,
        lw = 0.5,
        figsize=(8,4),
        color = 'g',
        label='T$_{\mathrm{equ}}=$',
        plot_data = True,
        plot_label=True,
        title='Thermal profile',
        ax = None):
    print('PLOTTING THERMAL PROFILE')
    saving = False
    if ax is None:
        saving = True
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
    
    params_names = config['PARAMS_NAMES']
    data_params = config['DATA_PARAMS']
    
    temp_params_names = config['TEMPS']
    unsearched_temp_params = config['UNSEARCHED_TEMPS']
    
    temp_params = {}
    for param in unsearched_temp_params:
        temp_params[param] = data_params[param]
    
    if 'P0' in params_names:
        temp_params['P0'] = np.median(samples[:,params_names.index('P0')])
    
    pressures = np.logspace(-6,temp_params['P0'],100)
    
    temp_curves = np.zeros((len(samples),len(pressures)))
    for pos_i,position in enumerate(samples):
        
        for param_i,param in enumerate(params_names):
            if param in temp_params_names:
                temp_params[param] = position[param_i]
        
        temp_curves[pos_i] = nc.guillot_global(
                pressures,
                1e1**temp_params['log_kappa_IR'],
                1e1**temp_params['log_gamma'],
                1e1**temp_params['log_gravity'],
                temp_params['t_int'],
                temp_params['t_equ'])
    quantiles = [(1-0.9973)/2,
                 (1-0.9545)/2,
                 (1-0.6827)/2,
                 0.5,
                 1-(1-0.6827)/2,
                 1-(1-0.9545)/2,
                 1-(1-0.9973)/2][3-nb_stds:3+nb_stds+1]
    quantile_curves = np.quantile(temp_curves,q=quantiles,axis = 0)
    
    data_pressures = np.logspace(-6,data_params['P0'],100)
    data_temperatures = nc.guillot_global(data_pressures,
                1e1**data_params['log_kappa_IR'],
                1e1**data_params['log_gamma'],
                1e1**data_params['log_gravity'],
                data_params['t_int'],
                data_params['t_equ'])
    if plot_data:
        ax.plot(data_temperatures,data_pressures,color = 'r',lw=lw*2,label='True $T_{\mathrm{equ}}$: '+'{v}'.format(v=data_params['t_equ']) + ' K')
    # median curve
    """
    if 't_equ' in params_names:
        ax.plot(quantile_curves[nb_stds],pressures,color = color,lw=lw)
    """
    if nb_stds > 0:
        for std_i in range(1,nb_stds+1):
            lower_curve = quantile_curves[nb_stds - std_i]
            higher_curve = quantile_curves[nb_stds + std_i]
            
            #ax.plot(lower_curve,pressures,color = color,lw=lw)
            #ax.plot(higher_curve,pressures,color = color,lw=lw)
            if std_i == 1:
                ax.fill_betweenx(pressures,lower_curve,higher_curve,color = color,alpha=1/(std_i+3),label= label + quantiles_to_string(np.quantile(samples[:,params_names.index('t_equ')],q=[(1-0.6827)/2,0.5,1-(1-0.6827)/2]),decimals = 0) + ' K')
            else:
                ax.fill_betweenx(pressures,lower_curve,higher_curve,color = color,alpha=1/(std_i+3))
    
    ax.set_yscale('log')
    ax.invert_yaxis()
    ax.set_ylim((10**temp_params['P0'],10**-6))
    ax.set_xlabel('Temperature [K]',fontsize=fontsize)
    ax.set_ylabel('Pressure [bar]',fontsize=fontsize)
    if plot_label:
        ax.legend(fontsize=fontsize)
    if title is not None:
        ax.set_title(title,fontsize=fontsize)
    if saving:
        fig.savefig(output_file + 'temperature_plot.png',dpi=300)
    else:
        return ax

def plot_retrieved_spectra_ck(
        forward_model_ck,
        config,
        samples,
        output_file,
        nb_picks = 100,
        fontsize = 10,
        lw = 0.5,
        figsize=(8,4),
        title='Retrieved spectrum',
        color='k',
        alpha=1,
        plot_label=True,
        ax = None):
    
    saving = False
    if ax is None:
        saving = True
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
    
    params_names = config['PARAMS_NAMES']
    data_params = config['DATA_PARAMS']
    
    ab_metals_params = config['ABUNDANCES']
    unsearched_ab_metals = config['UNSEARCHED_ABUNDANCES']
    
    temp_params_names = config['TEMPS']
    unsearched_temp_params = config['UNSEARCHED_TEMPS']
    
    ab_metals,temp_params = calc_retrieved_params(config,samples)
    
    if 'log_R' not in params_names:
        temp_params['log_R'] = data_params['log_R']
    
    wlen, flux = forward_model_ck.calc_spectrum(
                  ab_metals,
                  temp_params,
                  clouds_params = {},
                  external_pt_profile = None,
                  return_profiles = False,
                  contribution = False
                  )
    
    wlen_median,flux_median = convert_units(wlen, flux, log_radius = temp_params['log_R'], distance = config['DISTANCE'])
    
    
    if nb_picks > 0:
        
        random_samples = samples[sample(range(len(samples)),k=nb_picks)]
        flux_curves = np.zeros((nb_picks,len(wlen_median)))
        for pos_i,position in enumerate(random_samples):
            
            for param_i,param in enumerate(params_names):
                if param in temp_params_names:
                    temp_params[param] = position[param_i]
                if param in ab_metals_params:
                    ab_metals[param] = position[param_i]
            
            
            wlen,flux = forward_model_ck.calc_spectrum(
                          ab_metals,
                          temp_params,
                          clouds_params = {},
                          external_pt_profile = None,
                          return_profiles = False,
                          contribution = False
                          )
            
            wlen,flux = convert_units(wlen, flux, log_radius = temp_params['log_R'], distance = config['DISTANCE'])
            flux_curves[pos_i] = flux
    
        for pos_i in range(nb_picks):
            ax.plot(wlen_median,flux_curves[pos_i],color=color,lw=0.1,alpha=0.2)
    
    ax.plot(wlen_median,flux_median,color=color,lw=lw,label='Best-fit',alpha=alpha)
    
    #ax.set_xlabel('Wavelength [$\mu$m]',fontsize=fontsize)
    ax.set_ylabel('Flux [Wm$^{-2}\mu$m$^{-1}$]',fontsize=fontsize)
    if plot_label:
        ax.legend(fontsize=fontsize)
    if title is not None:
        ax.set_title(title,fontsize=fontsize)
    if saving:
        fig.savefig(output_file + 'retrieved_spectra_ck.png',dpi=300)
    else:
        return ax

def plot_retrieved_abunds(
        config,
        samples,
        pressure_distr = None,
        output_dir = '',
        nb_stds = 1,
        fontsize = 10,
        lw = 0.5,
        figsize=(8,4),
        xlim = [-10,0],
        title='Molecular abundance profiles',
        add_xlabel = True,
        add_ylabel = True,
        add_legend = False,
        errorbar_color = None,
        plot_marker = False,
        ax = None):
    print('PLOTTING ABUNDANCE PROFILES')
    saving = False
    if ax is None:
        saving = True
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
    
    params_names = config['PARAMS_NAMES']
    data_params = config['DATA_PARAMS']
    
    ab_metals_params = config['ABUNDANCES']
    unsearched_ab_metals = config['UNSEARCHED_ABUNDANCES']
    
    temp_params_names = config['TEMPS']
    unsearched_temp_params = config['UNSEARCHED_TEMPS']
    
    
    pressures = np.logspace(-6, data_params['P0'], 100)
    temperatures = nc.guillot_global(
                pressures,
                1e1**data_params['log_kappa_IR'],
                1e1**data_params['log_gamma'],
                1e1**data_params['log_gravity'],
                data_params['t_int'],
                data_params['t_equ'])
    
    quantiles = [(1-0.9973)/2,
                 (1-0.9545)/2,
                 (1-0.6827)/2,
                 0.5,
                 1-(1-0.6827)/2,
                 1-(1-0.9545)/2,
                 1-(1-0.9973)/2][3-nb_stds:3+nb_stds+1]
    
    data_case = 'free'
    retrieval_case = 'free'
    if 'C/O' in data_params.keys():
        data_case = 'chem_equ'
    if 'C/O' in params_names:
        retrieval_case = 'chem_equ'
    data_curves = {}
    if data_case == 'free':
        for param in data_params.keys():
            if param in ab_metals_params + unsearched_ab_metals:
                data_curves[param] = 10**data_params[param]*np.ones(len(pressures))
    else:
        CO_profile = data_params['C/O']*np.ones_like(pressures)
        FeH_profile = data_params['FeHs']*np.ones_like(pressures)
        mass_fractions = poor_mans_nonequ_chem.interpol_abundances(
                CO_profile,
                FeH_profile,
                temperatures,
                pressures)
        abundances = filter_relevant_mass_fractions(mass_fractions,mode='lbl')
        for param in abundances.keys():
            data_curves[param] = abundances[param]
        
        
    quantile_curves_retr = {}
    quantiles_param = {}
    if retrieval_case == 'free':
        for param in ab_metals_params:
            quantile_curves_retr[param] = np.zeros((nb_stds*2+1,len(pressures)))
            quantiles_param[param] = np.quantile(samples[:,params_names.index(param)],q=quantiles,axis=0)
            for q_i in range(len(quantiles)):
                quantile_curves_retr[param][q_i] = 10**quantiles_param[param][q_i]*np.ones(len(pressures))
    else:
        sample_curves = {}
        for param in poor_mans_abunds_lbl():
            sample_curves[param] = np.zeros((len(samples),len(pressures)))
        for pos_i,position in samples:
            CO_profile = position[params_names.index('C/O')]*np.ones_like(pressures)
            FeH_profile = position[params_names.index('FeHs')]*np.ones_like(pressures)
            mass_fractions = poor_mans_nonequ_chem.interpol_abundances(
                CO_profile,
                FeH_profile,
                temperatures,
                pressures)
            abundances = filter_relevant_mass_fractions(mass_fractions,mode='lbl')
            for param in poor_mans_abunds_lbl():
                sample_curves[param][pos_i] = abundances[param]
        for param in poor_mans_abunds_lbl():
            quantile_curves_retr[param] = np.zeros((nb_stds*2+1,len(pressures)))
            quantiles_param = np.quantile(sample_curves[param],q=quantiles,axis=0)
            for q_i in range(len(quantiles)):
                quantile_curves_retr[param][q_i] = quantiles_param[q_i]
    
    colors = {}
    index = 0
    common_params = []
    for param_data in data_curves.keys():
        if param_data in ab_metals_params:
            index += 1
            common_params.append(param_data)
    cmap = color_palette(palette='colorblind',n_colors = len(common_params),as_cmap=True)
    for param in common_params:
        colors[param] = cmap[common_params.index(param)]
            
    
    for param in data_curves.keys():
        if param in common_params:
            ax.plot(data_curves[param],pressures,color=colors[param],lw=lw,label=nice_name(param),zorder=0)
        """
        else:
            ax.plot(data_curves[param],pressures,lw=lw*2,alpha=0.3)
            ax.text(data_curves[param][-1],pressures[-1],nice_name(param))
        """
    
    if pressure_distr is None:
        for param in ab_metals_params:
            if param in common_params:
                ax.plot(quantile_curves_retr[param][nb_stds],pressures,color=colors[param],lw=lw,ls='--')
            else:
                ax.plot(quantile_curves_retr[param][nb_stds],pressures,lw=lw,ls='--')
            
            if nb_stds > 0:
                for std_i in range(1,nb_stds+1):
                    lower_curve = quantile_curves_retr[param][nb_stds - std_i]
                    higher_curve = quantile_curves_retr[param][nb_stds + std_i]
                    
                    if param in common_params:
                        #ax.plot(lower_curve,pressures,lw=lw,color=colors[param])
                        #ax.plot(higher_curve,pressures,lw=lw,color=colors[param])
                        ax.fill_betweenx(pressures,lower_curve,higher_curve,color=colors[param],alpha=0.5/(std_i+1))
                    else:
                        #ax.plot(lower_curve,pressures,lw=lw)
                        #ax.plot(higher_curve,pressures,lw=lw)
                        ax.fill_betweenx(pressures,lower_curve,higher_curve,alpha=1/(std_i+1))
    model_line = None
    if pressure_distr is not None:
        model_line={}
        print('Plotting errorbars')
        for param in ab_metals_params:
            
            #pressure_lower = calc_quantile(pressure_distr[param],q=0.16)
            #pressure_median = calc_quantile(pressure_distr[param],q=0.5)
            #pressure_higher = calc_quantile(pressure_distr[param],q=0.84)
            
            index_lower,max_index,index_higher = calc_FWHM(pressure_distr[param])
            p_low,p_max,p_high = pressures[[index_lower,max_index,index_higher]]
            
            #yerr = np.array([[abs(pressure_median-pressure_lower)],[abs(pressure_median-pressure_higher)]])
            yerr = np.array([[abs(p_max-p_low)],[abs(p_max-p_high)]])
            xerr = np.array([[abs(10**quantiles_param[param][nb_stds]-10**quantiles_param[param][nb_stds-1])],[abs(10**quantiles_param[param][nb_stds]-10**quantiles_param[param][nb_stds+1])]])
            
            mol_sample = samples[:,params_names.index(param)]
            p_bounds = [p_low,p_max,p_high]
            p0 = quantiles_param[param][nb_stds]
            if param == 'H2S_main_iso' and p0 > -5:
                p0 = -1.8
            model = Posterior_Classification_Errorbars(
                mol_sample,
                param,
                p0_SSG = [8,-p0,0.007,0.5,0.5],
                output_dir = output_dir,
                plotting = False
                )
            if param not in common_params:
                ax.errorbar(10**quantiles_param[param][nb_stds],p_max,xerr = xerr,yerr = yerr,lw=lw,capsize=2,capthick=lw)
            else:
                model_line[param] = Errorbars_plot(
                    ax,model,pressure_distr[param],p_bounds,marker_color = colors[param],lw = lw,un_c_len=2,
                    errorbar_color = errorbar_color,
                    plot_marker = plot_marker)
            
        
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.invert_yaxis()
    if add_xlabel:
        ax.set_xlabel('Mass fraction',fontsize=fontsize)
    if add_ylabel:
        ax.set_ylabel('Pressure [bar]',fontsize=fontsize)
    ax.set_xlim((10**xlim[0],10**xlim[1]))
    ax.set_ylim((10**data_params['P0'],10**-6))
    if add_legend:
        ax.legend(fontsize=fontsize)
    if title is not None:
        ax.set_title(title,fontsize=fontsize)
    if saving:
        fig.savefig(output_dir + 'abundances_plot.png',dpi=300)
        return ax,colors,model_line
    else:
        return ax,colors,model_line
    
def plot_retrieved_fluxes(
        config,
        samples,
        data_obj,
        forward_model_ck = None,
        nb_picks = 100,
        output_dir = '',
        fontsize = 10,
        lw = 0.8,
        figsize=(8,12),
        title='Retrieved spectrum',
        ax = None
        ):
    
    saving = False
    if ax is None:
        saving = True
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
    else:
        fig = plt.gcf()
    
    plotting_phot = data_obj.PHOTinDATA()
    plotting_RES = data_obj.RESinDATA()
    plotting_CC = data_obj.CCinDATA()
    
    nb_rows = plotting_phot + 3*(plotting_RES or plotting_phot) + 2*plotting_CC
    
    if plotting_phot or plotting_RES:
        wlen_borders_ck = data_obj.wlen_range_ck()
    
    if plotting_phot:
        ax0 = plt.subplot2grid((nb_rows,1),(0,0),fig=fig)
        
        # get data photometry
        PHOT_data_flux,PHOT_data_err,PHOT_data_filter,PHOT_filter_function,PHOT_filter_midpoint,PHOT_filter_width = data_obj.getPhot()
        
        instruments_phot = list(PHOT_data_flux.keys())
        
        cmap = plt.get_cmap('viridis')
        filter_pos = filter_position(PHOT_filter_midpoint)
        
        for instr in instruments_phot:
            ax0.plot(PHOT_data_filter[instr][0],PHOT_data_filter[instr][1],color=cmap(filter_pos[instr]/len(instruments_phot)))
        
        ax0.set_ylabel('Filter transmission',fontsize=fontsize)
        #ax0.set_xlabel('Wavelength [$\mu$m]',fontsize=fontsize)
        if title is not None:
            ax0.set_title(title,fontsize=fontsize)
        ax0.set_xlim((wlen_borders_ck[0],wlen_borders_ck[1]))
        
    if plotting_phot or plotting_RES:
        ax0 = plt.subplot2grid((nb_rows,1),(plotting_phot,0),rowspan=2,fig=fig)
        
        if plotting_RES:
            # get retrieved RES spectrum
            retr_RES_wlen,retr_RES_flux = open_spectrum(output_dir + 'retrieved_spectrum/RES_spectrum/data.txt')
            
            # get data RES
            RES_data_wlen,RES_data_flux,RES_data_err,RES_inv_cov,RES_data_flux_err = data_obj.getRESSpectrum()
            instruments_RES = list(RES_data_flux.keys())
            
            for instr in instruments_RES:
                label=''
                if instr == instruments_RES[-1]:
                    label='Synthetic GRAVITY spectrum'
                ax0.errorbar(RES_data_wlen[instr],RES_data_flux[instr],yerr = RES_data_flux_err[instr],color='b',lw=0.3,label=label)
            #ax0.plot(retr_RES_wlen,retr_RES_flux,'b',lw=lw)
            
        if plotting_phot:
            # get retrieved photometry
            retr_PHOT_data_flux = {}
            for instr in PHOT_data_flux.keys():
                retr_PHOT_data_flux[instr],retr_PHOT_data_err,retr_PHOT_filter_midpoint,retr_PHOT_filter_width = open_photometry(output_dir + 'retrieved_spectrum/photometry'+'/'+instr + '.txt')
                
            for instr in instruments_phot:
                label=''
                if instr == instruments_phot[-1]:
                    label='Synthetic photometry'
                ax0.errorbar(PHOT_filter_midpoint[instr],PHOT_data_flux[instr],xerr = PHOT_filter_width[instr]/2, yerr = PHOT_data_err[instr],color=cmap(filter_pos[instr]/len(instruments_phot)),label=label)
                
                if instr == instruments_phot[-1]:
                    label='Retrieved photometry'
                ax0.plot(PHOT_filter_midpoint[instr],retr_PHOT_data_flux[instr],'ro',ms=2,label=label)
        
        if forward_model_ck is not None:
            ax0 = plot_retrieved_spectra_ck(
                forward_model_ck,
                config,
                samples,
                output_file = output_dir,
                nb_picks = nb_picks,
                fontsize = fontsize,
                lw = lw,
                figsize=figsize,
                title=None,
                ax = ax0)
            
        ax0.set_xlim((wlen_borders_ck[0],wlen_borders_ck[1]))
        ax0.set_ylabel('Flux [Wm$^{-2}\mu$m$^{-1}$]',fontsize=fontsize)
        ax0.legend(fontsize=fontsize)
        if not plotting_phot:
            if title is not None:
                ax0.set_title(title,fontsize=fontsize)
        
        
        ax0 = plt.subplot2grid((nb_rows,1),(plotting_phot+2,0),rowspan=1,fig=fig)
        
        if plotting_RES:
            for instr in instruments_RES:
                ax0.plot(RES_data_wlen[instr],(RES_data_flux[instr]-retr_RES_flux)/RES_data_flux_err[instr],'bo',ms=2)
            
        if plotting_phot:
            for instr in instruments_phot:
                ax0.plot(PHOT_filter_midpoint[instr],(PHOT_data_flux[instr] - retr_PHOT_data_flux[instr])/PHOT_data_err[instr],'ro',ms=2)
        ax0.axhline(0,color='k',ls='--',lw=0.3)
        ax0.set_xlim((wlen_borders_ck[0],wlen_borders_ck[1]))
        ax0.set_ylim((-3,3))
        ax0.set_ylabel('Residuals ($\sigma$)')
        ax0.set_xlabel('Wavelength [$\mu$m]')
        
    if plotting_CC:
        ax0 = plt.subplot2grid((nb_rows,1),(plotting_phot+3*(plotting_phot or plotting_RES),0),rowspan=2,fig=fig)
        
        # get retrieved CC spectrum
        retr_CC_wlen,retr_CC_flux = {},{}
        data = open_spectra(output_dir + 'retrieved_spectrum/CC_spectrum')
        for instr in data.keys():
            retr_CC_wlen[instr],retr_CC_flux[instr] = data[instr]
        
        # get data CC spectrum
        CC_data_wlen,CC_data_flux = data_obj.getCCSpectrum()
        
        instruments_CC = list(CC_data_wlen.keys())
        
        for instr in instruments_CC:
            ax0.plot(CC_data_wlen[instr],CC_data_flux[instr],'k',lw=lw,label='Synthetic SINFONI spectrum')
            ax0.plot(retr_CC_wlen[instr],retr_CC_flux[instr],'r',lw=lw,label='Retrieved spectrum')
        
        wlen_borders_lbl,stepsize_lbl = data_obj.wlen_details_lbl()
        ax0.set_xlim((CC_data_wlen[instr][0],CC_data_wlen[instr][-1]))
        ax0.set_ylabel('Flux [Wm$^{-2}\mu$m$^{-1}$]',fontsize=fontsize)
        ax0.set_xlabel('Wavelength [$\mu$m]')
        ax0.legend(fontsize=fontsize)
    
    fig.tight_layout()
    
    if saving:
        fig.savefig(output_dir + 'retrieved_fluxes.png',dpi=600)
    else:
        return ax
    
def plot_retrieved_temp_abunds_em_contr(
        config,
        samples,
        forward_model_lbl,
        data_obj,
        which_em = 'molecules', # or 'molecules' refering to whether we take em contr fct from retrieved spectrum or from each molecule
        which_abund = 'retr_abund', # 'high_abund' or 'retr_abund' refering to the amount of the molecules to include when calculating em contr fct and flux
        which_included = 'included', # 'excluded' or 'included' refering to whether we only include one molecule at a time, or if we only exclude one molecule at a time but include all others
        plot_temp_profile = True,
        plot_em_contr = True,
        output_dir = '',
        title = 'Molecular abundance profile',
        fontsize = 12,
        lw = 0.8,
        figsize=(12,8),
        ax = None
        ):
    saving = False
    if ax is None:
        saving = True
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
    else:
        fig = plt.gcf()
    
    if which_em == 'retrieved':
        contribution = True
    else:
        contribution = False
    
    wlen_lbl_ref,contr_em_fct,CC_flux_diff_interped,pressure_distr = forward_model_lbl.calc_em_contr_pressure_distr(
        config,
        samples,
        data_obj,
        contribution = contribution,
        which_em = which_em, # or 'molecules' refering to whether we take em contr fct from retrieved spectrum or from each molecule
        which_abund = which_abund, # 'high_abund' or 'retr_abund' refering to the amount of the molecules to include when calculating em contr fct and flux
        which_included = which_included, # 'excluded' or 'included' refering to whether we only include one molecule at a time, or if we only exclude one molecule at a time but include all others
        output_dir = output_dir,
        plot_distr = True)
    
    plot_press_distr = False
    if plot_em_contr and (which_em == 'retrieved' or len(pressure_distr.keys())==1):
        plot_press_distr = True
    
    fig = plt.figure(figsize=figsize)
    if plot_temp_profile:
        ax0 = plt.subplot2grid((6,2+plot_em_contr + plot_press_distr*(1+3*(2+plot_em_contr))),(0,0),rowspan=4,colspan=1+plot_press_distr*2)
        
        ax0 = plot_retrieved_temperature_profile(
            config,
            samples,
            output_dir,
            nb_stds = 2,
            fontsize = fontsize,
            lw = lw,
            figsize=(8,4),
            color = 'g',
            title='Thermal profile',
            ax = ax0)
    
    
    ax0 = plt.subplot2grid((6,1 + plot_temp_profile + plot_em_contr + plot_press_distr*(1+3*(1 + plot_temp_profile + plot_em_contr))),(0,plot_temp_profile*plot_press_distr*3),rowspan=4,colspan=1+plot_press_distr*2)
    mol_lim = [-8,0]
    
    
    ax0,colors,model_line = plot_retrieved_abunds(
        config,
        samples,
        pressure_distr = pressure_distr,
        output_dir = output_dir,
        nb_stds = 1,
        fontsize = fontsize,
        lw = lw,
        figsize=figsize,
        add_xlabel = False,
        xlim = mol_lim,
        title=title,
        ax = ax0)
    ax0.tick_params(axis='x',which='both',bottom=False,labelbottom=False)
    ax0.set_ylim([1e2,1e-4])
    
    ax1 = plt.subplot2grid((6,1 + plot_temp_profile + plot_em_contr + plot_press_distr*(1+3*(1 + plot_temp_profile + plot_em_contr))),(4,plot_temp_profile*plot_press_distr*3),rowspan=2,colspan=1+plot_press_distr*2)
    
    for param in config['ABUNDANCES']:
        h,bins,patches = ax1.hist(samples[:,config['PARAMS_NAMES'].index(param)],bins=40,density = True,color=colors[param],alpha=0.4)
        
        if model_line is not None:
            ax1.plot(np.linspace(-10,0,len(model_line[param])),model_line[param],color=colors[param])
    #ax1.set_ylim(bottom = 10**-4)
    #ax1.set_yscale('log')
    ax1.set_xlim((mol_lim[0],mol_lim[1]))
    ax1.set_xlabel('Mass fraction (dex)',fontsize=fontsize)
    ax1.tick_params(axis='y',which='both',left=False,labelleft=False)
    ax1.set_ylabel('Posterior',fontsize=fontsize-2)
    
    
    if plot_press_distr:
        ax2 = plt.subplot2grid((6,1 + plot_temp_profile + plot_em_contr + plot_press_distr*(1+3*(1 + plot_temp_profile + plot_em_contr))),(0,1 + plot_press_distr*2+plot_temp_profile*plot_press_distr*3),rowspan=4,colspan=1)
        pressures = np.logspace(-6,config['DATA_PARAMS']['P0'],100)
        for key in pressure_distr.keys():
            ax2.plot(pressure_distr[key],pressures,color=colors[key])
        ax2.set_yscale('log')
        ax2.set_ylim([1e2,1e-4])
        ax2.tick_params(axis='both',which='both',left=False,labelleft=False,bottom=False,labelbottom=False)
        ax2.set_title('Range',fontsize=fontsize)
    
    if plot_em_contr:
        ax2 = plt.subplot2grid((6,1 + plot_temp_profile + plot_em_contr + plot_press_distr*(1+3*(1 + plot_temp_profile + plot_em_contr))),(0,plot_press_distr + 1 + plot_press_distr*2+plot_temp_profile*plot_press_distr*3),rowspan=4,colspan=1+plot_press_distr*2)
        pressures = np.logspace(-6,config['DATA_PARAMS']['P0'],100)
        if which_em == 'retrieved' or len(pressure_distr.keys())==1:
            X,Y = np.meshgrid(wlen_lbl_ref[::100], pressures)
            ax2.contourf(X, Y,contr_em_fct[:,::100],cmap=plt.cm.bone_r)
            ax2.set_xlim([np.min(wlen_lbl_ref),np.max(wlen_lbl_ref)])
            ax2.set_title('Contribution emission function',fontsize=fontsize)
            #ax2.set_xlabel('Wavelength [$\mu$m]')
            #ax2.set_ylabel('Pressures [bar]')
        else:
            # 'molecules'
            for key in pressure_distr.keys():
                ax2.plot(pressure_distr[key],pressures,color=colors[key])
            ax2.set_title('Pressure range',fontsize=fontsize)
        ax2.set_yscale('log')
        ax2.set_ylim([1e2,1e-4])
        ax2.tick_params(axis='both',which='both',left=False,labelleft=False,right=True,labelright=True,bottom=False,labelbottom=False)
        
        
        ax3 = plt.subplot2grid((6,1 + plot_temp_profile + plot_em_contr + plot_press_distr*(1+3*(1 + plot_temp_profile + plot_em_contr))),(4,plot_press_distr + 1+ plot_press_distr*2+plot_temp_profile*plot_press_distr*3),rowspan=2,colspan=1+plot_press_distr*2)
        
        for param in CC_flux_diff_interped.keys():
            ax3.plot(wlen_lbl_ref,CC_flux_diff_interped[param],color=colors[param],lw=0.3)
        ax3.set_xlabel('Wavelength [$\mu$m]',fontsize=fontsize)
        ax3.set_ylabel('Mol. tmpl.',fontsize=fontsize-2)
        ax3.tick_params(axis='y',which='both',left=False,labelleft=False)
        ax3.yaxis.set_label_position("right")
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1,wspace=0.15)
    if saving:
        fig.savefig(output_dir + 'temps_abundances_contr_em_'+which_em[:4]+'_ABUND' + which_abund[:4] + '_' + which_included[:4] + '_plot.png',dpi=600,bbox_inches = 'tight',pad_inches = 0)
        fig.savefig(output_dir + 'temps_abundances_contr_em_'+which_em[:4]+'_ABUND' + which_abund[:4] + '_' + which_included[:4] + '_plot.pdf',dpi=600,bbox_inches = 'tight',pad_inches = 0)
    else:
        return ax

