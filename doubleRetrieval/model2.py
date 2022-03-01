# -*- coding: utf-8 -*-
"""
Created on Fri May 28 10:15:36 2021

@author: jeanh
"""

import numpy as np
import sys
#sys.path.append("/scratch/software/petitRADTRANS/")
sys.path.append("/home/ipa/quanz/shared/")
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc

sys.path.append("/home/ipa/quanz/shared/petitRADTRANS")
from poor_mans_nonequ_chem import *

from doubleRetrieval.util import filter_relevant_mass_fractions,get_MMWs
import sys

def calc_MMW(abundances):
    # Molecular Mean Weight
    """
    MMWs = {}
    MMWs['H2'] = 2.
    MMWs['H2_main_iso'] = 2.
    MMWs['He'] = 4.
    MMWs['H2O'] = 18.
    MMWs['H2O_main_iso'] = 18.
    MMWs['CH4'] = 16.
    MMWs['CH4_main_iso'] = 16.
    MMWs['CO2'] = 44.
    MMWs['CO2_main_iso'] = 44.
    MMWs['CO'] = 28.
    MMWs['CO_all_iso'] = 28.
    MMWs['CO_main_iso'] = 28.
    MMWs['Na'] = 23.
    MMWs['K'] = 39.
    MMWs['NH3'] = 17.
    MMWs['NH3_main_iso'] = 17.
    MMWs['HCN'] = 27.
    MMWs['HCN_main_iso'] = 27.
    MMWs['C2H2,acetylene'] = 26.
    MMWs['PH3'] = 34.
    MMWs['H2S'] = 34.
    MMWs['H2S_main_iso'] = 34.
    MMWs['VO'] = 67.
    MMWs['TiO'] = 64.
    MMWs['TiO_all_iso'] = 64.
    MMWs['SiO_main_iso'] = 44.
    MMWs['O3_main_iso'] = 48.
    MMWs['PH3_main_iso'] = 34.
    MMWs['NH3_HITRAN'] = 17.
    MMWs['FeH_main_iso'] = 56.8
    MMWs['FeH_Chubb'] = 56.8
    MMWs['FeH'] = 56.8
    """
    MMW = 0.
    for key in abundances.keys():
        MMW += abundances[key]/get_MMWs(key)
    return 1./MMW

def calc_flux_from_model(
        rt_object,
        temp_params,
        chem_model,
        chem_params,
        clouds_params,
        mode='lbl',
        contribution=False):
    # thermal structure
    pressures,temperatures = calc_thermal_structure(temp_params)


    if chem_model == 'chem_equ':
        abundances = calc_chem_equ_abundances(chem_params, pressures,temperatures, mode=mode)
    elif chem_model == 'free':
        abundances = calc_vert_const_abundances(chem_params,pressures,temperatures,mode=mode)
    elif chem_model == 'fabian':
        abundances = calc_fabian_model(chem_params, pressures,temperatures, mode=mode)


    wlen, flux, abundances = rt_obj_calc_flux(rt_object,
                                              temperatures,
                                              abundances,
                                              1e1 ** temp_params['log_gravity'],
                                              clouds_params,
                                              contribution)

    return wlen, flux, abundances


def retrieval_model_initial(
        rt_object,
        pressures,
        temperatures,
        temp_params,
        ab_metals,
        clouds_params=None,
        mode='lbl',
        contribution = False
        ):
    
    # Forward model
    # Outputs the spectrum given by the chosen parameters

    abundances = calc_vert_const_abundances(ab_metals, pressures, temperatures, mode=mode)

    wlen,flux,abundances = rt_obj_calc_flux(rt_object,
                             temperatures,
                             abundances,
                             1e1**temp_params['log_gravity'],
                             clouds_params,
                             contribution)
    
    return wlen,flux,abundances

def retrieval_model_chem_disequ(
        rt_object,
        temp_params,
        ab_metals,
        clouds_params=None,
        mode='lbl',
        contribution = False,
        only_include = 'all',
        set_mol_abund = None
        ):
    
    
    pressures,temperatures,abundances = calc_chem_equ_abundances(temp_params,ab_metals,mode=mode)
    
    if only_include != 'all':
        for mol in only_include:
            assert(mol in abundances.keys() or mol == 'H2_main_iso')
        new_abundances = {key:abundances[key] for key in only_include if key!= 'H2_main_iso'}
        new_abundances['H2_main_iso'] = abundances['H2']
        new_abundances['H2'] = abundances['H2']
        new_abundances['He'] = abundances['He']
        abundances = {}
        abundances = new_abundances
    
    if mode == 'lbl':
        abundances['H2_main_iso'] = abundances['H2']
    
    if set_mol_abund is not None:
        for key in set_mol_abund.keys():
            print('Removing ',key)
            abundances[key] = set_mol_abund[key]
    
    
    wlen,flux,abundances = rt_obj_calc_flux(rt_object,
                             temperatures,
                             abundances,
                             1e1**temp_params['log_gravity'],
                             clouds_params,
                             contribution)
    
    return wlen,flux,abundances

def rt_obj_calc_flux(rt_object,
                     temperatures,
                     abundances,
                     gravity,
                     clouds_params,
                     contribution):
    
    MMW = calc_MMW(abundances)
    
    Pcloud = None
    if 'log_Pcloud' in clouds_params.keys():
        print('CAREFUL: YOU ARE USING CLOUDS')
        Pcloud = 10**clouds_params['log_Pcloud']
    
    kzz,fsed,sigma_lnorm = None,None,None
    if 'kzz' in clouds_params.keys() and 'fsed' in clouds_params.keys() and 'sigma_lnorm' in clouds_params.keys() and 'cloud_abunds' in clouds_params.keys():
        print('CAREFUL: YOU ARE USING CLOUDS')
        kzz = 10**clouds_params['kzz']*np.ones_like(temperatures)
        fsed = clouds_params['fsed']
        sigma_lnorm = clouds_params['sigma_lnorm']
        
        for cloud_abund in clouds_params['cloud_abunds'].keys():
            abundances[cloud_abund] = 10**clouds_params['cloud_abunds'][cloud_abund] * np.ones_like(temperatures)
    
    
    rt_object.calc_flux(temperatures,
                        abundances,
                        gravity,
                        MMW,
                        Pcloud = Pcloud,
                        contribution = contribution,
                        sigma_lnorm = sigma_lnorm,
                        fsed = fsed,
                        Kzz = kzz
                        )
    
    return nc.c/rt_object.freq, rt_object.flux, abundances

def calc_thermal_structure(temp_params):
    # calculate thermal structure
    pressures = np.logspace(-6, temp_params['P0'], 100)
    temperatures = nc.guillot_global(
        pressures,
        1e1 ** temp_params['log_kappa_IR'],
        1e1 ** temp_params['log_gamma'],
        1e1 ** temp_params['log_gravity'],
        temp_params['t_int'],
        temp_params['t_equ'])
    return pressures,temperatures

def calc_chem_equ_abundances(ab_metals, pressures,temperatures,mode='lbl'):

    COs = ab_metals['C/O']*np.ones_like(pressures)
    FeHs = ab_metals['FeHs']*np.ones_like(pressures)
    
    mass_fractions = poor_mans_nonequ_chem.interpol_abundances(
        COs,
        FeHs,
        temperatures,
        pressures)
    
    abundances = filter_relevant_mass_fractions(mass_fractions,mode)
    
    return abundances

def calc_vert_const_abundances(ab_metals, pressures,temperatures,mode='lbl'):

    # calculate profiles for molecular abundances
    abundances = {}
    metal_sum = 0
    for name in ab_metals.keys():
        abundances[name] = np.ones_like(pressures) * 1e1 ** ab_metals[name]
        metal_sum += 1e1 ** ab_metals[name]

    abH2He = 1. - metal_sum

    if mode == 'lbl':
        abundances['H2_main_iso'] = abH2He * 0.75 * np.ones_like(temperatures)
        abundances['H2'] = abH2He * 0.75 * np.ones_like(temperatures)
    else:
        abundances['H2'] = abH2He * 0.75 * np.ones_like(temperatures)

    abundances['He'] = abH2He * 0.25 * np.ones_like(temperatures)

    return abundances

def calc_fabian_model(chem_params, pressures,temperatures,mode='lbl'):
    abundances = {}
    return abundances