# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 09:40:40 2021

@author: jeanh
"""
import matplotlib.pyplot as plt
import numpy as np
# prior likelihood functions

class Prior:
    def __init__(self,
                 RANGE,
                 log_priors,
                 log_cube_priors
                 ):
        self.RANGE = RANGE
        self.log_priors = log_priors
        self.log_cube_priors = log_cube_priors
    
    def getParams(self):
        return self.RANGE.keys()
    
    def getRANGE(self):
        return self.RANGE
    def getLogPriors(self):
        return self.log_priors
    def getLogCubePriors(self):
        return self.log_cube_priors
    
    def plot(self,config,output_dir=''):
        nb_params = len(config['PARAMS_NAMES'])
        fig,ax = plt.subplots(nrows = 1,ncols=nb_params,figsize=(nb_params*3,3))
        for col_i,name in enumerate(config['PARAMS_NAMES']):
            if name in config['ABUNDANCES']:
                positions = np.linspace(self.RANGE['abundances'][0],self.RANGE['abundances'][1],1000)
            else:
                positions = np.linspace(self.RANGE[name][0],self.RANGE[name][1],1000)
            ax[col_i].plot(positions,[self.log_priors[name](x) for x in positions])
            if name in config['DATA_PARAMS'].keys():
                ax[col_i].axvline(config['DATA_PARAMS'][name],color='r',label='True: {numb:.2f}'.format(numb=config['DATA_PARAMS'][name]))
            ax[col_i].set_title(name)
        fig.savefig(output_dir + 'priors.png',dpi=300)
        