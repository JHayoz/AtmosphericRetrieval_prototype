# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 11:21:15 2021

@author: jeanh
"""

import numpy as np
import scipy.optimize as sco
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
import matplotlib.text as mtext

# Definitions of the posterior models
def Model_Flat(x,c):
    return np.ones_like(x)*c

def Model_SoftStep(x,a,b,c):
    return c / (1+np.exp(a*x+b))

def Inv_Model_SoftStep(y,a,b,c):
    return (np.log(c/y-1)-b)/a

def Model_upper_SoftStep(x,a,b,c):
    return c / (1+np.exp(-a*x+b))

def Inv_Model_upper_SoftStep(y,a,b,c):
    return (np.log(c/y-1)-b)/a
    
def Model_Gauss(x,h,m,s):
    return h/(np.sqrt(2*np.pi)*s)*np.exp(-1/2*((x-m)/s)**2)

def Model_SoftStepG(x,a,b,c,s,e):
    return (c+(e/(np.sqrt(2*np.pi)*s)*np.exp(-(x+b)**2/s**2)))/ (1+np.exp(a*(x+b)))

# Definition of the likelyhood for comparison of different posterior models
def log_likelihood(theta,x,y,Model):
    model = Model(x,*theta)
    if np.sum(model) == 0:
        return -10**100
    return -0.5*np.sum((y-model)**2/0.01**2)

class Handles(HandlerBase):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(Handles, self).__init__()
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        if orig_handle == 'CloudTopPressure':
            l1 = plt.Line2D([0.5*width], [0.5*height], marker = 's', ms=2.5,color='xkcd:magenta')
            l2 = plt.Line2D([x0+0.5*width-0.4*height,x0+0.5*width+0.4*height], [0.5*height,0.5*height], linestyle='-', color='xkcd:magenta',lw=1)
            l3 = plt.Line2D([0.5*width,0.5*width], [0.1*height,0.9*height], linestyle='-', color='xkcd:magenta',lw=1)
            return [l1, l2 ,l3]
        if orig_handle == 'SurfacePressure':
            l1 = plt.Line2D([0.5*width], [0.5*height], marker = 'o', ms=2.5,color='xkcd:red')
            l2 = plt.Line2D([x0+0.5*width-0.4*height,x0+0.5*width+0.4*height], [0.5*height,0.5*height], linestyle='-', color='xkcd:red',lw=1)
            l3 = plt.Line2D([0.5*width,0.5*width], [0.1*height,0.9*height], linestyle='-', color='xkcd:red',lw=1)
            return [l1, l2 ,l3]
        if orig_handle == 'ErrorBar':
            l1 = plt.Line2D([width/2], [0.5*height], marker = 'o', ms=10,color='w',markeredgecolor='k',alpha=0.2)
            l2 = plt.Line2D([x0,x0], [0.2*height,0.8*height], linestyle='-', color='k',lw=2)
            l3 = plt.Line2D([y0+width,y0+width], [0.2*height,0.8*height], linestyle='-', color='k',lw=2)
            l4 = plt.Line2D([x0,y0+width], [0.5*height,0.5*height], color='k',lw=2)
            return [l1, l2 ,l3, l4]
        elif orig_handle == 'Limit':
            l1 = plt.Line2D([2*width/3], [0.5*height], marker = 'o', ms=10,color='w',markeredgecolor='k',alpha=0.2)
            l2 = plt.Line2D([x0+width/3,x0+width/3], [0.2*height,0.8*height], linestyle='-', color='k',lw=2)
            l3 = plt.Line2D([y0+width,y0+width], [0.2*height,0.8*height], linestyle='-', color='k',lw=2)
            l4 = plt.Line2D([x0+width/3,y0+width], [0.5*height,0.5*height], color='k',lw=2)
            l5 = plt.Line2D([x0,y0+width/3], [0.5*height,0.5*height], color='k',ls=':',lw=2)
            l6 = plt.Line2D([x0+3*0.2*height,x0,x0+3*0.2*height], [0.2*height,0.5*height,0.8*height], color='k',ls='-',lw=2)
            return [l1, l2 ,l3, l4, l5, l6]
        elif orig_handle == 'UpperLimit':
            #l1 = plt.Line2D([width/2], [0.5*height], marker = 'o', ms=10,color='w',markeredgecolor='k',alpha=0.2)
            l2 = plt.Line2D([x0+width,x0+width], [0.2*height,0.8*height], linestyle='-', color='k',lw=2)
            l3 = plt.Line2D([x0,y0+width], [0.5*height,0.5*height], color='k',lw=2)
            l4 = plt.Line2D([x0+3*0.2*height,x0,x0+3*0.2*height], [0.2*height,0.5*height,0.8*height], color='k',ls='-',lw=2)
            return [l1, l2 ,l3, l4]
        elif orig_handle == 'LowerLimit':
            l1 = plt.Line2D([width/2], [0.5*height], marker = 'o', ms=10,color='w',markeredgecolor='k',alpha=0.2)
            l2 = plt.Line2D([x0+width,x0+width], [0.2*height,0.8*height], linestyle='-', color='k',lw=2)
            l3 = plt.Line2D([x0,y0+width], [0.5*height,0.5*height], color='k',lw=2)
            l4 = plt.Line2D([x0+3*0.2*height,x0,x0+3*0.2*height], [0.2*height,0.5*height,0.8*height], color='k',ls='-',lw=2)
            return [l1, l2 ,l3, l4]
        elif orig_handle == 'Unconstrained':
            l1 = plt.Line2D([width/2], [0.5*height],marker = 'o',ms=10,color='w',markeredgecolor='k',alpha=0.2)
            l2 = plt.Line2D([x0+width-3*0.2*height,x0+width,x0+width-3*0.2*height], [0.2*height,0.5*height,0.8*height], linestyle='-', color='k',lw=2)
            l3 = plt.Line2D([x0,y0+width], [0.5*height,0.5*height], color='k',lw=2,ls=':')
            l4 = plt.Line2D([x0+3*0.2*height,x0,x0+3*0.2*height], [0.2*height,0.5*height,0.8*height], color='k',ls='-',lw=2)
            return [l1, l2 ,l3, l4]
        elif orig_handle == 'SNR5':
            l1 = plt.Line2D([x0,x0+width], [0.5*height,0.5*height],lw=2,color='k',ls='-',alpha=0.2)
            l2 = plt.Line2D([width/2], [0.5*height],marker = 'o',ms=10,color='w',markeredgecolor='k')
            return [l1,l2]
        elif orig_handle == 'SNR10':
            l1 = plt.Line2D([x0,x0+width], [0.5*height,0.5*height],lw=2,color='k',ls='-',alpha=0.2)
            l2 = plt.Line2D([width/2], [0.5*height],marker = 's',ms=10,color='w',markeredgecolor='k')
            return [l1,l2]
        elif orig_handle == 'SNR15':
            l1 = plt.Line2D([x0,x0+width], [0.5*height,0.5*height],lw=2,color='k',ls='-',alpha=0.2)
            l2 = plt.Line2D([width/2], [0.5*height],marker = 'D',ms=10,color='w',markeredgecolor='k')
            return [l1,l2]
        elif orig_handle == 'SNR20':
            l1 = plt.Line2D([x0,x0+width], [0.5*height,0.5*height],lw=2,color='k',ls='-',alpha=0.2)
            l2 = plt.Line2D([width/2], [0.5*height],marker = 'v',ms=10,color='w',markeredgecolor='k')
            return [l1,l2]
        elif orig_handle == 'R20':
            l1 = plt.Line2D([width/2], [0.5*height],marker = 'o',ms=10,color='w',markeredgecolor='k',alpha=0.2)
            l2 = plt.Line2D([x0,x0+width], [0.5*height,0.5*height],lw=2,color='C3',ls='-')
            return [l1,l2]
        elif orig_handle == 'R35':
            l1 = plt.Line2D([width/2], [0.5*height],marker = 'o',ms=10,color='w',markeredgecolor='k',alpha=0.2)
            l2 = plt.Line2D([x0,x0+width], [0.5*height,0.5*height],lw=2,color='C2',ls='-')
            return [l1,l2]
        elif orig_handle == 'R50':
            l1 = plt.Line2D([width/2], [0.5*height],marker = 'o',ms=10,color='w',markeredgecolor='k',alpha=0.2)
            l2 = plt.Line2D([x0,x0+width], [0.5*height,0.5*height],lw=2,color='C0',ls='-')
            return [l1,l2]
        elif orig_handle == 'R100':
            l1 = plt.Line2D([width/2], [0.5*height],marker = 'o',ms=10,color='w',markeredgecolor='k',alpha=0.2)
            l2 = plt.Line2D([x0,x0+width], [0.5*height,0.5*height],lw=2,color='C1',ls='-')
            return [l1,l2]
        elif orig_handle == 'Prior':
            l1 = plt.Line2D([width/2], [0.5*height],marker = 'o',ms=10,color='w',markeredgecolor='k',alpha=0.2)
            l2 = plt.Line2D([x0,x0+width], [0.5*height,0.5*height],lw=2,color='Black',ls='-')
            return [l1,l2]
        else:
            title = mtext.Text(x0, y0, orig_handle + '',weight='bold', usetex=False, **self.text_props,fontsize=18)
            return [title]

def Posterior_Classification_Errorbars(
        mol_sample,
        mol_name,
        p0_SSG = [8,6,0.007,0.5,0.5],
        output_dir = '',
        plotting=False
        ):
    
    p0_SS=None
    p0_u_SS=None
    x_bins = np.linspace(-10,0,1000)
    binned_data = np.histogram(mol_sample,bins=100,density=True)
    
    max_L = binned_data[1][np.argmax(binned_data[0])]
    
    cum_distr = np.array([sum(binned_data[0][:i]) for i in range(len(binned_data[0]))])/sum(binned_data[0])
    
    percentile_16 = binned_data[1][np.argmin(np.abs(cum_distr - 0.16))]
    percentile_84 = binned_data[1][np.argmin(np.abs(cum_distr - 0.84))]
    sigma = (percentile_84-percentile_16)
    if sigma <= 0:
        sigma = 0.1
    
    p0_SSG = [8,-max_L,0.007,sigma,0.5]
    p0_Gauss=[max(binned_data[0]),max_L,sigma]
    #print('SSG',p0_SSG)
    #print('G',p0_Gauss)
    model_likelihood = []
                    
    # Try to Fit each model to the retrieved data
    #params_F,params_SS,params_SSG,params_G,params_u_SS = None,None,None,None,None
    try:
        params_F,cov_F = sco.curve_fit(Model_Flat,(binned_data[1][:-1]+binned_data[1][1:])/2,binned_data[0])
        model_likelihood.append(log_likelihood(params_F,(binned_data[1][:-1]+binned_data[1][1:])/2,binned_data[0],Model_Flat))
    except:
        model_likelihood.append(-np.inf)
                        
    try:
        params_SS,cov_SS = sco.curve_fit(Model_SoftStep,(binned_data[1][:-1]+binned_data[1][1:])/2,binned_data[0],p0=p0_SS)
        model_likelihood.append(log_likelihood(params_SS,(binned_data[1][:-1]+binned_data[1][1:])/2,binned_data[0],Model_SoftStep))
    except:
        model_likelihood.append(-np.inf)
                    
    try:
        params_SSG,cov_SSG = sco.curve_fit(Model_SoftStepG,(binned_data[1][:-1]+binned_data[1][1:])/2,binned_data[0],p0=p0_SSG)
        model_likelihood.append(log_likelihood(params_SSG,(binned_data[1][:-1]+binned_data[1][1:])/2,binned_data[0],Model_SoftStepG))
        line_SSG = Model_SoftStepG(x_bins,*params_SSG)
    except:
        model_likelihood.append(-np.inf)
        params_SSG = p0_SSG
                    
    try:
        params_G,cov_G = sco.curve_fit(Model_Gauss,(binned_data[1][:-1]+binned_data[1][1:])/2,binned_data[0],p0=p0_Gauss)
        model_likelihood.append(log_likelihood(params_G,(binned_data[1][:-1]+binned_data[1][1:])/2,binned_data[0],Model_Gauss))
    except:
        model_likelihood.append(-np.inf)

    try:
        params_u_SS,cov_u_SS = sco.curve_fit(Model_upper_SoftStep,(binned_data[1][:-1]+binned_data[1][1:])/2,binned_data[0],p0=p0_u_SS)
        model_likelihood.append(log_likelihood(params_u_SS,(binned_data[1][:-1]+binned_data[1][1:])/2,binned_data[0],Model_upper_SoftStep))
    except:
        model_likelihood.append(-np.inf)
    # Select the optimal model for the considered data case
    s_ssg_max=5
    span = 10
    post_limit = 0
    print(mol_name,model_likelihood)
    #print(params_SSG)
    if model_likelihood[3]!=-np.inf:
        if params_G[2]>span/3.0:
            model_likelihood[3]=-np.inf
    if model_likelihood[1]!=-np.inf:
        if Model_SoftStep(post_limit,*params_SS)>=params_SS[-1]/10:
            model_likelihood[1]=-np.inf
    if model_likelihood[2]!=-np.inf:
        if np.max(line_SSG)<=1.4*params_SSG[2] or np.max(line_SSG)>=15*params_SSG[2]:
            #print('1')
            model_likelihood[2]=-np.inf
        if line_SSG[0]>=1.05*params_SSG[2]:
            #print('2')
            model_likelihood[2]=-np.inf
            """
        if line_SSG[-1]>=params_SSG[2]/20:
            #print('3')
            model_likelihood[2]=-np.inf
            """
        if params_SSG[-2]>=s_ssg_max:
            #print('4')
            model_likelihood[2]=-np.inf
    if model_likelihood[4]!=-np.inf:
        if params_u_SS[0]<0:
            model_likelihood[4]=-np.inf
    print(mol_name,model_likelihood)
    # Storing the best fit model for the parameters of interest
    #print(model_likelihood)
    best_fit = np.argmax(model_likelihood)
    if best_fit == 0:
        best_post_model = ['F',params_F]
    elif best_fit == 1:
        best_post_model = ['SS',params_SS]
    elif best_fit == 2:
        best_post_model = ['SSG',params_SSG]
    elif best_fit == 3:
        best_post_model = ['G',params_G]
    elif best_fit == 4:
        best_post_model = ['USS',params_u_SS]
    else:
        print(str(best_fit) + ' is not a valid model!')
    if plotting:
        fig = plt.figure()
        ax = plt.gca()
        h = ax.hist(mol_sample,bins=100,alpha=0.2,density=True)
        
        formatting = lambda x: '{v:0.2f}'.format(v=x)
        try:
            ax.plot(x_bins,Model_Flat(x_bins,*params_F),'g-',lw=1,label='F: ' + ', '.join(map(formatting,params_F)))
        except Exception as e:
            print(e)
            pass
        try:
            ax.plot(x_bins,Model_SoftStep(x_bins,*params_SS),'r-',lw=1,label='SS: ' + ', '.join(map(formatting,params_SS)))
        except Exception as e:
            print(e)
            pass
        try:
            ax.plot(x_bins,Model_SoftStepG(x_bins,*params_SSG),'b-',lw=1,label='SSG: ' + ', '.join(map(formatting,params_SSG)))
        except Exception as e:
            print(e)
            pass
        try:
            ax.plot(x_bins,Model_Gauss(x_bins,*params_G),'m-',lw=1,label='G: ' + ', '.join(map(formatting,params_G)))
        except Exception as e:
            print(e)
            pass
        try:
            ax.plot(x_bins,Model_upper_SoftStep(x_bins,*params_u_SS),'y-',lw=1,label='USS: ' + ', '.join(map(formatting,params_u_SS)))
        except Exception as e:
            print(e)
            pass
        
        ax.plot([-15,0],[0,0],'k-',alpha=1)
        ax.set_title(mol_name + '\nBest-fit: '+best_post_model[0])
        ax.set_ylim([-max(h[0])/4,1.1*max(h[0])])
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlim((-10,0))
        ax.legend()
        fig.savefig(output_dir + mol_name +'_classification_plot.png')
        
    
    return best_post_model

def log_arrow(mid,width):
    return [10**(np.log10(mid)-width),mid,10**(np.log10(mid)+width)]

def Errorbars_plot(ax,model,pressure_distr,p_bounds,marker_color,lw = 0.5,un_c_len=2,
                   errorbar_color = None,
                   plot_marker = False):
    x=np.linspace(-10,0,1000)
    p_low,p_max,p_high = p_bounds
    yerr = np.array([[abs(p_max-p_low)],[abs(p_max-p_high)]])
    line = None
    
    marker=None
    ms=2.5*lw
    markerfacecolor=marker_color
    markeredgecolor=marker_color
    if errorbar_color is None:
        errorbar_color = marker_color
    if plot_marker:
        marker = 'o'
    
    if model[0] == 'F':
        line = Model_Flat(x,*list(model[1]))
        y_low = abs(p_max - p_low)
        y_high = abs(p_high - p_max)
        HM = -5
        ax.errorbar(x = 10**HM,y=p_max,xerr = None,yerr = yerr,color=errorbar_color,capsize=2,capthick=lw,lw=lw,marker=marker,ms=ms,markerfacecolor=markerfacecolor,markeredgecolor=markeredgecolor)
        ax.plot([10**(HM-un_c_len),10**(HM+un_c_len)],[p_max,p_max],ls=':',color=errorbar_color,linewidth=lw)
        ax.plot([10**(HM-un_c_len+un_c_len/8),10**(HM-un_c_len),10**(HM-un_c_len+un_c_len/8)],log_arrow(p_max,un_c_len/16),ls='-',color=errorbar_color,linewidth=lw)
        ax.plot([10**(HM+un_c_len-un_c_len/8),10**(HM+un_c_len),10**(HM+un_c_len-un_c_len/8)],log_arrow(p_max,un_c_len/16),ls='-',color=errorbar_color,linewidth=lw)
        
        
    if model[0] == 'SS':
        line = Model_SoftStep(x,*list(model[1]))
        HM = -model[1][1]/model[1][0]
        s = Inv_Model_SoftStep(0.16*model[1][-1],model[1][0],model[1][1],model[1][2])
        y_low = abs(p_max - p_low)
        y_high = abs(p_high - p_max)
        xerr = np.array([[0],[abs(10**HM - 10**(HM + s))]])
        ax.errorbar(x = 10**HM,y=p_max,xerr = xerr,yerr = yerr,color=errorbar_color,capsize=2,capthick=lw,lw=lw,marker=marker,ms=ms,markerfacecolor=markerfacecolor,markeredgecolor=markeredgecolor)
        
        ax.plot([10**(HM),10**(HM-un_c_len)],[p_max,p_max],ls=':',color=errorbar_color,linewidth=lw)
        ax.plot([10**(HM-un_c_len+un_c_len/8),10**(HM-un_c_len),10**(HM-un_c_len+un_c_len/8)],log_arrow(p_max,un_c_len/16),ls='-',color=errorbar_color,linewidth=lw)
        
        
    if model[0] == 'SSG':
        line = Model_SoftStepG(x,*list(model[1]))
        ind = np.argmax(line)
        x_up = x[np.where(x>x[ind])]
        xp = x_up[np.argmin(np.abs(Model_SoftStepG(x_up,*model[1])-1/2*np.max(line)))]
        x_down = x[np.where(x<x[ind])]
        xm = x_down[np.argmin(np.abs(Model_SoftStepG(x_down,*model[1])-(np.max(line)/2+model[1][2]/2)))]
        xerr = np.array([[abs(10**x[ind] - 10**xm)],[abs(10**x[ind] - 10**xp)]])
        ax.errorbar(x=10**x[ind],y=p_max,xerr = xerr,yerr = yerr,color=errorbar_color,capsize=2,capthick=lw,lw=lw,marker=marker,ms=ms,markerfacecolor=markerfacecolor,markeredgecolor=markeredgecolor)
        
        ax.plot([10**xm,10**(xm-un_c_len)],[p_max,p_max],ls=':',color=errorbar_color,linewidth=lw)
        ax.plot([10**(xm-un_c_len+un_c_len/8),10**(xm-un_c_len),10**(xm-un_c_len+un_c_len/8)],log_arrow(p_max,un_c_len/16),ls='-',color=errorbar_color,linewidth=lw)
        
        

        
    if model[0] == 'G':
        line = Model_Gauss(x,*list(model[1]))
        s=model[1][-1]
        mean=model[1][-2]
        xerr = np.array([[abs(10**mean - 10**(mean - s))],[abs(10**mean - 10**(mean + s))]])
        ax.errorbar(x=10**mean,y=p_max,xerr = xerr,yerr = yerr,color=errorbar_color,capsize=2,capthick=lw,lw=lw,marker=marker,ms=ms,markerfacecolor=markerfacecolor,markeredgecolor=markeredgecolor)
    if model[0] == 'USS':
        line = Model_upper_SoftStep(x,*list(model[1]))
        HM = model[1][1]/model[1][0]
        s = Inv_Model_SoftStep(0.16*model[1][-1],model[1][0],model[1][1],model[1][2])
        y_low = abs(p_max - p_low)
        y_high = abs(p_high - p_max)
        xerr = np.array([[abs(10**HM - 10**(HM - s))],[0]])
        ax.errorbar(x = 10**HM,y=p_max,xerr = xerr,yerr = yerr,color=errorbar_color,capsize=2,capthick=lw,lw=lw,marker=marker,ms=ms,markerfacecolor=markerfacecolor,markeredgecolor=markeredgecolor)
        
        ax.plot([10**HM,10**(HM+un_c_len)],[p_max,p_max],ls=':',color=errorbar_color,linewidth=lw)
        ax.plot([10**(HM+un_c_len-un_c_len/8),10**(HM+un_c_len),10**(HM+un_c_len-un_c_len/8)],log_arrow(p_max,un_c_len/16),ls='-',color=errorbar_color,linewidth=lw)
        
    return line

def compute_model_line(model):
    model_chosen = model[0]
    x=np.linspace(-10,0,1000)
    if model_chosen == 'F':
        model_line = Model_Flat(x,*list(model[1]))
    elif model_chosen == 'SS':
        model_line = Model_SoftStep(x,*list(model[1]))
    elif model_chosen == 'SSG':
        model_line = Model_SoftStepG(x,*list(model[1]))
    elif model_chosen == 'USS':
        model_line = Model_upper_SoftStep(x,*list(model[1]))
    elif model_chosen == 'G':
        model_line = Model_Gauss(x,*list(model[1]))
    return model_line

def compute_SSG_errorbars(model):
    x=np.linspace(-10,0,1000)
    line = Model_SoftStepG(x,*list(model[1]))
    ind = np.argmax(line)
    x_up = x[np.where(x>x[ind])]
    xp = x_up[np.argmin(np.abs(Model_SoftStepG(x_up,*model[1])-1/2*np.max(line)))]
    x_down = x[np.where(x<x[ind])]
    xm = x_down[np.argmin(np.abs(Model_SoftStepG(x_down,*model[1])-(np.max(line)/2+model[1][2]/2)))]
    return [xm,x[ind],xp]

"""
FUNCTION DEFINED BY BJOERN

def Posterior_Classification(self,parameters=['H2O','CO2','CO','H2SO484(c)','R_pl','M_pl'],relative=None,limits = None,plot_classification=True,p0_SSG=[8,6,0.007,0.5,0.5],p0_SS=None,s_max=2,s_ssg_max=5):
    self.best_post_model = {}
    self.best_post_limit = {}

    count_param = 0

    # Iterate over all parameters of interest
    for param in parameters:
        try:
            keys = list(self.params.keys())
            ind = np.where(np.array(keys)==str(param))[0][0]
            post = self.equal_weighted_post[:,ind]
            if relative is not None:
                ind_rel = np.where(np.array(keys)==str(relative))[0][0]
                post_rel = self.equal_weighted_post[:,ind_rel]

            if limits is None:
                prior = self.priors[ind]
                                
                if prior in ['ULU', 'LU']:
                    if prior == 'LU':
                        if relative is None:
                            post = np.log10(post)
                        else:
                            post = np.log10(post) - np.log10(post_rel) 
                        self.best_post_limit[param] = sorted(list(self.priors_range[ind]))
                    else:
                        if relative is None:
                            post = np.log10(1-post)
                        else:
                            post = np.log10(post) - np.log10(post_rel) 
                        self.best_post_limit[param] = [-7,0]
                elif prior in ['G','LG']:
                    mean = self.priors_range[ind][0]
                    sigma  = self.priors_range[ind][1]
                    self.best_post_limit[param] = [mean-5*sigma,mean+5*sigma]
                    if prior == 'LG':
                        post = np.log10(post)
                else:
                    self.best_post_limit[param] = limits[count_param]

            span = abs(self.best_post_limit[param][1]-self.best_post_limit[param][0])
            center = span/2 + self.best_post_limit[param][0]
            p0_Gauss=[span/5,center,1.0]
            p0_SS=[-span/5,center,1.0]
            p0_u_SS=[span/5,center,1.0]


            x_bins = np.linspace(self.best_post_limit[param][0],self.best_post_limit[param][1],1000)
            binned_data = np.histogram(post,bins=100,range=self.best_post_limit[param],density=True)
                            
            model_likelihood = []
                            
            # Try to Fit each model to the retrieved data
            try:
                params_F,cov_F = sco.curve_fit(Model_Flat,(binned_data[1][:-1]+binned_data[1][1:])/2,binned_data[0])
                model_likelihood.append(log_likelihood(params_F,(binned_data[1][:-1]+binned_data[1][1:])/2,binned_data[0],Model_Flat))
            except:
                model_likelihood.append(-np.inf)
                                
            try:
                params_SS,cov_SS = sco.curve_fit(Model_SoftStep,(binned_data[1][:-1]+binned_data[1][1:])/2,binned_data[0],p0=p0_SS)
                model_likelihood.append(log_likelihood(params_SS,(binned_data[1][:-1]+binned_data[1][1:])/2,binned_data[0],Model_SoftStep))
            except:
                model_likelihood.append(-np.inf)
                            
            try:
                params_SSG,cov_SSG = sco.curve_fit(Model_SoftStepG,(binned_data[1][:-1]+binned_data[1][1:])/2,binned_data[0],p0=p0_SSG)
                model_likelihood.append(log_likelihood(params_SSG,(binned_data[1][:-1]+binned_data[1][1:])/2,binned_data[0],Model_SoftStepG))
                line_SSG = Model_SoftStepG(x_bins,*params_SSG)
            except:
                model_likelihood.append(-np.inf)
                            
            try:
                params_G,cov_G = sco.curve_fit(Model_Gauss,(binned_data[1][:-1]+binned_data[1][1:])/2,binned_data[0],p0=p0_Gauss)
                model_likelihood.append(log_likelihood(params_G,(binned_data[1][:-1]+binned_data[1][1:])/2,binned_data[0],Model_Gauss))
            except:
                model_likelihood.append(-np.inf)

            try:
                params_u_SS,cov_u_SS = sco.curve_fit(Model_upper_SoftStep,(binned_data[1][:-1]+binned_data[1][1:])/2,binned_data[0],p0=p0_u_SS)
                model_likelihood.append(log_likelihood(params_u_SS,(binned_data[1][:-1]+binned_data[1][1:])/2,binned_data[0],Model_upper_SoftStep))
            except:
                model_likelihood.append(-np.inf)

            # Select the optimal model for the considered data case
            if model_likelihood[3]!=-np.inf:
                if params_G[2]>span/6.0:
                    model_likelihood[3]=-np.inf
            if model_likelihood[1]!=-np.inf:
                if Model_SoftStep(self.best_post_limit[param][1],*params_SS)>=params_SS[-1]/10:
                    model_likelihood[1]=-np.inf
            if model_likelihood[2]!=-np.inf:
                if np.max(line_SSG)<=1.4*params_SSG[2] or np.max(line_SSG)>=15*params_SSG[2]:
                    model_likelihood[2]=-np.inf
                if line_SSG[0]>=1.05*params_SSG[2]:
                    model_likelihood[2]=-np.inf
                if line_SSG[-1]>=params_SSG[2]/20:
                    model_likelihood[2]=-np.inf
                if params_SSG[-2]>=s_ssg_max:
                    model_likelihood[2]=-np.inf
            if model_likelihood[4]!=-np.inf:
                if params_u_SS[0]<0:
                    model_likelihood[4]=-np.inf

            # Storing the best fit model for the parameters of interest
            best_fit = np.argmax(model_likelihood)
            if best_fit == 0:
                self.best_post_model[param] = ['F',params_F]
            elif best_fit == 1:
                self.best_post_model[param] = ['SS',params_SS]
            elif best_fit == 2:
                self.best_post_model[param] = ['SSG',params_SSG]
            elif best_fit == 3:
                self.best_post_model[param] = ['G',params_G]
            elif best_fit == 4:
                self.best_post_model[param] = ['USS',params_u_SS]
            else:
                print(str(best_fit) + ' is not a valid model!')
    
            if plot_classification:
                plt.figure(figsize=(10,10))
                h = plt.hist(post,bins=100,range=self.best_post_limit[param],alpha=0.2,density=True)


                if best_fit == 0:
                    plt.plot(x_bins,Model_Flat(x_bins,*params_F),'g-',lw=5)
                if best_fit == 1:
                    plt.plot(x_bins,Model_SoftStep(x_bins,*params_SS),'r-',lw=5)
                if best_fit == 2:
                    plt.plot(x_bins,Model_SoftStepG(x_bins,*params_SSG),'b-',lw=5)
                if best_fit == 3:
                    plt.plot(x_bins,Model_Gauss(x_bins,*params_G),'m-',lw=5)               
                if best_fit == 4:
                    plt.plot(x_bins,Model_upper_SoftStep(x_bins,*params_u_SS),'y-',lw=5)   
                plt.plot([-15,0],[0,0],'k-',alpha=1)
                plt.ylim([-max(h[0])/4,1.1*max(h[0])])
                plt.yticks([])
                plt.xticks([])
                plt.xlim(self.best_post_limit[param])
                plt.show()
        except:
            print(str(param) + ' was not a retrieved parameter in this retrieval run')

        count_param += 1


def Limit_Plot(self,n,N,model,SNR_O,ms,lw,eb,c,m,centre =-3.5 ,un_c_len=2):
    x=np.linspace(-15,15,1000)     
    print(model)
    
    if model[0] == 'F':
        plt.plot([centre,centre-un_c_len],[2*(N-n)+SNR_O,2*(N-n)+SNR_O],ls=':',color=c,linewidth=lw)
        plt.plot([centre-un_c_len+3*eb,centre-un_c_len,centre-un_c_len+3*eb],[2*(N-n)+eb+SNR_O,2*(N-n)+SNR_O,2*(N-n)-eb+SNR_O],ls='-',color=c,linewidth=lw)
        plt.plot([centre+un_c_len,centre],[2*(N-n)+SNR_O,2*(N-n)+SNR_O],ls=':',color=c,linewidth=lw)
        plt.plot([centre+un_c_len-3*eb,centre+un_c_len,centre+un_c_len-3*eb],[2*(N-n)+eb+SNR_O,2*(N-n)+SNR_O,2*(N-n)-eb+SNR_O],ls='-',color=c,linewidth=lw)
        plt.plot(centre,2*(N-n)+SNR_O,color=c,marker = m,markersize = ms,markeredgecolor='black',markeredgewidth=lw/4)

    if model[0] == 'SS':
        HM = -model[1][1]/model[1][0]
        s = Inv_Model_SoftStep(0.16*model[1][-1],model[1][0],model[1][1],model[1][2])
        plt.plot([HM,HM-un_c_len],[2*(N-n)+SNR_O ,2*(N-n)+SNR_O ],ls='-',color=c,linewidth=lw)
        plt.plot([HM-un_c_len+3*eb,HM-un_c_len,HM-un_c_len+3*eb],[2*(N-n)+eb+SNR_O ,2*(N-n)+SNR_O ,2*(N-n)-eb+SNR_O],ls='-',color=c,linewidth=lw)
        plt.plot([HM,s],[2*(N-n)+SNR_O ,2*(N-n)+SNR_O],ls='-',color=c,linewidth=lw)
        plt.plot([s,s],[2*(N-n)+eb+SNR_O,2*(N-n)-eb+SNR_O],ls='-',color=c,linewidth=lw)
        plt.plot(HM,2*(N-n)+SNR_O ,color=c,marker = m,markersize = ms,markeredgecolor='black',markeredgewidth=lw/4)

    if model[0] == 'SSG':
        line = Model_SoftStepG(x,*list(model[1]))
        ind = np.argmax(line)
        x_up = x[np.where(x>x[ind])]
        xp = x_up[np.argmin(np.abs(Model_SoftStepG(x_up,*model[1])-1/2*np.max(line)))]
        x_down = x[np.where(x<x[ind])]
        xm = x_down[np.argmin(np.abs(Model_SoftStepG(x_down,*model[1])-(np.max(line)/2+model[1][2]/2)))]
        plt.plot([xp,xm],[2*(N-n)+SNR_O,2*(N-n)+SNR_O],ls='-',color=c,linewidth=lw)
        plt.plot([xp,xp],[2*(N-n)+SNR_O-eb,2*(N-n)+SNR_O+eb],ls='-',color=c,linewidth=lw)
        plt.plot([xm,xm],[2*(N-n)+SNR_O-eb,2*(N-n)+SNR_O+eb],ls='-',color=c,linewidth=lw)
        plt.plot([xm,xm-un_c_len],[2*(N-n)+SNR_O,2*(N-n)+SNR_O],ls=':',color=c,linewidth=lw)
        plt.plot([xm-un_c_len+3*eb,xm-un_c_len,xm-un_c_len+3*eb],[2*(N-n)+SNR_O-eb,2*(N-n)+SNR_O,2*(N-n)+SNR_O+eb],ls='-',color=c,linewidth=lw)
        plt.plot(x[ind],2*(N-n)+SNR_O,color=c,marker = m,markersize = ms,markeredgecolor='black',markeredgewidth=lw/4)

    if model[0] == 'G':
        s=model[1][-1]
        mean=model[1][-2]
        q50 = mean #np.quantile(data,0.5,axis=0)
        q84 = mean+s #np.quantile(data,0.84,axis=0)
        q16 = mean-s #np.quantile(data,0.16,axis=0)
        plt.plot([q84,q16],[2*(N-n)+SNR_O,2*(N-n)+SNR_O],ls='-',color=c,linewidth=lw)
        plt.plot([q16,q16],[2*(N-n)+eb+SNR_O,2*(N-n)-eb+SNR_O],ls='-',color=c,linewidth=lw)
        plt.plot([q84,q84],[2*(N-n)+eb+SNR_O,2*(N-n)-eb+SNR_O],ls='-',color=c,linewidth=lw)
        plt.plot([q50,q50],[2*(N-n)+SNR_O,2*(N-n)+SNR_O],color=c,marker = m,markersize = ms,markeredgecolor='black',markeredgewidth=lw/4)

    if model[0] == 'USS':
        HM = -model[1][1]/model[1][0]
        s = Inv_Model_SoftStep(0.16*model[1][-1],model[1][0],model[1][1],model[1][2])
        plt.plot([-HM,-HM+un_c_len],[2*(N-n)+SNR_O ,2*(N-n)+SNR_O ],ls='-',color=c,linewidth=lw)
        plt.plot([-HM+un_c_len+3*eb,-HM+un_c_len,-HM+un_c_len+3*eb],[2*(N-n)+eb+SNR_O ,2*(N-n)+SNR_O ,2*(N-n)-eb+SNR_O],ls='-',color=c,linewidth=lw)
        plt.plot([-HM,s],[2*(N-n)+SNR_O ,2*(N-n)+SNR_O],ls='-',color=c,linewidth=lw)
        plt.plot([s,s],[2*(N-n)+eb+SNR_O,2*(N-n)-eb+SNR_O],ls='-',color=c,linewidth=lw)
        plt.plot(-HM,2*(N-n)+SNR_O ,color=c,marker = m,markersize = ms,markeredgecolor='black',markeredgewidth=lw/4)


"""