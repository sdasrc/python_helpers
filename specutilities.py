import os, sys
import numpy as np
pylibdir = os.environ['PYLIBDIR']
sys.path.insert(1, pylibdir)
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse, Rectangle
from matplotlib import transforms
import matplotlib.ticker
class MyLocator(matplotlib.ticker.AutoMinorLocator):
    def __init__(self, n=10):
        super().__init__(n=n)
matplotlib.ticker.AutoMinorLocator = MyLocator   
plt.rcParams['figure.figsize'] = (7,5)
plt.rcParams['font.size'] = 18
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['xtick.labelcolor'] = 'black'
plt.rcParams['ytick.labelcolor'] = 'black'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble']=r'\usepackage{amsmath}'
cb_blue,cb_purple,cb_magenta,cb_orange,cb_gold = '#648FFF','#785EF0','#DC267F','#FE6100','#FFB000'
from prosfitsutilities import *

def kauf_line():
    ex0 = np.linspace(-3.5, 1.5, 100)
    ey0 = 1.3 + 0.61/(ex0 - 0.05)
    ey0[ex0 > 0.05] = np.nan
    return ex0, ey0

def kew_line():
    ex1 = np.linspace(-3.5, 1.5, 100)
    ey1 = 1.19 + 0.61/(ex1 - 0.47)
    ey1[ex1 > 0.47] = np.nan
    return ex1, ey1

def cid_line():
    ex2 = np.linspace(-3.5, 1.5, 100)
    ey2 = 1.01* ex2 + 0.48
    ey2[ex2 < -0.435] = np.nan
    return ex2, ey2


# where do they lie???
def kauf_class(x,y):
    typex, typey = type_str(x), type_str(y)
    if typex=='list' or 'array' in typex : x = np.array(x)
    elif typex=='float' or typex=='int': x = np.array([x])
    else: raise ValueError("Input(s) must be int/float/array.")
    if typey=='list' or 'array' in typey: y = np.array(y)
    elif typey=='float' or typey=='int': y = np.array([y])
    else: raise ValueError("Input(s) must be int/float/array.")
        
    # x = log(NII/Ha), y = log(OIII/Hb)
    ycut = 1.3 + 0.61/(x - 0.05)
    res = (y > ycut).astype('int')
    res[x >= 0.05] = 1
    return res

def kew_class(x,y):
    typex, typey = type_str(x), type_str(y)
    if typex=='list' or 'array' in typex : x = np.array(x)
    elif typex=='float' or typex=='int': x = np.array([x])
    else: raise ValueError("Input(s) must be int/float/array.")
    if typey=='list' or 'array' in typey: y = np.array(y)
    elif typey=='float' or typey=='int': y = np.array([y])
    else: raise ValueError("Input(s) must be int/float/array.")
        
    ycut = 1.19 + 0.61/(x - 0.47)
    res = (y > ycut).astype('int')
    res[x >= 0.4] = 1
    return res

def cid_class(x,y):
    typex, typey = type_str(x), type_str(y)
    if typex=='list' or 'array' in typex : x = np.array(x)
    elif typex=='float' or typex=='int': x = np.array([x])
    else: raise ValueError("Input(s) must be int/float/array.")
    if typey=='list' or 'array' in typey: y = np.array(y)
    elif typey=='float' or typey=='int': y = np.array([y])
    else: raise ValueError("Input(s) must be int/float/array.")
        
    ycut = 1.01* x + 0.48
    return y > ycut

def balmer_corr_halpha(Halpha_6563,Hbeta_4861):
    Halpha_6563, Hbeta_4861 = arr_check(Halpha_6563), arr_check(Hbeta_4861)
    Lsolar = 3.827e26
    Ebv = 1.97*np.log10((Halpha_6563/Hbeta_4861)/2.86) #see Dominguez et al 2013
    A_Ha = 3.33*Ebv #+-0.8# using the Calzetti2000 reddening curve
    # balmer_OK=((em_corr[:,4]/em_corr_err[:,4])>3) & ((em_corr[:,0]/em_corr_err[:,0])>3)
    A_Ha[A_Ha < 0] = 0
    Halpha_6563_corr = Halpha_6563* (10**(0.4*A_Ha))
    return Halpha_6563_corr          

def bptclass(logNIIHa,logOIIIHb):
        
    typex, typey = type_str(logNIIHa), type_str(logOIIIHb)
    
    if 'list' in typex or 'array' in typex : logNIIHa = np.array(logNIIHa)
    elif 'float' in typex or 'int' in typex: logNIIHa = np.array([logNIIHa])
    else: raise ValueError("Input(s) must be int/float/array.")
        
    if 'list' in typey or 'array' in typey: logOIIIHb = np.array(logOIIIHb)
    elif 'float' in typey or 'int' in typey: logOIIIHb = np.array([logOIIIHb])
    else: raise ValueError("Input(s) must be int/float/array.")
        
    final_class = -1*np.ones(len(logNIIHa))
    iskauf = kauf_class(logNIIHa,logOIIIHb).astype('int')
    iscid = cid_class(logNIIHa,logOIIIHb).astype('int')
    goodroi = (np.isfinite(iskauf)) & (np.isfinite(iscid))
    final_class[goodroi & (iskauf == 0)] = 0 # sfg
    final_class[goodroi & (iskauf == 1) & (iscid == 0)] = 1 # liner
    final_class[goodroi & (iskauf == 1) & (iscid == 1)] = 2 # seyfert
    
    return final_class.astype('int')

def get_confmatrix(xdatarr, ydatarr, vals2match):
    Nmat = len(vals2match)
    confmatrx = np.zeros((Nmat,Nmat))
    for yidx in range(Nmat):
        for xidx in range(Nmat):
            confmatrx[yidx,xidx] = (np.sum((ydatarr == vals2match[yidx]) & (xdatarr == vals2match[xidx])))
            
    confmatrx = confmatrx.astype('int')   

    confmatrx_fracs = np.zeros((Nmat,Nmat))
    for yidx in range(Nmat):
        confmatrx_fracs[:,yidx] = confmatrx[:,yidx]*100/np.sum(confmatrx[:,yidx]) \
              if np.sum(confmatrx[:,yidx]) else 0

    return confmatrx.astype('int'), confmatrx_fracs.astype('int')

def plot_confmatrx2(confmatrx, confmatrx_fracs, xdatarr, vals2match, xpltlabel, ypltlabel, pltticklabels):
    fig, axes = plt.subplots(1,2,figsize=(15, 6),layout='compressed')
    ax = axes[0]
    cax = ax.matshow(confmatrx, cmap='Blues', clim=(0,len(ck)))
    ax.xaxis.set_ticks_position('bottom')

    ax.set_xticks(np.arange(len(vals2match)))
    ax.set_yticks(np.arange(len(vals2match)))
    ax.set_xticklabels(pltticklabels)
    ax.set_yticklabels(pltticklabels)

    for (ii, jj), val in np.ndenumerate(confmatrx):
        ax.text(jj, ii, '{0}'.format(val), ha='center', va='center', color='black')
    for jj in range(confmatrx.shape[1]):
        ax.text(jj, -0.7, f'{np.sum(xdatarr == vals2match[jj]):.0f}', ha='center', va='center', color='black')
        
    ax.set_ylabel(ypltlabel)
    ax.set_xlabel(xpltlabel)
    plt.colorbar(cax, ax=ax, label=r'$\mathrm{N(sources)}$')
    # ax.set_title(r'$\mathrm{Numbers}$')

    # ==================================== # 

    ax = axes[1]
    cax = ax.matshow(confmatrx_fracs, cmap='Blues')
    ax.xaxis.set_ticks_position('bottom')
    for jj in range(confmatrx.shape[1]):
        ax.text(jj, -0.7, f'{np.sum(xdatarr == vals2match[jj]):.0f}', ha='center', va='center', color='black')
    for (ii, jj), val in np.ndenumerate(confmatrx_fracs):
        ax.text(jj, ii, '{0:.2f}'.format(val), ha='center', va='center', color='black')
        
    ax.set_xticks(np.arange(len(vals2match)))
    ax.set_yticks(np.arange(len(vals2match)))
    ax.set_xticklabels(pltticklabels)
    ax.set_yticklabels(pltticklabels)
    # ax.set_title(r'$\mathrm{Fractions}$')
    plt.colorbar(cax, ax=ax, label=r'$\mathrm{Percentage}$')

    # ax.set_ylabel(r'$\mathrm{Arnaudova~(in~prep)}$')
    ax.set_xlabel(xpltlabel)      