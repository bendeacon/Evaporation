"""
A module containing functions to calculate uncertainty in QSPR models
"""

import numpy as np
from importlib import reload
import matplotlib.pyplot as plt

from qspr import stracorn, stracorn_cc, stracorn_lp
from uncertainty import hybmdl
reload(hybmdl)
reload(stracorn)
reload(stracorn_cc)
reload(stracorn_lp)


def compUnct_Ksc(XPred=None, disp=1):
    dat_Plp = np.loadtxt("qspr/Kow_Plp.txt")
    dat_Kcc = np.loadtxt("qspr/Kow_Kcc.txt")
    dat_Ksc = np.loadtxt("qspr/Kow_Ksc.txt")
    
    paras0 = np.log( np.array([4.2, 0.31, 0.69]) )
    bnds = ((-10, 10), (-10, 10), (-10, 10))
    
    sig2_y = np.array([0.05])
    sig2_z = np.array([0.05, 0.05])
    
    Xy = dat_Ksc[:,0].reshape((-1, 1))
    Y = np.log10( dat_Ksc[:,1].reshape((-1,1)) )
    Xz = ( dat_Kcc[:,0].reshape((-1,1)), dat_Plp[:,0].reshape((-1,1)) )
    Z = ( np.log10(dat_Kcc[:,1].reshape((-1,1))), np.log10(dat_Plp[:,1].reshape((-1,1))) )
    
    paras = np.empty_like (paras0)
    np.copyto(paras, paras0)
    
    rlt_plugin = hybmdl.PluginMain(qspr_K_sc_plugin, qspr_K_cc_lp, Xy, Y, Xz, Z, paras0, sig2_y, sig2_z, 10, bnds)
    if XPred is None:
        rlt_pred = hybmdl.pred(qspr_K_sc, qspr_K_cc_lp, Xy, rlt_plugin[0], rlt_plugin[1], rlt_plugin[2], rlt_plugin[3])   
        rlt_pred_0 = hybmdl.pred(qspr_K_sc, qspr_K_cc_lp, Xy, paras0, rlt_plugin[1], rlt_plugin[2], rlt_plugin[3])
    else:
        rlt_pred = hybmdl.pred(qspr_K_sc, qspr_K_cc_lp, XPred, rlt_plugin[0], rlt_plugin[1], rlt_plugin[2], rlt_plugin[3])
        rlt_pred_0 = hybmdl.pred(qspr_K_sc, qspr_K_cc_lp, XPred, paras0, rlt_plugin[1], rlt_plugin[2], rlt_plugin[3])
    
    if disp > 1:        
        plt.plot(np.log10(Xy[:,0]), np.squeeze(rlt_pred[2]), 'x', np.log10(Xy[:,0]), Y, 'o')
        plt.plot(np.log10(Xy[:,0]), np.squeeze(rlt_pred0[2]), '^')
        plt.show(0)

    return rlt_pred, rlt_pred_0
        
def qspr_K_cc_lp(theta, Kow):
    ''' Function to predict the volumetric partition coefficient of corneocyte:water and lipid:water
    Here used as a combined function of the LOW-LEVEL of the multi-level model
    Args:
      theta -- model parameters
      Kow, dim: [n_dat, 1]
    Rtns:
      Z -- dim: [n_dat, 2]; each row contains [Kcc, Klp]
    '''

    theta = np.squeeze(np.asarray(theta)) # to avoid some mysterious conversion of np array to np matrix by the optimiser        
    paras_cc = theta[:2]
    paras_lp = theta[2]

    n_dat = Kow.shape[0]    #print X.shape
    Z = np.zeros((n_dat, 2))
    Z[:,0] = stracorn_cc.compK(paras_cc, Kow)
    Z[:,1] = stracorn_lp.compK(paras_lp, Kow)         
    return np.log10(Z)

    
def qspr_K_sc(theta, Kow, Z_lg10):
    ''' Function to predict the volumetric partition coefficient between stratum corneum and water    
    Here used as a function of the TOP-LEVEL of the multi-level model
    Args:
      theta -- model parameters
      Kow, dim: [n_dat, 1]
      Z -- Kcc & Klp, dim: [n_dat, 2]
    Rtns:
      Y -- Ksc, predicted partition coefficient between stratum corneum and water
    '''
    Y = stracorn.compK_from_cc_lp(theta, Kow, 10**Z_lg10)
    #Y1 = stracorn.compK(theta, Kow)
    return np.log10(Y)#(Y, Y1)

def qspr_K_sc_plugin(theta, Kow, func_low):
    ''' Function to predict the volumetric partition coefficient between stratum corneum and water    
    Here used as a test function of the top-level model
    Args:
      theta -- model parameters
      Kow, dim: [n_dat, 1]
    Rtns:
      Y -- Ksc_pred, predicted coefficient between stratum corneum (and water)
    '''
    K_cc_lp_lg10 = func_low(theta, Kow)
    Y = qspr_K_sc(theta, Kow, K_cc_lp_lg10)
    return Y


