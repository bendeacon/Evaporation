'''
Module for QSPR models in stratum corneum lipid (stracorn_lp).
The models are for given compound's partition coefficient between lipid and water,
and its diffusion coefficient in lipid

As a convention, the model parameters, i.e. regression coefficients, should be in natural log
    This is needed for parameter estimation when constraints on positive parameters can be avoided.
We follow "Int. J. Pharm 398 (2010) 114" to use "K" for volumetric partition 
    coefficient, and "P" for mass partition coefficient. Volumetric partition coefficient
    is what is used in the diffusion-based model, but mass partition coefficient is
    usually measured in experiments.
'''

import numpy as np
#from importlib import reload
import matplotlib.pyplot as plt

from qspr.constants import rho_lip, rho_wat

### Functions calculating partition coefficients ###

def compP(paras, Kow):
    ''' Function to compute the MASS partition coefficient between lipid and water    
    '''
    coef = np.exp(paras)        
    lg10Kow = np.log10(Kow)   
    P = 10 ** (coef*lg10Kow)
    return P
        

def compK(paras, Kow):
    ''' Function to compute the VOLUMETRIC partition coefficient between lipid and water
    '''
    if len( Kow.shape ) > 1:
        Kow1 = Kow.flatten()
    else:
        Kow1 = Kow
    K = rho_lip/rho_wat * compP(paras, Kow1)
    return K
        
        
def compSSE_lg10P(paras, data, disp=False):
    ''' Function to calculate the SSE (sum of square error) of predicted and experimental P in log10 scale
    given <paras>, <data>; <data> is a numpy 2-d array with two columns: [Kow, P], where P is the MASS partition coefficient between lipid and water
    '''
    
    Kow = data[:,0]
    P_data_lg10 = np.log10( data[:,1] )
    P_pred_lg10 = np.log10( compP(paras, Kow) )
    
    err = P_data_lg10 - P_pred_lg10
    sse =  np.sum( np.square(err) )

    if (disp):        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)        
        ax.plot( P_data_lg10, P_pred_lg10, 'ro' )
        mmin = np.min([ax.get_xlim(), ax.get_ylim()])        
        mmax = np.max([ax.get_xlim(), ax.get_ylim()])
        ax.plot([mmin, mmax], [mmin, mmax], ls='--')       
        plt.show()

    return sse        

    
def compNLH_lg10P(paras, data, sig2=1, retSig2=False):
    ''' Function to calculate the negative log likelihood of predicted and experimental P in log10 scale
    given <paras>, <data>, and [sig2]; <data> is a numpy 2-d array with two columns: [Kow, P], where P is the MASS partition coefficient between lipid and water
    Args:
        retSig2 -- default is False, thus return the negative log likelihood; if True then return calculated variance
    '''
    n_dat = data.shape[0]

    sse = compSSE_lg10P(paras, data)
    likelihood = -0.5*n_dat*np.log(sig2) - 0.5*n_dat*np.log(2*np.pi) - 0.5*sse/sig2
    nlh = -likelihood
        
    if (retSig2):
        return sse/n_dat
    else:
        return nlh

        
### Functions calculating diffusion coefficients ###

def compD(paras, MW):
    ''' Function to calculate the diffusion coefficient in lipid
    D = a * exp(-b * r^2) where r can be calculated from MW
    Args:
        paras: ln of [a, b]
    '''
    a_ln = paras[0]
    b = np.exp(paras[1])
    r = np.power( 0.91*MW/4*3/np.pi, 1.0/3 )

    D_ln = a_ln - b*r*r
    D = np.exp(D_ln)
    return D
    
def compD_ln(paras, MW):
    ''' Function to calculate natural log of compD
    Args:
        paras: ln of [a, b]
    '''
    return np.log(compD(paras, MW))
    
def compSSE_lnD(paras, data, disp=False):
    ''' Function to calculate the SSE (sum of square error) of predicted and experimental D in log10 scale
    given <paras>, <data>; <data> is a numpy 2-d array with two columns: [MW, D]
    '''
    MW = data[:,0]
    D_data_lg10 = np.log10( data[:,1] )
    D_pred_lg10 = np.log10( compD(paras, MW) )

    err = D_data_lg10 - D_pred_lg10
    sse =  np.sum( np.square(err) )

    if (disp):        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)        
        ax.plot( D_data_lg10, D_pred_lg10, 'ro' )
        mmin = np.min([ax.get_xlim(), ax.get_ylim()])        
        mmax = np.max([ax.get_xlim(), ax.get_ylim()])
        ax.plot([mmin, mmax], [mmin, mmax], ls='--')
        plt.show()

    return sse  
   
    
def compNLH_lnD(paras, data, sig2=1, retSig2=False):
    ''' Function to calculate the negative log likelihood of predicted and experimental D in ln scale
    given <paras>, <data>, and [sig2]; <data> is a numpy 2-d array with two columns: [MW, D]
    Args:    
        retSig2 -- default is False, thus return the negative log likelihood; if True then return calculated variance
    '''
    n_dat = data.shape[0]

    sse = compSSE_lnD(paras, data)
    likelihood = -0.5*n_dat*np.log(sig2) - 0.5*n_dat*np.log(2*np.pi) - 0.5*sse/sig2
    nlh = -likelihood
        
    if (retSig2):
        return sse/n_dat
    else:
        return nlh

