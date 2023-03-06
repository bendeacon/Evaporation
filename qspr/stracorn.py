'''
Module for QSPR models in stratum corneum (stracorn) which contains two phases: lipid (lp) and corneocyte (cc)
The models are for given compound's partition coefficient between stratum corneum and water,
and its (effective) diffusion coefficient in stratum corneum

As a convention, the model parameters, i.e. regression coefficients, should be in natural log
    This is needed for parameter estimation when constraints on positive parameters can be avoided.
We follow "Int. J. Pharm 398 (2010) 114" to use "K" for volumetric partition 
    coefficient, and "P" for mass partition coefficient. Volumetric partition coefficient
    is what is used in the diffusion-based model, but mass partition coefficient is
    usually measured in experiments.
'''

import numpy as np
import matplotlib.pyplot as plt

from qspr.constants import rho_pro, rho_lip, rho_wat

def compVolFrac(ref_source='Liming'):
    ''' Compute the volume fractions of protein, lipid and water in SC
    There are conflicting literature data and that's why this function is here
    '''

    if ref_source == 'Longjian': # Ind. Eng. Chem. Res. 2008, 47, 6465–6472
        w_pro = 0.45*0.875 # water content 55% w/w, thus dry mass is 45%, of which protein is 87.5%
        w_lip = 0.45*0.125 # water content 55% w/w, thus dry mass is 45%, of which lipid is 12.5%
        w_wat = 0.55
    elif ref_source == 'Liming': #International Journal of Pharmaceutics 398 (2010) 114–122
        w_pro = 0.77  # 77% protein in dry mass  
        w_lip = 0.23  # 23% lipid in dry mass
        w_wat = 2.99  # grams of water absorbed by per gram of dry SC 
    elif ref_source == 'Average': # see footnote b of Table 6 of the above Liming Wang's paper
        w_pro = 0.90
        w_lip = 0.10
        w_wat = 2.84
    elif ref_source == 'Raykar':  # see footnote b of Table 6 of the above Liming Wang's paper
        w_pro = 0.84
        w_lip = 0.16
        w_wat = 2.91
    else:
        raise ValueError('Invalid ref_source')        
    
    v_pro = w_pro/rho_pro    
    v_lip = w_lip/rho_lip    
    v_wat = w_wat/rho_wat

    v_total = v_pro + v_lip + v_wat
    phi_pro = v_pro / v_total
    phi_lip = v_lip / v_total
    phi_wat = v_wat / v_total

    return (phi_pro, phi_lip, phi_wat)
    
### Functions calculating partition coefficients ###

def compK(paras, Kow):
    ''' Function to predict the volumetric partition coefficient between stratum corneum and water    
    '''

    a = np.exp(paras[0])
    b = np.exp(paras[1])
    c = np.exp(paras[2])

    phi_pro, phi_lip, phi_wat = compVolFrac()
    K = phi_pro*rho_pro/rho_wat* a*(Kow**b) + phi_lip*rho_lip/rho_wat*(Kow**c) + phi_wat        
    return K
    

def compK_fixedLP(paras, Kow, c):
    ''' Function to predict the volumetric partition coefficient between stratum corneum and water
    with fixed lipid term, i.e. c is fixed   
    '''    
    paras1 = np.concatenate((paras, c))
    return compK(paras1, Kow)
    
    
def compK_from_cc_lp(paras, Kow, K_cc_lp):
    ''' Function to predict the volumetric partition coefficient between stratum corneum and water
    from given volumetric partition coefficients of corneocyte:water and lipid:water
    Args:        
        K_cc_lp contains [Kcc, Klp]
    '''

    Kcc = K_cc_lp[:,0]
    Klp = K_cc_lp[:,1]
    
    phi_pro, phi_lip, phi_wat = compVolFrac()
    
    K = phi_pro*Kcc + phi_lip*Klp + phi_wat
    return K
    

def compSSE_lg10K(paras, data, c=None, disp=False):
    ''' Function to calculate the SSE (sum of square error) of predicted and experimental K in log10 scale
    given <paras>, data>; <data> is a numpy 2-d array with two columns: [Kow, K], where K is the volumetric partition coefficient between stratum corneum and water
    '''
    Kow = data[:,0]
    K_data_lg10 = np.log10( data[:,1] )
    if c is None:
        K_pred_lg10 = np.log10( compK(paras, Kow) )
    else:
        K_pred_lg10 = np.log10( compK_fixedLP(paras, Kow, c) )
    
    err = K_data_lg10 - K_pred_lg10
    sse =  np.sum( np.square(err) )

    if (disp):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)        
        ax.plot( K_data_lg10, K_pred_lg10, 'ro' )
        mmin = np.min([ax.get_xlim(), ax.get_ylim()])        
        mmax = np.max([ax.get_xlim(), ax.get_ylim()])
        ax.plot([mmin, mmax], [mmin, mmax], ls='--')       
        plt.show()      

    return sse
    

def compNLH_lg10K(paras, data, c=None, sig2=1, retSig2=False):
    ''' Function to calculate the negative log likelihood of predicted and experimental K in log10 scale
    given <paras>, data>; <data> is a numpy 2-d array with two columns: [Kow, K], where K is the volumetric partition coefficient between stratum corneum and water
    Note that the data given in lg10Ksc is the VOLUMETRIC partition coefficient between stratum corneum and water
    Args:
        retSig2 -- default is False, thus return the negative log likelihood; if True then return calculated variance
    '''
    n_dat = data.shape[0]

    sse = compSSE_lg10K(paras, data, c)
    likelihood = -0.5*n_dat*np.log(sig2) - 0.5*n_dat*np.log(2*np.pi) - 0.5*sse/sig2
    nlh = -likelihood
        
    if (retSig2):
        return sse/n_dat
    else:
        return nlh
