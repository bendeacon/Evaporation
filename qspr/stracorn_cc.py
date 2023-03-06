'''
Module for QSPR models in stratum corneum corneocyte (stracorn_cc).
The models are for given compound's partition coefficient between corneocyte and water,
and its diffusion coefficient in corneocyte

As a convention, the model parameters, i.e. regression coefficients, should be in natural log
    This is needed for parameter estimation when constraints on positive parameters can be avoided.
We follow "Int. J. Pharm 398 (2010) 114" to use "K" for volumetric partition 
    coefficient, and "P" for mass partition coefficient. Volumetric partition coefficient
    is what is used in the diffusion-based model, but mass partition coefficient is
    usually measured in experiments.
'''

import numpy as np
import matplotlib.pyplot as plt

from qspr.constants import rho_pro, rho_wat

### Functions calculating partition coefficients ###

def compK(paras, Kow):
    ''' Function to predict the volumetric partition coefficient between corneocyte and water
    P = a * Kow^b
    '''

    a = np.exp(paras[0])
    b = np.exp(paras[1])
    if len( Kow.shape ) > 1:
        Kow1 = Kow.flatten()
    else:
        Kow1 = Kow
    K = rho_pro/rho_wat * a * (Kow1**b)
    return K
    