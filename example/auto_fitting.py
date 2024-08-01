#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 16:32:09 2021

@author: bendeacon
"""
import os, sys
import numpy as np
from scipy.optimize import minimize, basinhopping, brute
import matplotlib.pyplot as plt
from importlib import reload
from pandas import read_csv


# Using multiple processers
N_PROCESS = 8

# Import various skin 
from core import chemical
reload(chemical)
from core import config
reload(config)
from core import vehicle
reload(vehicle)
from core import viaepd
reload(viaepd)
from core import dermis
reload(dermis)
from core import skin_setup
reload(skin_setup)


def compSSE_DPK() :
    """Compute the objective function for optimisation
    The objective function is to minimise the sum of square error between model prediction and data
    Args:
        paras - [alpha, beta] as in diffusivity QSPR in corneocyte
        fn_conf - name of configuration function, only using the geometric and initial condition settings
        chem_list - a list of Chemical objects
        perm_list - a list of permeability measurements
    """
    #run dpk and get the data out
    #import experimental data
    #i need to work on getting the .csv files in the correct format automatically
    from multiprocessing import Pool
    from sys import exit
    #from 'example' import runDPK_invitro
    
    #dpk = compDPK(fn_conf, sc_Kw_paras=paras0[0], sc_D_paras=paras0[1])
    model = read_csv('/Users/bendeacon/Documents/PhD/Year1/week51/Test_code/simu/MassFrac.csv', header=int(0))
    model_2 = model.values
    model_2 = model_2*100
    model_2_values_accum = model_2[:,-1]
    model_2_time_points = model_2_values_accum[[0,1,2,4,8,16,24]]
    
    experiment =    read_csv('/Users/bendeacon/Documents/PhD/Year1/week51/Test_code/simu/Ibuprofen.csv', header=None) 
    exp_2 = experiment.values
    experiment_accum = exp_2[:, -1]
    
    err = model_2_time_points - experiment_accum
    sse =  np.sum( np.square(err) )
    return sse


def calibDPK():
    """Calibration of permeability model by adjusting parameters
    """
    #input the experimental data
    fn_conf = 'example/config/Ibuprofen_CE_SCHomo2.cfg'
    
    paras0 = [22.5, 3.85e-13] # [Ksc/w, Dsc] as in SC diffusion
    
    #bnds = ((-0.3, 0), (50, 200))
    bnds = None
    if bnds == None:
        res = minimize(compSSE_DPK, paras0, args=(fn_conf), \
        method='BFGS', options={'disp': True, 'maxiter': 100})
    else:
        res = minimize(compSSE_DPK, paras0, args=(fn_conf, _chem_list, Kp_lg10), \
        method='L-BFGS-B', bounds=bnds, options={'disp': True, 'maxiter': 100})
                       
    return res.x
