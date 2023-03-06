#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 11:39:07 2021

@author: bendeacon
"""

import os, sys
import numpy as np
from scipy.optimize import minimize, basinhopping, brute
import matplotlib.pyplot as plt
from importlib import reload
from pandas import read_csv

from multiprocessing import Pool
from sys import exit
 #from 'example' import runDPK_invitro

#dpk = compDPK(fn_conf, chem=chem_list[i], sc_Kw_paras=None, sc_D_paras=D_paras)
model = read_csv('/Users/bendeacon/Documents/PhD/Year1/week51/Test_code/simu/MassFrac.csv', header=int(0))
model_2 = model.values
model_2 = model_2*100
model_2_values_accum = model_2[:,-1]
model_2_time_points = model_2_values_accum[[0,1,2,4,8,16,24]]

experiment =    read_csv('/Users/bendeacon/Documents/PhD/Year1/week51/Test_code/simu/propylparaben.csv', header=None) 
exp_2 = experiment.values
experiment_accum = exp_2[:, -1]

err = model_2_time_points - experiment_accum
sse =  np.sum( np.square(err) )