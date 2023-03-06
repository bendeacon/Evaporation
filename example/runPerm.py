# -*- coding: utf-8 -*-
"""
A module containing files for calculating steady-state permeability
    and related optimisation for parameter estimation
"""
#import pdb
import os
import numpy as np
from scipy.optimize import minimize, basinhopping, brute
import matplotlib.pyplot as plt
from importlib import reload

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

def compPerm(fn_conf, chem=None, sc_Kw_paras=None, sc_D_paras=None, disp=1) :
    """Compute steady state permeability
    Args:
        fn_conf -- the .cfg file, which gives the configuration of the simulation
        chem -- if given, it overrides the values given in fn_conf
        sc_Kw_paras -- if given, overrides the QSPR parameters to calculate Kw in stratum corneum
        sc_D_paras -- if given, overrides the QSPR parameters to calculate D in stratum corneum
    """
    # Read the .cfg, i.e. configuration, file to set up simulation
    _conf = config.Config(fn_conf)
    if sc_Kw_paras is not None:
        _conf.Kw_sc_paras = sc_Kw_paras
    if sc_D_paras is not None:
        _conf.D_sc_paras = sc_D_paras
    
 
    # Setup the chemical
    if chem is not None:
        _chem = chem
    else:
        _chem = chemical.Chemical(_conf)
    # Setup skin and create compartments
    _skin = skin_setup.Skin_Setup(_chem, _conf)
    _skin.createComps(_chem, _conf)

    # Simulation time (in seconds) and steps
    #t_start, t_end, Nsteps = [0, 3600*48, 101]
    #t_start, t_end, Nsteps = [0, 60, 3]
    t_start, t_end, Nsteps = [0, 3600*48, 100]
    t_range = np.linspace(t_start, t_end, Nsteps)    
    for i in range(Nsteps-1):
        flux_vh_sc = -_skin.compFlux([0,0], 3)[0]
        flux_sc_down = -_skin.compFlux([1,0], 3)[0]
        if np.fabs( (flux_vh_sc-flux_sc_down) / flux_vh_sc ) < 1e-5 :
            break
        #elif i == Nsteps-1 :
        #    raise ValueError('Simulation time too short to reach steady-state; re-run the simulation with longer time.')
        
        if disp >= 2:
            print('Time = ', t_range[i], 'Flux vh_sc= ', '{:.3e}'.format(flux_vh_sc), \
                  'Flux sc_down=', '{:.3e}'.format(flux_sc_down), )
        
        print( 'Vehicle conc =', _skin.getComp(0,0).getMeshConc() ) 
        # Simulate
        _skin.solveMoL(t_range[i], t_range[i+1])
        
    if disp >= 1:
        print('MW= ', '{:.2f}'.format(_chem.mw), 'Flux vh_sc= ', '{:.3e}'.format(flux_vh_sc), \
                  'Flux sc_down=', '{:.3e}'.format(flux_sc_down))
        
    return flux_vh_sc / _conf.init_conc_vh

def compSSE_Perm(paras, fn_conf, chem_list, perm_list) :
    """Compute the objective function for optimisation
    The objective function is to minimise the sum of square error between model prediction and data
    Args:
        paras - [alpha, beta] as in diffusivity QSPR in corneocyte
        fn_conf - name of configuration function, only using the geometric and initial condition settings
        chem_list - a list of Chemical objects
        perm_list - a list of permeability measurements
    """
    from multiprocessing import Pool
    from sys import exit
    
    n_dat = len(chem_list)
    perm_lg10 = np.zeros((n_dat, 1))
    
    D_paras = np.concatenate( (np.exp(paras), [-1, -1]) )
    
    
    if N_PROCESS == 1 :
        for i in range(n_dat):
            perm = compPerm(fn_conf, chem=chem_list[i], sc_Kw_paras=None, sc_D_paras=D_paras)
            perm_lg10[i] = np.log10(perm)
    else :
        arg_list = [None]*n_dat
        for i in range(n_dat):
            arg_list[i] = (fn_conf, chem_list[i], None, D_paras)
        #print(arg_list)
        
        
        with Pool(N_PROCESS) as pool:
            perm = pool.starmap(compPerm, arg_list)
        #print(perm)
        #exit()
        perm_lg10 = np.log10(perm)
        
    err = perm_lg10 - np.array(perm_list)
    sse =  np.sum( np.square(err) )
    return sse


def calibPerm():
    """Calibration of permeability model by adjusting parameters
    """
    fn_conf = 'example/config/Nicotine.cfg'
    _conf = config.Config(fn_conf)
    
    Kow_lg10 = [-3.01, -1.38, -0.77, 1.61, 2.03, 3.05, 3.97, 4.57]
    Mw = [182.2, 18, 32, 362.5, 102.2, 144.2, 206.3, 158.3]
    Kp_lg10 = [-8.16, -6.32, -6.56, -7.48, -5.11, -5.16, -5.00, -4.30]
    #Kow_lg10 = [-3.01, -1.38]
    #Mw = [182.2, 18]
    #Kp_lg10 = [-8.16, -6.32]
    
    n_dat = len(Kp_lg10)
    _chem_list = [chemical.Chemical(_conf) for i in range(n_dat)]
    for i in range(n_dat):
        _chem_list[i].set_mw(Mw[i])
        _chem_list[i].set_K_ow(10**Kow_lg10[i])
    
    paras0 = np.log([9.47, 9.32e-8]) # [alpha, beta] as in corneocyte diffusivity calculation
    
    #bnds = ((-0.3, 0), (50, 200))
    bnds = None
    if bnds == None:
        res = minimize(compSSE_Perm, paras0, args=(fn_conf, _chem_list, Kp_lg10), \
        method='BFGS', options={'disp': True, 'maxiter': 100})
    else:
        res = minimize(compSSE_Perm, paras0, args=(fn_conf, _chem_list, Kp_lg10), \
        method='L-BFGS-B', bounds=bnds, options={'disp': True, 'maxiter': 100})
                       
    return res.x