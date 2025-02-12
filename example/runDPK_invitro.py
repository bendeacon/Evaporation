# -*- coding: utf-8 -*-
"""
A module containing files for calculating kinetics
    for in-vitro experiments
"""
import os, sys
import numpy as np
from scipy.optimize import minimize, basinhopping, brute
import matplotlib.pyplot as plt
from importlib import reload
from pandas import read_csv
import time 
import scipy as sp
import numpy as np
from scipy import optimize
import math
import shutil
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

global _conf


def compDPK(fn_conf, chem=None, sc_Kw_paras=None, sc_D_paras=None, disp=1, wk_path='./simu/') :
    """Compute DPK
    Args:
        fn_conf -- the .cfg file, which gives the configuration of the simulation
        chem -- if given, it overrides the values given in fn_conf
        wk_path -- path to save simulation results
    """
    # Read the .cfg, i.e. configuration, file to set up simulation
    #print(sc_Kw_paras, sc_D_paras)
 
    b_vary_vehicle = True  #True
    b_inf_source = False
 
    _conf = config.Config(fn_conf)
    if sc_Kw_paras is not None:
        _conf.Kw_sc_paras = sc_Kw_paras  
    #print(_conf.Kw_sc_paras)
    #print(type(_conf.Kw_sc_paras))           
    if sc_D_paras is not None:
        _conf.D_sc_paras = sc_D_paras  
    #print(_conf)
    # Setup the chemical
    if chem is not None:
        _chem = chem
        #print(chem)
    else:
        _chem = chemical.Chemical(_conf)
    #print(_conf.D_sc_paras) 
    #print(_conf.Kw_sc_paras)
    

    # Setup skin and create compartments
    _skin = skin_setup.Skin_Setup(_chem, _conf)
    _skin.createComps(_chem, _conf)     

    # Simulation time (in seconds) and steps
    t_start, t_end, Nsteps = [0, 3600*24, 25]
    #t_start, t_end, Nsteps = [0, 1800, 181]
    t_range = np.linspace(t_start, t_end, Nsteps)  
    # t_range = np.r_[np.linspace(0, 1000, 2), np.linspace(1200, 1400, 201)]
    Nsteps = len(t_range)
    
    nComps = _skin.nxComp*_skin.nyComp
    total_mass = np.sum( _skin.compMass_comps() )
    
    # Create directory to save results
    newpath = wk_path
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    fn = wk_path + 'MassFrac.csv'
    saveMass(total_mass, fn, b_1st_time=True)    
    
    for i in range(Nsteps):
        
        if b_inf_source is True:
            mass = _skin.compMass_comps()
            m_v = _skin.comps[0].getMass_OutEvap()
            m_all = np.insert(mass, 0, m_v) / total_mass
        else:
            if b_vary_vehicle is True:
                mass = _skin.compMass_comps()
                m_v = _skin.comps[0].getMass_OutEvap()
                mass_evap = np.insert(mass, 0, m_v)
                total_mass = np.sum(mass_evap)
                m_all = np.insert(mass, 0, m_v) / total_mass
            else:
                mass = _skin.compMass_comps()
                m_v = _skin.comps[0].getMass_OutEvap()
                m_all = np.insert(mass, 0, m_v) / total_mass
        #print(mass, m_v, m_all, total_mass)
        
        if disp >= 2:
            np.set_printoptions(precision=2)
            print('Time = ', t_range[i], '% mass: ', m_all)            
            
        # Create directory to save results
        newpath = wk_path + str(t_range[i])
        if not os.path.exists(newpath):
            os.makedirs(newpath)
            
        # Save fraction of mass in all compartments
        fn = wk_path + 'MassFrac.csv'
        saveMass(np.insert(m_all, 0, t_range[i]), fn)
        
        # Save current concentrations        
        for j in range(nComps):
            fn = newpath + '/comp' + str(j) + '_' + _conf.comps_geom[j].name        
            _skin.comps[j].saveMeshConc(True, fn)
        
        if i == Nsteps-1:
            break
        
        # Simulate
        _skin.solveMoL(t_range[i], t_range[i+1])    
    
    #return mass

def compDPK_multiChem(fn_conf, disp=1, wk_path='./simu/') :
    """Compute DPK for multiple chemicals which all penetrate into skin
    Args:
        fn_conf -- the .cfg file, which gives the configuration of the simulation
        chem -- if given, it overrides the values given in fn_conf
        wk_path -- path to save simulation results
    """
    # Read the .cfg, i.e. configuration, file to set up simulation
    _conf = config.Config(fn_conf)
    nChem = _conf.nChem
    if nChem > 1:
        _chem = [chemical.Chemical() for i in range(nChem)]
        _conf_chem = [config.Config() for i in range(nChem)]
    else:
        raise ValueError('This function should only be used with multiple chemicals co-penetrating')
        
    # set up _chem[]
    for i in range(nChem):
        fn = fn_conf + '.chem' + str(i)
        _conf_chem[i].readFile(fn)
        _conf_chem[i].combine(_conf)
        _chem[i].setChemConf(_conf_chem[i])
        
    # set up skin objects, one for each chemical species
    _skin = [skin_setup.Skin_Setup(_chem[i], _conf_chem[i]) for i in range(nChem)]
    total_mass = np.zeros(nChem) # to store the total mass of each chemical
    for i in range(nChem):        
        _skin[i].createComps(_chem[i], _conf_chem[i])    
        total_mass[i] = np.sum( _skin[i].compMass_comps() )

    # Simulation time (in seconds) and steps
    t_start, t_end, Nsteps = [0, 3600*50, 51]
    #t_start, t_end, Nsteps = [0, 1800, 181]
    t_range = np.linspace(t_start, t_end, Nsteps)  
    # t_range = np.r_[np.linspace(0, 1000, 2), np.linspace(1200, 1400, 201)]
    Nsteps = len(t_range)    
    
    
    # Create directory to save results
    newpath = wk_path
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    for i in range(nChem):
        fn = wk_path + 'MassFrac.csv' + 'chem' + str(i)
        saveMass(total_mass[i], fn, b_1st_time=True)    
    
    nComps = _skin[0].nxComp*_skin[0].nyComp
    
    for j in range(Nsteps):
        
        if disp >= 2:
            print('Time = \n', t_range[j])

        # At each time step, update the partition coefficient due to co-penetration
        # todo: test code now
        multiplier = .0
        _skin[0].updateKw_fromOther(_conf_chem[0], _skin[1], multiplier)
            
        # Create directory to save results
        newpath = wk_path + str(t_range[j])
        if not os.path.exists(newpath):
            os.makedirs(newpath)
            
        # Save results
        for i in range(nChem):
        
            mass = _skin[i].compMass_comps()
            m_v = _skin[i].comps[0].getMass_OutEvap()
            m_all = np.insert(mass, 0, m_v) / total_mass[i]
        
            if disp >= 2:
                np.set_printoptions(precision=2)
                print('\tChem no. ', i, '% mass: ', m_all)            
            
            # Save fraction of mass in all compartments
            fn = wk_path + 'MassFrac.csv' + 'chem' + str(i)
            saveMass(np.insert(m_all, 0, t_range[j]), fn)
        
            # Save current concentrations        
            for k in range(nComps):
                fn = newpath + '/comp' + str(k) + '_' + _conf.comps_geom[k].name        
                _skin[i].comps[k].saveMeshConc(True, fn)
                
        # Conduct simulation                
        for i in range(nChem):
            if j == Nsteps-1:
                break        
            # Simulate
            _skin[i].solveMoL(t_range[j], t_range[j+1])
    
    #return mass
    
    
def saveMass(nparray, fn, b_1st_time=False) :
        """ Save mass and fractions to file
        Args: 
            nparray -- the data to be saved
            b_1st_time -- if True, write to a new file; otherwise append to the existing file
        """
        if b_1st_time :
            file = open(fn, 'w')
        else :
            file = open(fn, 'a')
        
        if type(nparray) is np.ndarray:
            nd = len(nparray)
            for i in range(nd):
                file.write("{:.6e}".format(nparray[i]))            
                if i<nd-1:
                    file.write(",")
                #print("nparray", file)
        else:
            file.write( "{:.6e}".format(nparray) )
        file.write('\n')
        file.close()

        
def compDPK_KwVar(fn_conf, wk_path='./simu/', N=50, N_PROCESS=1) :        
    """ Simulate with the uncertainty in Kw for lipid and corneocytes 
        Args:
            N - number of MC samples
    """
    _conf = config.Config(fn_conf)
    _chem = chemical.Chemical(_conf)        
        
    # Run uncertainty quantification 
    import example.runUnctQSPR as rUq
    reload(rUq)
    #print(np.array([_chem.mw]))
    Kw_var, Kw_base = rUq.compUnct_Ksc(np.array([[_chem.K_ow]]))
    print('Base case, Klp= ', 10**Kw_base[0][0][1], ' Kcc= ', 10**Kw_base[0][0][0])
    #print(Kw_var[0][0])
    m_cc, m_lp = Kw_var[0][0]
    print('Re-calibrated, Klp= ', 10**m_lp, ' Kcc= ', 10**m_cc)
    sd_cc, sd_lp = np.sqrt( np.diag(Kw_var[1][0]) ) 
    
    # Set up KwDParas to be passed to simulation
    from core.stracorn import KwDParas
    
    from multiprocessing import Pool    
    if N_PROCESS == 1 :
        sc_Kw_paras = KwDParas()
        sc_Kw_paras.lp.option = 'VALE'
        sc_Kw_paras.cc.option = 'VALE'
        for i in range(N):  
            sc_Kw_paras.lp.value = np.array( [10**np.random.normal(m_lp, sd_lp)] )
            sc_Kw_paras.cc.value = np.array( [10**np.random.normal(m_cc, sd_cc)] )
            wk_path_i = wk_path + 'rep_' + str(i) + '/'
            print('\t Rep ', i, 'Klp= ', sc_Kw_paras.lp.value, ' Kcc= ', sc_Kw_paras.cc.value, '\n')
            compDPK(fn_conf, _chem, sc_Kw_paras, disp=3, wk_path=wk_path_i)
    else:
        raise ValueError('N_PROCESS must be 1; there is a bug in multiprocessing waiting to be fixed!')
        arg_list = [None]*N
        sc_Kw_paras = [ KwDParas() for i in range(N) ]
        for i in range(N):
            sc_Kw_paras[i].lp.value = np.array( [10**np.random.normal(m_lp, sd_lp)] )
            sc_Kw_paras[i].cc.value = np.array( [10**np.random.normal(m_cc, sd_cc)] )
            wk_path_i = wk_path + 'rep_' + str(i) + '/'
            print('\t Rep ', i, 'Klp= ', sc_Kw_paras[i].lp.value, ' Kcc= ', sc_Kw_paras[i].cc.value, '\n')
            arg_list[i] = (fn_conf, _chem, sc_Kw_paras[i], None, 3, wk_path_i) 
        
        with Pool(N_PROCESS) as pool:
            pool.starmap(compDPK, arg_list)

def compSSE_DPK(paras, fn_conf) :
    """Compute the objective function for optimisation
    The objective function is to minimise the sum of square error between model prediction and data
    Args:
        paras - [Ksc/w, Dsc] as in SC diffusion
        fn_conf - name of configuration function, only using the geometric and initial condition settings
    """
  #run dpk and get the data out
    from multiprocessing import Pool
    from sys import exit
    K_paras = 10**(float(paras[0]))
    D_paras = 10**(float(paras[1]))
    #Ve_K_paras = 10**(float(paras[2]))
    #Ve_D_paras = 10**(float(paras[3]))
    print(K_paras,D_paras)
    
    #run compdpk file
    file_input_change(paras)
    #convert back for running the code
    compDPK(fn_conf, None, K_paras, D_paras)
    model = read_csv('/Users/bendeacon/Documents/PhD/Year4/week2/Evaporation_Surrey_model/simu/MassFrac.csv', header=int(0))
    model_2 = model.values
    model_2 = model_2*100
    model_2_values_accum = model_2[:,-1]
    model_2_time_points = model_2_values_accum[[0,1,2,4,8,16,24]]
    
    experiment = read_csv('/Users/bendeacon/Documents/PhD/Year4/week2/Evaporation_Surrey_model/simu/Dimethyl_fumarate.csv', header=None) 
    exp_2 = experiment.values
    experiment_accum = exp_2[:, -1]
    print(model_2_time_points)
    print(experiment_accum)
    err = model_2_time_points - experiment_accum
    sse =  np.sum( np.square(err) )
    #print(sse)
    
    return sse

def calibDPK():
    """Calibration of permeability model by adjusting parameters
    """
    #input the experimental data
    
    fn_conf = 'example/config/Fitting_Perm/Dimethylfumarate_CE_2.cfg'
    #logparameter 
    paras0 = [0.01687, 3.45E-13] #, 1, 1.30791E-10]# [Ksc/w, Dsc, Kve/w, Dve] as in SC diffusion
    
    bounds = [(0, 5000), (1e-25, 1e-5)]#,(0.001,10), (1e-19, 1e-9)]

    
    res = minimize(compSSE_DPK, paras0, args=(fn_conf), \
        method='Nelder-Mead', bounds = bounds, tol=0.1, options={ 'disp': True, 'maxiter': 800})
       
    return res.x

# =============================================================================
# def calibDPK_Multi_Start():
#     """Calibration of permeability model by adjusting parameters
#     This starts from multiple positions
#     """
#     #input the experimental data
#     chemical = 'Propylparaben'
#     fn_conf = 'example/config/%s_CE_SCHomo_2.cfg' %chemical
#     #x0 = [4.47, 3.28e-13, 5, 3.28e-11] # [Ksc/w, Dsc, Kve/w, Dve] as in SC diffusion
#     N = 30
#     res_list = sp.empty(N, dtype = object)
#     res_list_2 = sp.empty(N, dtype = object)
#     logkow = 3.04
#     #x0min = 0.054*logkow**(2)+0.1551*logkow-0.2921 #standard bound
#     #x0max = 0.0752*logkow**(2)+0.3408*logkow+0.4286 #standard bound
#     #z0min = 0.0423*logkow**(2)+0.0427*logkow-0.4612 #standard bound
#     #z0max = 0.0795*logkow**(2)+0.0175*logkow+0.0704 #standard bound
#     x0min = 0.054*logkow**(2)+0.1551*logkow-0.86073 #bound includes rmse
#     x0max = 0.0752*logkow**(2)+0.3408*logkow+0.997236 #bound includes rmse
#     #z0min = 0.0423*logkow**(2)+0.0427*logkow-0.5612 #bound includes rmse
#     #z0max = 0.0795*logkow**(2)+0.0175*logkow+0.1704 #bound includes rmse
#     y0min = -19
#     y0max = -8.69897
#     #w0min = -19
#     #w0max = -8.69897
#     bounds = [(x0min, x0max), (y0min, y0max)]#(z0min, z0max), (w0min, w0max)]
#     for i in range(N):
#         x0 = [sp.random.uniform(x0min, x0max),sp.random.uniform(y0min, y0max)] #, sp.random.uniform(z0min, z0max),  sp.random.uniform(w0min, w0max)] # [Ksc/w, Dsc, Kve/w, Dve] as in SC diffusion
#         # for i in range(N):
#         #     x0 = [sp.random.uniform(x0min, x0max),-11.6317936, sp.random.uniform(z0min, z0max), -9.106606428] # [Ksc/w, Dsc, Kve/w, Dve] as in SC diffusion
#         #     for i in range(N):
#         #           x0 = [sp.random.uniform(x0min, x0max),sp.random.uniform(y0min, y0max), sp.random.uniform(z0min, z0max), -9.106606428] # [Ksc/w, Dsc, Kve/w, Dve] as in SC diffusion
#         #           for i in range(N):
#         #               x0 = [sp.random.uniform(x0min, x0max),sp.random.uniform(y0min, y0max), sp.random.uniform(z0min, z0max),  sp.random.uniform(w0min, w0max)] # [Ksc/w, Dsc, Kve/w, Dve] as in SC diffusion
#         res = minimize(compSSE_DPK, x0, args=(fn_conf), method='Nelder-Mead', jac=None, \
#              hess=None, hessp=None, bounds=bounds, constraints=None, tol=1, callback=None, options={ 'disp': True, 'maxiter': 800})
#         res_list[i] = res
#         res_list_2[i] = res.x
#     for i in res_list_2:
#         paras = i
#         original = r'/Users/bendeacon/Documents/PhD/Year2/week17/Test_code/simu/MassFrac.csv'
#         new = r'/Users/bendeacon/Documents/PhD/Year2/week17/Test_code/graph_data/%s.csv' %i
#         file_input_change(paras)
#         fn_conf = 'example/config/%s_CE_SCHomo_2.cfg'%chemical
#         compDPK(fn_conf, None)
#         shutil.copy(original,new)
#         
#         
#         
#     #print(res_list)
#     #sort_res_list = res_list[sp.argsort([res.fun for res in res_list])]
#     return res_list
# =============================================================================
 

def file_input_change(paras):
    """This is used to alter the input file to make sure the K and D parameters are correct!
    input taken is the parameters"""
    os.remove("/Users/bendeacon/Documents/PhD/Year4/week2/Evaporation_Surrey_model/example/config/Fitting_Perm/Dimethylfumarate_CE_2.cfg")
    K_paras = (float(paras[0]))
    D_paras = (float(paras[1]))
    #Ve_K_paras = 10**(float(paras[2]))
    #Ve_D_paras = 10**(float(paras[3]))
    #print(K_paras,D_paras)
    
    f_new = open("/Users/bendeacon/Documents/PhD/Year4/week2/Evaporation_Surrey_model/example/config/Fitting_Perm/Dimethylfumarate_CE_2.cfg", "a")
    f = open("/Users/bendeacon/Documents/PhD/Year4/week2/Evaporation_Surrey_model/example/config/Fitting_Perm/Dimethylfumarate_CE.cfg")
    content = f.readlines()
    content = "".join(content)
    f.close()
    lines = f"\nKW_SC           {K_paras} \nD_SC            {D_paras}"
    #lines_2 = f"\nKW_VE           {Ve_K_paras} \nD_VE            {Ve_D_paras}"
    lines = str(lines)
    #lines_2 = str(lines_2)
    f_new.write(content)
    f_new.write(lines)
    #f_new.write(lines_2)
   
    f_new.close()
