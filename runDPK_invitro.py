# -*- coding: utf-8 -*-
"""
A module containing files for calculating kinetics
    for in-vitro experiments
"""
import os, sys
import numpy as np
#from scipy.optimize import minimize, basinhopping, brute
import matplotlib.pyplot as plt
from importlib import reload


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

def compDPK(fn_conf, chem=None, sc_Kw_paras=None, sc_D_paras=None, disp=1, wk_path='./simu/') :
    """Compute DPK
    Args:
        fn_conf -- the .cfg file, which gives the configuration of the simulation
        chem -- if given, it overrides the values given in fn_conf
        wk_path -- path to save simulation results
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
        
        mass = _skin.compMass_comps()
        m_v = _skin.comps[0].getMass_OutEvap()
        m_all = np.insert(mass, 0, m_v) / total_mass
        
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
    t_start, t_end, Nsteps = [0, 3600*24, 25]
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
        
        
    
