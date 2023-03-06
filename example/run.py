# -*- coding: utf-8 -*-
"""
An example script to simulate dermal absorption
with configuration defined in ../config/Nicotine.cfg
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload


# Import various skin 
from core import chemical
reload(chemical)
from core import config
reload(config)
from core import viaepd
reload(viaepd)
from core import dermis
reload(dermis)
from core import skin_setup
reload(skin_setup)


# Read the .cfg, i.e. configuration, file to set up simulation
#_conf = config.Config('Caffeine.cfg')
_conf = config.Config('./config/Nicotine.cfg')

# Setup the chemical
_chem = chemical.Chemical(_conf)

# Setup skin and create compartments
_skin = skin_setup.Skin_Setup(_chem, _conf)
_skin.createComps(_chem, _conf)

# Simulation time (in seconds) and steps
t_start, t_end, Nsteps = [0, 36000, 61]
t_range = np.linspace(t_start, t_end, Nsteps)

# total mass in each compartment
nComps = _skin.nxComp*_skin.nyComp
mass = np.zeros([Nsteps, nComps])

b_first = True

newpath = './simu'
if not os.path.exists(newpath):
    os.makedirs(newpath)



for i in range(Nsteps):
    
    # Create directory to save results
    newpath = './simu/' + str(t_range[i])
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        
    # Save current concentrations
    for j in range(nComps):
        fn = newpath + '/comp' + str(j) + '_' + _conf.comps_geom[j].name        
        _skin.comps[j].saveMeshConc(True, fn)
        
    mass[i,:] = _skin.compMass_comps()
    print('Time = ', t_range[i], 'Flux vh_sc= ', '{:.3e}'.format(_skin.compFlux([0,0], 3)[0]), \
       'Flux sc_down=', '{:.3e}'.format(_skin.compFlux([1,0], 3)[0]) )
    
    if i == Nsteps-1:
        break
    
    # Simulate
    _skin.solveMoL(t_range[i], t_range[i+1])
    

plt.plot(t_range/60, mass[:,0]/mass[0,0], \
         t_range/60, mass[:,1]/mass[0,0], \
         t_range/60, mass[:,2]/mass[0,0], \
         t_range/60, mass[:,3]/mass[0,0])
plt.show()
