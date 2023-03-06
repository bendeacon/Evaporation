# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 17:38:34 2017

@author: tc0008
"""

import importlib
import numpy as np

from core import comp
importlib.reload(comp)
from core import dermis
importlib.reload(dermis)

class Blood:
    """ """
    def __init__(self, frac_unbound, k_clear, body_mass=70, gender='M'):
        """ """
  
        # Gender dependent calculation of 
        #    -- cardiac output (flow rate)
        #    -- blood volume (Morgan, Mikhail, Murray. Clinical Anesthesiology. 3rd Edition.)
        #       Adult male: 75 mL/Kg, adult female: 65, infants: 80, neonates: 85, premature neonates: 96
        if gender == 'M' :
            cardiac_output = 5.6* 1e-3/60 # // 5.6 L/min, converted to m^3/s
            vol_blood_per_kg = 75* 1e-6 # // 75 mL/Kg, converted to m^3/Kg
        elif gender == 'F' :
            cardiac_output = 4.9* 1e-3/60 # // 4.9 L/min, converted to m^3/s
            vol_blood_per_kg = 65* 1e-6 # 65 mL/Kg, converted to m^3/Kg
        else :
            raise ValueError('Invalid gender')        
        frac_skin = 0.05 # fraction of skin blood flow as in total cardiac output
        self.flow_capil = frac_skin * cardiac_output

        # This is one way of calculation of blood with no regard of gender
        #   Now we switch to a gender-dependent calculation as above
        if False:
            frac_blood = 0.07 # fraction of blood as in body mass
            blood_density = 1060 # // kg / m^3
            self.vol_blood_body = frac_blood * body_mass / blood_density
        
        self.vol_blood_body = vol_blood_per_kg * body_mass

        self.conc = .0
        self.k_clear = k_clear
        self.f_unbound = frac_unbound

        self.conc_sink = .0
        self.vol_sink = 1.0 # Arbitrary reference value

        self.dim = 2 # including blood, and sink

    ### (START OF) Class methods dealing with ODE computation ###
    
    def updateMassInOutDermis(self, massIn, massOut, factor):
        """ This function will be called after calculations in dermis
        """
        self.mass_into_dermis = massIn*factor
        self.mass_outfrom_dermis = massOut*factor

    def compODEdydt (self, t, y, args=None):
        f = np.zeros(2)
        f[0] = (-self.mass_into_dermis + self.mass_outfrom_dermis)/self.vol_blood_body
        f[0] -= self.k_clear * y[0] / self.vol_blood_body
        f[1] = self.k_clear * y[0] / self.vol_cleared

    ### (END OF) Class methods dealing with ODE computation ###
    
    def setMeshConc_all(self, conc):
        self.conc = conc[0]
        self.conc_sink = conc[1]
    def getMeshConc_all(self):
        return np.array([self.conc, self.conc_sink])

    def saveConc(self, b_1st_time, fn) :
        """ Save concentrations to file
        Args: b_1st_time -- if True, write to a new file; otherwise append to the existing file
        """
        if b_1st_time :
            file = open(fn, 'w')
        else :
            file = open(fn, 'a')
            
        file.write( "{:.6e}".format(self.conc) )
        file.write('\t')
        file.write( "{:.6e}".format(self.conc_sink) )
        file.write('\n')