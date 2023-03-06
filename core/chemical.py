# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:37:24 2017

@author: tc0008
"""
import numpy as np

class Chemical:
    """Class definition for Chemical whose mass transfer is of interest
    """
    def __init__(self, conf=None):
        if conf is not None :
            self.setChem(conf.mw, conf.K_ow, conf.pKa, conf.frac_non_ion, conf.frac_unbound, conf.acid_base)
    
    def setChemConf(self, conf=None):
        if conf is not None :
            self.setChem(conf.mw, conf.K_ow, conf.pKa, conf.frac_non_ion, conf.frac_unbound, conf.acid_base)
            
    def setChem(self, mw, K_ow, pKa, frac_non_ion = None, frac_unbound = None, acid_base = None):        
        
        self.mw = mw
        self.r_s = ( 0.9087 * mw * 3/4/np.pi ) ** (1.0/3) * 1e-10 # from Anstrom to meter
        
        self.K_ow = K_ow
        self.pKa = pKa
        self.acid_base = acid_base        
        self.frac_non_ion = frac_non_ion if frac_non_ion is not None else self.compIon()
        self.frac_unbound = frac_unbound if frac_unbound is not None else self.compBinding()
    
    def get_mw(self):
        return self.mw
    def get_K_ow(self):
        return self.K_ow
    def set_mw(self, mw):
        self.mw = mw
    def set_K_ow(self, K_ow):
        self.K_ow = K_ow
    def get_frac_unbound(self):
        return self.frac_unbound
    def get_frac_non_ion(self):
        return self.frac_non_ion
        
    ### (END OF) instance initialisation methods ###
    
    def compIon(self) :
        """ Compute the fraction of solute non-ionised at pH 7.4 (frac_non_ion) 
        Refs: Florence AT, Attwood D (2006). Physicochemical Principles of Pharmacy, Pharmaceutical Press, London, p. 77.
        """
        if self.acid_base == 'A' : # weak acid
            self.frac_non_ion = 1 / ( 1 + 10**(7.4-self.pKa) )
        elif self.acid_base == 'B' : # weak base
            self.frac_non_ion = 1 / ( 1 + 10**(self.pKa-7.4) )
        else:
            raise ValueError('Invalid specification for acid/base')
    
    def compBinding(self) :
        """ Compute the fraction of unbound in a 2.7% albumin solution at pH 7.4 (frac_unbound)
        Refs:  Yamazaki K, Kanaoka M (2004). Journal of Pharmaceutical Sciences, 93: 1480.
        """
        if self.acid_base == 'A' : # weak acid
            self.frac_unbound = 1 - ( 0.7936 * np.exp(np.log10(self.K_ow)) + 0.2239 ) / ( 0.7936 * np.exp(np.log10(self.K_ow)) + 1.2239 )
        elif self.acid_base == 'B' : # weak base
            self.frac_unbound = 1 - ( 0.5578 * np.exp(np.log10(self.K_ow)) + 0.0188 ) / ( 0.5578 * np.exp(np.log10(self.K_ow)) + 1.0188 )
        else:
            raise ValueError('Invalid specification for acid/base')
