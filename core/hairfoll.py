# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 13:28:52 2017

@author: tc0008
"""
import importlib
import numpy as np

from core import comp
importlib.reload(comp)


class HairFoll(comp.Comp):
    """Class definition for HairFoll
    which is the hair follicle, currently modelled as a homogenised sebum media
    """
    
    def __init__(self, xlen, ylen, dz_dtheta, nx, ny, init_conc, Kw, D, coord_sys, bdy_cond):
        comp.Comp.__init__(self)
        comp.Comp.setup(self, xlen, ylen, dz_dtheta, nx, ny, coord_sys, bdy_cond)
        
        self.init_conc = init_conc
        comp.Comp.set_Kw(self, Kw)
        comp.Comp.set_D(self, D)        
        
    def createMesh(self, chem, coord_x_start, coord_y_start) :
        """ Create mesh for this compartment
        Args:
                coord_x_start, coord_y_start: starting coordinates
        """
        init_conc = .0
        self.compParDiff(chem)
        comp.Comp.createMeshHomo(self, 'HF', chem, init_conc, coord_x_start, coord_y_start)
        
        
    def compParDiff(self, chem) :
        """ Compute the partition coefficient with respect to water
        and the diffusion coefficient
        """
        # todo: include Senpei's QSPR model
        if self.Kw < 0: # Negative means it requires calculation; otherwise use user-input value
            Kw = 5.62
            comp.Comp.set_Kw(self, Kw)
        else:
            Kw = self.Kw
        
        if self.D < 0:
            D = 1.12e-11
            comp.Comp.set_D(self, D)
        else:
            D = self.D
        
        return (Kw, D)

    def compODEdydt(self, t, y, args=None):
        """ The wrapper function for computing the right hand side of ODEs
        """
        return comp.Comp.compODEdydt_diffu (self, t, y, args)
        
    def saveCoord(self, fn_x, fn_y) :
        comp.Comp.saveCoord(self, fn_x, fn_y, '.hf')
