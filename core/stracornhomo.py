# -*- coding: utf-8 -*-
"""

@author: tc0008
"""
import importlib
import numpy as np

from core import comp
importlib.reload(comp)


class StraCornHomo(comp.Comp):
    """Class definition for homogenised stratum corneum
    """
    
    def __init__(self, xlen, ylen, dz_dtheta, nx, ny, init_conc, Kw, D, \
                 coord_sys, bdy_cond):
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
        comp.Comp.createMeshHomo(self, 'SC', chem, init_conc, coord_x_start, coord_y_start)
        #self.meshes[0].setConc(1)

        
    def compParDiff(self, chem) :
        """ Compute the partition coefficient with respect to water
        and the diffusion coefficient
        """
        import warnings
        warnings.warn('Partition and diffusion coefficients in homogenised stratum corneum are NOT calculated. '
                      'Only user-supplied values are used.')

    def compODEdydt(self, t, y, args=None):
        """ The wrapper function for computing the right hand side of ODEs
        """
        return comp.Comp.compODEdydt_diffu(self, t, y, args)
        
    def saveCoord(self, fn_x, fn_y):
        comp.Comp.saveCoord(self, fn_x, fn_y, '.ve')
