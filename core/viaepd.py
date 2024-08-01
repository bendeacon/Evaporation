# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 13:28:52 2017

@author: tc0008
"""
import importlib
import numpy as np

from core import comp
importlib.reload(comp)


class ViaEpd(comp.Comp):
    """Class definition for ViaEpd
    which is the viable epidermis, currently modelled as a homogenised media
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
        comp.Comp.createMeshHomo(self, 'VE', chem, init_conc, coord_x_start, coord_y_start)
        #self.meshes[0].setConc(1)

        
    def compParDiff(self, chem) :
        """ Compute the partition coefficient with respect to water
        and the diffusion coefficient
        """
        #print(self.Kw)
        rou_lipid = 0.9
        rou_water = 1
        K_lip = rou_lipid / rou_water * chem.get_K_ow()**0.69 # Partition in SC lipid
  
        binding_factor = 0.65 + 0.325/chem.get_frac_unbound() + 0.025*chem.get_frac_non_ion()*K_lip

        if self.Kw < 0: # Negative means it requires calculation; otherwise use user-input value
            Kw = 0.6 * binding_factor
            comp.Comp.set_Kw(self, Kw)
        elif self.Kw > 0:
            Kw = self.Kw
           
        
        # c.f. L. Chen's Phar. Res. paper; -8.15 used because of unit (m2/s)
        #   Kasting's original paper used -4.15 because of unit (cm2/s)
        D_free = 10 ** ( -8.15-0.655*np.log10(chem.get_mw()) )
        if self.D < 0: # Negative means it requires calculation; otherwise use user-input value
            D = D_free / binding_factor
            comp.Comp.set_D(self, D)
        elif self.D > 0:
            D = self.D
        #print('self.D = ', self.D)  
        print("Kepi",Kw,"Depi", D)
        return (Kw, D)

    def compODEdydt(self, t, y, args=None):
        """ The wrapper function for computing the right hand side of ODEs
        """
        return comp.Comp.compODEdydt_diffu (self, t, y, args)
        
    def saveCoord(self, fn_x, fn_y) :
        comp.Comp.saveCoord(self, fn_x, fn_y, '.ve')
