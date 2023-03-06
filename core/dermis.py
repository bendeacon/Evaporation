# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 13:28:52 2017

@author: tc0008
"""
import importlib
import numpy as np

from core import comp
importlib.reload(comp)
from core import viaepd
importlib.reload(viaepd)

class Dermis(viaepd.ViaEpd):
    """Class definition for Dermis
    which is the dermis, currently modelled as a homogenised media,
    the same as viable epidermis but with the possibility of
    blood flow through
    """
    
    def __init__(self, xlen, ylen, dz_dtheta, nx, ny, init_conc, Kw, D, \
                 coord_sys, bdy_cond, b_has_blood=False):
        viaepd.ViaEpd.__init__(self, xlen, ylen, dz_dtheta, nx, ny, init_conc, Kw, D, coord_sys, bdy_cond)
        
        ## blood flow ralated varaibles
        self.b_has_blood = b_has_blood
        self.bld_skin_flow = None # total blood flow rate in skin
        self.bld_fu = None # fraction of unbounded solute in blood
        self.par_de2blood = None # partition coefficient from dermis to blood
        self.bld_conc = .0 # concentration in blood
        self.dermis_totalV = None # total volume of dermis
        self.mass_into_dermis = .0 
        self.mass_outof_dermis = .0
    
    def get_dim(self):
        return viaepd.ViaEpd.get_dim(self)
    def getBloodConc(self):
        return self.bld_conc
    def setBloodConc(self, conc):
        self.bld_conc = conc
        
    def createDermisBlood(self, bld_skin_flow, bld_fu, par_de2blood, bld_conc, skin_area):
        """ Create variables relating to blood flow
        """
        assert(self.b_has_blood)
        self.bld_skin_flow = bld_skin_flow # total blood flow rate in skin
        self.bld_fu = bld_fu # fraction of unbounded solute in blood
        self.par_de2blood = par_de2blood # partition coefficient from dermis to blood
        self.bld_conc = bld_conc # initial concentration in blood
        self.dermis_totalV = self.x_length*skin_area # total volume of dermis
        
    def createMesh(self, chem, coord_x_start, coord_y_start) :
        """ Create mesh for this compartment
        Args:
                coord_x_start, coord_y_start: starting coordinates
        """
        init_conc = .0
        self.compParDiff(chem)
        comp.Comp.createMeshHomo(self, 'DE', chem, init_conc, coord_x_start, coord_y_start)        
        
    def compParDiff(self, chem) :
        """ Compute the partition coefficient with respect to water
        and the diffusion coefficient
        """
        viaepd.ViaEpd.compParDiff(self, chem)
        
        
    ### (START OF) Class methods dealing with ODE computation ###
    
    def compODEdydt(self, t, y, args=None):
        """ The wrapper function for computing the right hand side of ODEs
        """
        f = comp.Comp.compODEdydt_diffu (self, t, y, args)
        if self.b_has_blood :
            f1 = self.compODEdydt_blood(self, t, y)
            f += f1
        return f
        
    def compODEdydt_blood(self, t, y, args=None):
        """ Compute the right hand side of ODEs due to blood flow
        """
        assert(self.b_has_blood)
        f = np.zeros(y.shape) # f contains dy/dt to be returned
        self.mass_into_dermis = .0 # reset both for later calculation for blood
        self.mass_outof_dermis = .0
        # Calculate mass flow from blood to this mesh
        
        for i in range(self.nx) : #  x direction up to down
            for j in range(self.ny) : # y direction left to right
                idx_this = i*self.ny+j
                
                meshThis = self.meshes[idx_this]
                conc_this = y[idx_this]
                volume_this = meshThis.compVolume()
                
                flow_this_mesh = self.bld_skin_flow * volume_this / self.dermis_totalV
                fin = flow_this_mesh * self.bld_conc
                fout = flow_this_mesh * conc_this * (meshThis.chem.get_frac_unbound()/self.bld_fu) / self.par_de2blood
                
                f[idx_this] = (fin-fout)/volume_this
                self.mass_into_dermis += fin
                self.mass_outof_dermis += fout
            # for j
        # for i        
        return f

    ### (END OF) Class methods dealing with ODE computation ###

    def saveCoord(self, fn_x, fn_y) :
        comp.Comp.saveCoord(self, fn_x, fn_y, '.de')
