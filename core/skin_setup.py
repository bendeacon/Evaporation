# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 09:57:44 2017

@author: tc0008
"""
import warnings
import importlib
#import numpy as np

from core import vehicle
importlib.reload(vehicle)
from core import stracorn
importlib.reload(stracorn)
from core import stracornhomo
importlib.reload(stracornhomo)
from core import viaepd
importlib.reload(viaepd)
from core import dermis
importlib.reload(dermis)
from core import hairfoll
importlib.reload(hairfoll)
from core import skin
importlib.reload(skin)

class Skin_Setup(skin.Skin):
    """ Class definition for Skin_Setup
    which intends to set up the compartments in simulation as instructed by user
    """
    def __init__(self, chem, conf):
        skin.Skin.__init__(self)
        self.comp_structure = conf.sComps
        
        if chem is None:
            skin.Skin.set_n_species(self, 0)
        elif type(chem) is not list :
            skin.Skin.set_n_species(self, 1)
        else :
            skin.Skin.set_n_species(self, len(chem)) # Number of chemcial species
        
        if conf.sComps.find('B') == 1 and conf.sComps.find('D') == 1 : 
            self.b_has_blood = True # Has blood compartment ('B') in dermis ('D')
        else :
            self.b_has_blood = False
    
    def createComps(self, chem, conf) :
        """ Create compartments
        Letter code:
            V: vehicle            S: 2D stratum cornuem       O: 1D stratum corneum
            E: viable epidermis   D: dermis
            B: blood              H: Hair
        """
        
        # Read structure of the compartments
        #tokens = list( filter(None, self.comp_structure.split(',')) )        
        #nrow = len(tokens)
        #ncol = len(tokens[0])        
        skin.Skin.createComps(self, conf.comps_nrow, conf.comps_ncol)        
        
        # Actually create the compartments
        
        dim_all = 0
        current_x = 0
        current_y = 0        
        
        for i in range(conf.comps_nrow) :
            
            for j in range(conf.comps_ncol) :
                
                idx = i*conf.comps_ncol + j
                
                # determine the boundary conditions for this compartment
                bdy_up = 'ZeroFlux' if i==0 else 'FromOther'                
                bdy_down = 'ZeroFlux' if i==conf.comps_nrow-1 else 'FromOther'
                #bdy_down = 'ZeroConc' if i==conf.comps_nrow-1 else 'FromOther'
                
                if conf.comps_ncol == 1 : # only one column of compartments
                    bdy_left = 'Periodic'
                    bdy_right = 'Periodic'
                else :
                    warnings.warn('Periodic boundary conditions have not been implemented for more-than-one-column compartments, ZeroFlux is used instead')
                    bdy_left = 'ZeroFlux' if j==0 else 'FromOther'                    
                    bdy_right = 'ZeroFlux' if j==conf.comps_ncol-1 else 'FromOther'
                
                bdy_cond = [bdy_up, bdy_left, bdy_right, bdy_down]
                

                if conf.comps_geom[idx].name == 'V':
                    comp = self.createVH(chem, current_x, current_y, conf.comps_geom[idx].len_x, conf.comps_geom[idx].len_y, 
                                         conf.comps_geom[idx].n_mesh_x, conf.comps_geom[idx].n_mesh_y, conf.init_conc_vh,
                                         conf.Kw_vh, conf.D_vh, bdy_cond, conf.b_infinite_vehicle,
                                         conf.rho_chem, conf.rho_solvent, conf.mw_solvent, conf.phase_chem,
                                         conf.k_evap_solvent_vehicle, conf.k_evap_solute_vehicle, conf.solubility_vehicle) 
                elif conf.comps_geom[idx].name == 'S':
                    comp = self.createSC(chem, current_x, current_y, conf.comps_geom[idx].n_layer_x_sc, conf.comps_geom[idx].n_layer_y_sc, \
                                         conf.comps_geom[idx].n_mesh_x_sc_lp, conf.comps_geom[idx].n_mesh_y_sc_lp, conf.init_conc_sc, \
                                         conf.Kw_sc, conf.D_sc, conf.Kw_sc_paras, conf.D_sc_paras, bdy_cond)
                    if type(self.comps[0]) is vehicle.Vehicle :
                        self.comps[0].K_lip_water = comp.meshes[0].Kw
                        #print("K_lip_water = ", self.comps[0].K_lip_water)
                elif conf.comps_geom[idx].name == 'O':
                    comp = self.createSCH(chem, current_x, current_y, conf.comps_geom[idx].len_x, conf.comps_geom[idx].len_y,
                                          conf.comps_geom[idx].n_mesh_x, conf.comps_geom[idx].n_mesh_y, conf.init_conc_sc,
                                          conf.Kw_sc, conf.D_sc, bdy_cond)
                    if type(self.comps[0]) is vehicle.Vehicle:
                        self.comps[0].K_lip_water = comp.meshes[0].Kw
                elif conf.comps_geom[idx].name == 'E':
                    comp = self.createVE(chem, current_x, current_y, conf.comps_geom[idx].len_x, conf.comps_geom[idx].len_y, \
                                         conf.comps_geom[idx].n_mesh_x, conf.comps_geom[idx].n_mesh_y, conf.init_conc_ve, \
                                         conf.Kw_ve, conf.D_ve, bdy_cond)
                elif conf.comps_geom[idx].name == 'D':
                    comp = self.createDE(chem, current_x, current_y, conf.comps_geom[idx].len_x, conf.comps_geom[idx].len_y, \
                                         conf.comps_geom[idx].n_mesh_x, conf.comps_geom[idx].n_mesh_y, conf.init_conc_de, \
                                         conf.Kw_de, conf.D_de, bdy_cond)
                elif conf.comps_geom[idx].name == 'H':
                    comp = self.createHF(chem, current_x, current_y, conf.comps_geom[idx].len_x, conf.comps_geom[idx].len_y, \
                                         conf.comps_geom[idx].n_mesh_x, conf.comps_geom[idx].n_mesh_y, conf.init_conc_hf, \
                                         conf.Kw_hf, conf.D_hf, bdy_cond)
                else :
                    pass
                skin.Skin.setComp(self, comp, i, j)
                dim_all += comp.get_dim()
                current_y += comp.get_y_length()
                
            current_x += comp.get_x_length()
            current_y = 0
            
        skin.Skin.set_dim_all(self, dim_all)
        
        # Link compartments through boundary conditions
        for i in range(conf.comps_nrow) :
            for j in range(conf.comps_ncol) :
                
                # down boundary
                if i == conf.comps_nrow-1 : # down-most compartmnet, its down boundary is zero-flux
                    n_dBdy = 0
                    mesh_dBdy = None
                else :
                    compDown = skin.Skin.getComp(self, i+1, j)
                    n_dBdy = compDown.get_ny()
                    mesh_dBdy = compDown.meshes[0:n_dBdy]
                    
                # right boundary
                if j==conf.comps_ncol-1 : # right-most compartment, its right boundary is zero-flux
                    n_rBdy = 0
                    mesh_rBdy = None
                else :
                    compRight = skin.Skin.getComp(self, i, j+1)
                    n_rBdy = compRight.get_nx()
                    mesh_rBdy = compRight.meshes[::compRight.get_ny()]
                    
                comp = skin.Skin.getComp(self, i, j)
                comp.createBdy(n_rBdy, n_dBdy)
                comp.setBdyMesh(mesh_rBdy, mesh_dBdy)
                #print('n_rBdy=', n_rBdy, ' n_dBdy=', n_dBdy)
                    
        
    ### (START OF) Create individual compartments ###
    
    def createVH(self, chem, 
                 coord_x_start, coord_y_start, xlen, ylen, n_grids_x, n_grids_y,
                 init_conc, Kw, D, bdyCond, b_inf_source,
                 rho_solute, rho_solvent, mw_solvent, phase_solute,
                 k_evap_solvent, k_evap_solute, solubility) :
        """ Create vehicle """        
        veh = vehicle.Vehicle(chem, xlen, ylen, self.dz_dtheta, \
                              n_grids_x, n_grids_y, init_conc, Kw, D, self.coord_sys, \
                              bdyCond, b_inf_source, rho_solute, rho_solvent, mw_solvent, phase_solute,\
                              k_evap_solvent, k_evap_solute, solubility)
        veh.createMesh(chem, coord_x_start, coord_y_start)
        return veh
        
    def createSC(self, chem, 
                 coord_x_start, coord_y_start, n_layer_x, n_layer_y, n_mesh_x_lp, n_mesh_y_lp, 
                 init_conc, Kw, D, Kw_paras, D_paras, bdyCond) :
        """ Create stratum corneum """
        offset_y = 0                
        sc = stracorn.StraCorn(n_layer_x, n_layer_y, self.dz_dtheta, offset_y, \
                               n_mesh_x_lp, n_mesh_y_lp, init_conc, Kw, D, self.coord_sys, bdyCond)
        sc.setParDiff_paras(Kw_paras, D_paras)
        sc.createMesh(chem, coord_x_start, coord_y_start)
        return sc
        
    def createVE(self, chem, 
                 coord_x_start, coord_y_start, xlen, ylen, n_grids_x, n_grids_y,
                 init_conc, Kw, D, bdyCond) :
        """ Create viable epidermis """        
        via_epidermis = viaepd.ViaEpd(xlen, ylen, self.dz_dtheta, \
                                      n_grids_x, n_grids_y, init_conc, Kw, D, self.coord_sys, bdyCond)
        via_epidermis.createMesh(chem, coord_x_start, coord_y_start)
        return via_epidermis

    def createSCH(self, chem,
                 coord_x_start, coord_y_start, xlen, ylen, n_grids_x, n_grids_y,
                 init_conc, Kw, D, bdyCond) :
        """ Create homogenised stratum corneum """
        sc_homo = stracornhomo.StraCornHomo(xlen, ylen, self.dz_dtheta,
                                            n_grids_x, n_grids_y, init_conc, Kw, D, self.coord_sys, bdyCond)
        sc_homo.createMesh(chem, coord_x_start, coord_y_start)
        return sc_homo

    def createDE(self, chem, 
                 coord_x_start, coord_y_start, xlen, ylen, n_grids_x, n_grids_y,
                 init_conc, Kw, D, bdyCond) :
        """ Create dermis """        
        derm = dermis.Dermis(xlen, ylen, self.dz_dtheta, 
                             n_grids_x, n_grids_y, init_conc, Kw, D, self.coord_sys, bdyCond, self.b_has_blood)
        derm.createMesh(chem, coord_x_start, coord_y_start)
        return derm
        
    def createHF(self, chem, 
                 coord_x_start, coord_y_start, xlen, ylen, n_grids_x, n_grids_y,
                 init_conc, Kw, D, bdyCond) :
        """ Create viable epidermis """        
        hf = hairfoll.HairFoll(xlen, ylen, self.dz_dtheta, 
                               n_grids_x, n_grids_y, init_conc, Kw, D, self.coord_sys, bdyCond)
        #print('Kw=', Kw, 'D=', D)
        hf.createMesh(chem, coord_x_start, coord_y_start)
        return hf
        
    ### (END OF) Create individual compartments ###
    
    
    ### (START OF) Methods dealing with penetration enhancers ###
    
    def updateKw_fromOther(self, this_conf, other_skin, multiplier) :
        """ Update the partition coefficient (thus this chemical species) based on
        the concentration of <other_skin> (thus the other chemical species)    """
        for i in range(this_conf.comps_nrow) :            
            for j in range(this_conf.comps_ncol) :                
                idx = i*this_conf.comps_ncol + j
                
                this_comp = self.comps[idx]
                other_comp = other_skin.comps[idx]
                
                if this_conf.comps_geom[idx].name != 'V': # if it's vehicle, do not update
                    this_comp.setMeshes_Kw_fromOther(other_comp, multiplier)
    # End of updateKw_fromOther()
                        
    ### (END OF) Methods dealing with penetration enhancers ###