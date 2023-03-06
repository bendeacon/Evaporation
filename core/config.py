# -*- coding: utf-8 -*-
'''
Created on Tue Apr 18 22:01:17 2017

@author: tc0008
'''

import warnings
from importlib import reload
import numpy as np

from core import stracorn
reload(stracorn)

class Comp_Geom:
    """ This class contains the geometric parameters of a compartment
    """
    def __init__(self):
        self.name = None
        
        self.len_x = -1
        self.len_y = -1
        self.n_mesh_x = -1
        self.n_mesh_y = -1        
        
        # The following are specific to stratum corneum
        self.n_layer_x_sc = -1
        self.n_layer_y_sc = -1
        self.n_mesh_x_sc_lp = -1
        self.n_mesh_y_sc_lp = -1
        
class Config:
    """ This class intends to read from a text configuration
    file the properties of the chemical(s), the setup of the skin compartments 
    and their dimensions, and the simulation parameters,
    and then feeds them to the appropriate classes (e.g. Chemical, Skin etc.)
    """
    
    def __init__(self, fn_config=None):
        """         """
        ## Default values
        self.comps_nrow = 0
        self.comps_ncol = 0
        self.comps_geom = None      

        if fn_config is not None:
            self.readFile(fn_config)            
            self.updateGeometry()
        
        # parameters used in QSPRs to calculate Kw and D
        self.Kw_sc_paras = None
        self.D_sc_paras = None
    
    def readFile(self, fn_config):
        """ Read text from a configuration file, and parse using method <readTokens> """
        with open(fn_config, 'r') as f:
            lines = f.readlines()
            f.close()
            for lin in lines:
                tokens = list( filter(None, lin.split()) )
                if len(tokens) != 0 :
                    self.readTokens(tokens)
                    
                    
    def readTokens(self, tokens):
        
        #print(tokens)

        if tokens[0][0] == '#' : # comments, ignore
            return            
        
        # setup compartments
        elif tokens[0] == 'COMPARTMENT_SETUP': 
            self.sComps = tokens[1]
            tks = list( filter(None, tokens[1].split(',')) )
            self.comps_nrow = len(tks)
            self.comps_ncol = len(tks[0])
            #print(tks)            
            
            self.comps_geom = [Comp_Geom() for i in range(self.comps_nrow*self.comps_ncol)]
            for i in range(self.comps_nrow):
                for j in range(self.comps_ncol):
                    #print(tks[i][j])
                    self.comps_geom[i*self.comps_ncol+j].name = tks[i][j]

        elif tokens[0] == 'COMP': 
            comp_idx = int(tokens[1])
            if self.comps_geom[comp_idx].name == 'S': # Stratum corneum
                self.comps_geom[comp_idx].n_layer_x_sc, self.comps_geom[comp_idx].n_layer_y_sc, \
                self.comps_geom[comp_idx].n_mesh_x_sc_lp, self.comps_geom[comp_idx].n_mesh_y_sc_lp \
                = [int(k) for k in tokens[2:]]
                
                # The purpose of running this is to calculate the dimensions and number of
                #   meshes in SC from the supplied information
                sc = stracorn.StraCorn(self.comps_geom[comp_idx].n_layer_x_sc, self.comps_geom[comp_idx].n_layer_y_sc, \
                                       1, 0, self.comps_geom[comp_idx].n_mesh_x_sc_lp, self.comps_geom[comp_idx].n_mesh_y_sc_lp, \
                                        0, -1, -1, None, None)
                self.comps_geom[comp_idx].len_x = sc.get_x_length()
                self.comps_geom[comp_idx].len_y = sc.get_y_length()
                self.comps_geom[comp_idx].n_mesh_x = sc.get_nx()
                self.comps_geom[comp_idx].n_mesh_y = sc.get_ny()
            else :
                self.comps_geom[comp_idx].len_x, self.comps_geom[comp_idx].len_y = [float(k) for k in tokens[2:4]]
                self.comps_geom[comp_idx].n_mesh_x, self.comps_geom[comp_idx].n_mesh_y = [int(k) for k in tokens[4:]]


        ### parameters relating to chemical(s)
        elif tokens[0] == 'CHEM_NO' : # number of compounds
            self.nChem = int(tokens[1])
        elif tokens[0] == 'CHEM_MW' : # molecular weight
            self.mw = float(tokens[1])
        elif tokens[0] == 'CHEM_KOW' : # partition coefficient between octanol and water
            self.K_ow = float(tokens[1])
        elif tokens[0] == 'CHEM_PKA' : # pKa -- acide dissociation constant
            self.pKa = float(tokens[1])
        elif tokens[0] == 'CHEM_NONION' : # fraction of solute non-ionised at pH 7.4
            self.frac_non_ion = float(tokens[1])
        elif tokens[0] == 'CHEM_UNBND' : # fraction of solute unbound in a 2.7% albumin solution at pH 7.4    
            self.frac_unbound = float(tokens[1])
        elif tokens[0] == 'CHEM_ACIDBASE' :
            self.acid_base = tokens[1]
        elif tokens[0] == 'CHEM_DENSITY' :
            self.rho_chem = float(tokens[1])  
        elif tokens[0] == 'CHEM_PHASE' :
            self.phase_chem = tokens[1]
            
        ### vehicle specific parameters
        elif tokens[0] == 'INFINITE_VH' : # 
            self.b_infinite_vehicle = bool(int(tokens[1]))
            #print(tokens[1])
            #print(self.b_infinite_vehicle)
        elif tokens[0] == 'AREA_VH' :
            self.area_vehicle = float(tokens[1])
        elif tokens[0] == 'EVAP_SOLVENT_VH' :
            self.k_evap_solvent_vehicle = float(tokens[1])
        elif tokens[0] == 'SOLVENT_DENSITY' :
            self.rho_solvent = float(tokens[1])
        elif tokens[0] == 'SOLVENT_MW' :
            self.mw_solvent = float(tokens[1])
        elif tokens[0] == 'EVAP_SOLUTE_VH' :
            self.k_evap_solute_vehicle = float(tokens[1])
        elif tokens[0] == 'SOLUBILITY_VH' :
            self.solubility_vehicle = float(tokens[1])          
        
        ### Inital conditions
        elif tokens[0] == 'INIT_CONC_VH' : # vehicle
            self.init_conc_vh = float(tokens[1])
        elif tokens[0] == 'INIT_CONC_SC' : # stratum corneum
            self.init_conc_sc = float(tokens[1])
        elif tokens[0] == 'INIT_CONC_VE' : # viable epidermis
            self.init_conc_ve = float(tokens[1])
        elif tokens[0] == 'INIT_CONC_DE' : # dermis
            self.init_conc_de = float(tokens[1])
        elif tokens[0] == 'INIT_CONC_HF' : # hair follicle
            self.init_conc_hf = float(tokens[1])
        elif tokens[0] == 'INIT_CONC_BD' : # blood
            self.init_conc_bd = float(tokens[1])
            
            
        # partition and diffusion coefficients
        elif tokens[0] == 'KW_VH' :
            self.Kw_vh = float(tokens[1])
        elif tokens[0] == 'D_VH' : 
            self.D_vh = float(tokens[1])
        elif tokens[0] == 'KW_SC' :
            self.Kw_sc = float(tokens[1])
        elif tokens[0] == 'D_SC' : 
            self.D_sc = float(tokens[1])
        elif tokens[0] == 'KW_VE' :
            self.Kw_ve = float(tokens[1])
        elif tokens[0] == 'D_VE' : 
            self.D_ve = float(tokens[1])    
        elif tokens[0] == 'KW_DE' :
            self.Kw_de = float(tokens[1])
        elif tokens[0] == 'D_DE' : 
            self.D_de = float(tokens[1])
        elif tokens[0] == 'KW_HF' :
            self.Kw_hf = float(tokens[1])
        elif tokens[0] == 'D_HF' : 
            self.D_hf = float(tokens[1])
        elif tokens[0] == 'K_DE2BD' : # dermis to blood partition
            self.K_de2bd = float(tokens[1])
        elif tokens[0] == 'CLEAR_BD' : 
            self.Cl_bd = float(tokens[1])         

        # name not found
        else :
            warnings.warn('Unrecognised line in config file')
            
    def updateGeometry(self) :
        """ Update the gemoetric parameters based on what is provided in configuration file
        """
        for i in range(self.comps_nrow):
            for j in range(self.comps_ncol):
                idx = i*self.comps_ncol + j
                
                if self.comps_geom[idx].len_x < 0:
                    if j > 0 :
                        self.comps_geom[idx].len_x = self.comps_geom[idx-1].len_x                    
                    else :
                        raise ValueError('Invalid value for conf.comps_geom[idx].len_x')
                        
                if self.comps_geom[idx].len_y < 0:
                    if i > 0 :
                        self.comps_geom[idx].len_y = self.comps_geom[(i-1)*self.comps_ncol+j].len_y
                    elif self.comps_geom[ (i+1)*self.comps_ncol + j ].name == 'S' : # stratum corneum
                        self.comps_geom[idx].len_y = self.comps_geom[(i+1)*self.comps_ncol + j].len_y
                    else :
                        raise ValueError('Invalid value for conf.comps_geom[idx].len_y')
         
    def combine(self, other_conf):
        """ Combine <other_conf> into this conf object
        If this conf contains the same attributes, they will be replaced by those in <other_conf>
        and warning messages will be reported
        """
        members = [attr for attr in dir(other_conf) if not callable(getattr(other_conf, attr)) and not attr.startswith("__")]
        for attr in members:
            val = getattr(other_conf, attr)
            if hasattr(self, attr):
                msg = "Conflict values when trying to combine two config objects; attributes: " + attr + "; self value = " + str(getattr(self, attr)) + " , other value = " + str(val)
                warnings.warn(msg, DeprecationWarning)
            setattr(self, attr, val)