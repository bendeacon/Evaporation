# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 10:52:24 2017

@author: tc0008
"""

import importlib
import numpy as np

from core import mesh
importlib.reload(mesh)
from core import comp
importlib.reload(comp)
from core import point
from core import para

class KwDParas():
    """ """
    def __init__(self):
        self.lp = para.optval()
        self.lp.option = 'DEFT'
        self.lp.value = None
        self.cc = para.optval()
        self.cc.option = 'DEFT'
        self.cc.value = None
        
    
class StraCorn(comp.Comp):
    """ Class definition for StraCorn, 
    which is the stratum corneum, currently modelled as a heterogeneous media
    """
    
    def __init__(self, n_layer_x, n_layer_y, dz_dtheta, offset_y, 
                 n_meshes_x_lp, n_meshes_y_lp, init_conc, Kw, D, coord_sys, bdy_cond) :
        
        comp.Comp.__init__(self)
          
        # dimension related; c.f. Readme.docx for more details
        self.w = 8.0 # offset ratio, 8.0
        self.nx_grids_lipid = n_meshes_x_lp # Number of x-grids for lipid layer, 2
        self.nx_grids_cc = n_meshes_x_lp*2    # Number of x-grids for corneocyte layer, 4
        self.ny_grids_lipid = n_meshes_y_lp # Number of y-grids for lipid layer, 2
        self.ny_grids_cc_dn = n_meshes_y_lp # Number of y-grids for dn-part of the offset corneocyte layer, 2
        
        self.eta = 7.644E-4 # water viscosity at 305 K (32 deg C) (Pa s),  
        
        self.geom_g = 0.075e-6  
        self.geom_d = 40e-6   
        self.geom_s = 0.075e-6  
        self.geom_t = 0.8e-6 

        self.geom_dm = self.w*(self.geom_d-self.geom_s)/(1+self.w)
        self.geom_dn = self.geom_d-self.geom_s-self.geom_dm
	
        # Vertical direction, lipid layer is at both top and bottom of the stratum corneum
        nx = (self.nx_grids_lipid+self.nx_grids_cc)*n_layer_x + self.nx_grids_lipid
        # Lateral direction, [dh] [s] [dm] [s], here d=dh+dm+s, w=dm/dh
        ny = int( ( self.ny_grids_lipid*2 + self.ny_grids_cc_dn + self.ny_grids_cc_dn*self.w ) * n_layer_y )
        
        xlen = n_layer_x*(self.geom_g+self.geom_t)+self.geom_g
        ylen = n_layer_y*(self.geom_d+self.geom_s)
        
        comp.Comp.setup(self, xlen, ylen, dz_dtheta, nx, ny, coord_sys, bdy_cond)

        # For the volume fraction calculation, we assume a Cartesian coordinate
        #   and the z-direction width m_dz_dtheta is directly used.
        #   This won't affect if other coordinates are used
        self.V_mortar = ( self.geom_g*(self.geom_d+self.geom_s)+self.geom_t*self.geom_s ) * dz_dtheta
        self.V_brick = self.geom_d*self.geom_t * dz_dtheta
        self.V_all = self.V_mortar + self.V_brick

        self.offset_y =  offset_y
        
        #self.Kw_paras = np.array([-1]*3)
        #self.D_paras = np.array([-1]*4)
        self.Kw_paras = KwDParas()
        self.D_paras = KwDParas()
        
        
    def setKwOptions(self, Kw_paras=None):
        """ Set up the way to calculate or input parameters for Kw 
            self.Kw_paras has two members:
                lp: lipid, which has two members itself:
                    option: 'DEFT' (QSPR calcualtion using default parameter values)
                            'QSPR' (QSPR calcualtion using user supplied parameter values)
                            'VALE' (Do not calculate, use user supplied Kw directly)
                    value: None when option is 'DEFT'
                           <a> as in K = (rho_lip/rho_wat) * Kow^a, when option is 'QSPR'
                           The Kw value, when option is 'VALE'
                cc: corneocyte, which has two members itself:
                    option: 'DEFT' (QSPR calcualtion using default parameter values)
                            'QSPR' (QSPR calcualtion using user supplied parameter values)
                            'VALE' (Do not calculate, use user supplied Kw directly)
                    value: None when option is 'DEFT'
                           <a, b> as in K = (rho_pro/rho_wat) * a * Kow^b, when option is 'QSPR'
                           The Kw value, when option is 'VALE'              
        """        
        #print(self.Kw_paras)
        if Kw_paras is None:
            self.Kw_paras.lp.option = 'DEFT'
            self.Kw_paras.lp.value = None
            self.Kw_paras.cc.option = 'DEFT'
            self.Kw_paras.cc.value = None
        else:
            self.Kw_paras.lp.option = Kw_paras.lp.option
            self.Kw_paras.lp.value = np.copy(Kw_paras.lp.value)
            self.Kw_paras.cc.option = Kw_paras.cc.option
            self.Kw_paras.cc.value = np.copy(Kw_paras.cc.value)
         
        
    def setDOptions(self, D_paras=None):
        """ Set up the way to calculate or input parameters for D 
            self.D_paras has two members:
                lp: lipid, which has two members itself:
                    option: 'DEFT' (QSPR calcualtion using default parameter values)
                            'QSPR' (QSPR calcualtion using user supplied parameter values)
                            'VALE' (Do not calculate, use user supplied D directly)
                    value: None when option is 'DEFT'
                           <a, b> as in D = a * exp(-b*r_s_inA*r_s_inA), when option is 'QSPR'
                           The D value, when option is 'VALE'
                cc: corneocyte, which has two members itself:
                    option: 'DEFT' (QSPR calcualtion using default parameter values)
                            'QSPR' (QSPR calcualtion using user supplied parameter values)
                            'VALE' (Do not calculate, use user supplied D directly)
                    value: None when option is 'DEFT'
                           <alpha, beta> as given in the reference, when option is 'QSPR'
                           The D value, when option is 'VALE'  
        """        
        if D_paras is None:
            self.D_paras.lp.option = 'DEFT'
            self.D_paras.lp.value = None
            self.D_paras.cc.option = 'DEFT'
            self.D_paras.cc.value = None
        else:
            self.D_paras.lp.option = D_paras.lp.option
            self.D_paras.lp.value = np.copy(D_paras.lp.value)
            self.D_paras.cc.option = D_paras.cc.option
            self.D_paras.cc.value = no.copy(D_paras.cc.value)
            
    
            
    def createMesh(self, chem, coord_x_start, coord_y_start, water_frac_surface=.55) :
        """ Create mesh for this compartment
        Args:
                coord_x_start, coord_y_start: starting coordinates
                water_frac_surface : water content (w/w); saturation at .55; dry skin at .25
        """
               
        # some initial settings
        bOffset = False
        cc_subtype = 0 # 0 = d_n; 1 = s; 2 = d_m, 3 = s;
  
        dx_lipid = self.geom_g / self.nx_grids_lipid
        dx_cc = self.geom_t / self.nx_grids_cc
        dy_lipid = self.geom_s / self.ny_grids_lipid
        dy_cc = self.geom_dn / self.ny_grids_cc_dn
        
        self.meshes = [mesh.Mesh() for i in range(self.nx*self.ny)] # organised in row dominant               
        
        # work out the starting point from given offset
        
        length = self.geom_d + self.geom_s
        len_vec = [self.geom_dn, self.geom_s, self.geom_dm, self.geom_s]

        while self.offset_y > length :
            self.offset_y -= length
        length = .0
        for i in range(4) :
            length += len_vec[i]
            if self.offset_y < length :
                break
        cc_subtype_offset = i
        length = self.offset_y - length + len_vec[i]

        if cc_subtype_offset == 0 or cc_subtype_offset == 2 :
            # corneocyte width
            idx_y_offset = int( length / dy_cc )
            dy_offset = dy_cc
        elif cc_subtype_offset == 1 or cc_subtype_offset == 3 :
            # lipid width
            idx_y_offset = int( length / dy_lipid )
            dy_offset = dy_lipid
        else :
            raise ValueError('Invalid subtype name')


        # Now prepare values for the loop to create meshes
        
        water_frac_sat = 0.55 # saturated water content (w/w)
        water_increment_per_x = (water_frac_sat - water_frac_surface) / self.x_length
    
        idx_x = 0
        idx_y = idx_y_offset
        cc_subtype = cc_subtype_offset

        coord_x = coord_x_start
        coord_y = coord_y_start
        # starting from lipid on the top layer
        current_point = point.Point(coord_x, coord_y, dx_lipid, dy_offset, 'LP', 'LP')
        init_conc = 0
        
        for i in range(self.nx) : # verticle direction up to down
            water_frac = water_frac_surface + (current_point.x_coord - coord_x_start) * water_increment_per_x

            for j in range(self.ny) : # lateral direction left to right
                idx = i*self.ny + j
                
                # assign type
                if current_point.x_type == 'LP' or current_point.y_type == 'LP' :
                    # Entire lipid layer (1st ==) or lateral lipid between two coreneocytes (2nd ==)
                    name = 'LP'
                else : # corneocyte
                    name = 'CC'
                #   and then create meshes
                Kw, D = self.compParDiff(name, chem, water_frac, water_frac_sat, 
                        self.V_mortar, self.V_brick, self.V_all, self.eta)                    
                self.meshes[idx].setup(name, chem, init_conc, Kw, D,
                        current_point.x_coord, current_point.y_coord, current_point.dx, current_point.dy, 
                        self.dz_dtheta)
                
                ### update current_point, fairly complicated
                
                if j==self.ny-1 : # last element in the lateral direction, move down
                
                    idx_x += 1
                    coord_y = coord_y_start
                    
                    if current_point.x_type == 'LP' : 
                        coord_x += dx_lipid
                        
                        if idx_x == self.nx_grids_lipid : # reaching end of lateral lipid layer
                            if cc_subtype_offset == 0 or cc_subtype_offset == 2 :
                                current_point.setPoint(coord_x, coord_y, dx_cc, dy_offset, 'CC', 'CC')
                            elif cc_subtype_offset == 1 :
                                if not bOffset:
                                    current_point.setPoint(coord_x, coord_y, dx_cc, dy_offset, 'CC', 'CC')
                                else :
                                    current_point.setPoint(coord_x, coord_y, dx_cc, dy_offset, 'CC', 'LP')
                            elif cc_subtype_offset == 3 :
                                if not bOffset:
                                    current_point.setPoint(coord_x, coord_y, dx_cc, dy_offset, 'CC', 'LP')
                                else :
                                    current_point.setPoint(coord_x, coord_y, dx_cc, dy_offset, 'CC', 'CC')
                            else :
                                raise ValueError('Invalid subtype name')	  
                            idx_x = 0 
                        else : # NOT reaching end of lateral lipid layer
                            current_point.setPoint(coord_x, coord_y, dx_lipid, dy_offset, 'LP', 'LP')
                            
                    elif current_point.x_type == 'CC' :
                        coord_x += dx_cc
                        
                        if idx_x == self.nx_grids_cc : # reaching end of lateral corneocyte layer
                            current_point.setPoint(coord_x, coord_y, dx_lipid, dy_offset, 'LP', 'LP')
                            idx_x = 0
                            bOffset = not bOffset
                        else : # NOT reaching end of lateral corneocyte layer
                            if cc_subtype_offset == 0 or cc_subtype_offset == 2 :
                                current_point.setPoint(coord_x, coord_y, dx_cc, dy_offset, 'CC', 'CC')
                            elif cc_subtype_offset == 1 :
                                if not bOffset :   
                                    current_point.setPoint(coord_x, coord_y, dx_cc, dy_offset, 'CC', 'CC')
                                else :            
                                    current_point.setPoint(coord_x, coord_y, dx_cc, dy_offset, 'CC', 'LP')
                            elif cc_subtype_offset == 3 :
                                if not bOffset :   
                                    current_point.setPoint(coord_x, coord_y, dx_cc, dy_offset, 'CC', 'LP')
                                else :           
                                    current_point.setPoint(coord_x, coord_y, dx_cc, dy_offset, 'CC', 'CC')
                            else :
                                raise ValueError('Invalid subtype name')	
                                
                    else :
                        raise ValueError('Invalid current_point.x_type')
                            
                    idx_y = idx_y_offset
                    cc_subtype = cc_subtype_offset

                else : # not the last element in the lateral direction, thus move to the right
                
                    idx_y += 1
				
                    if current_point.x_type == 'LP' : # current row is lipid
                    
                        if cc_subtype == 0 : # now within dh
                            coord_y += dy_cc
                            if idx_y == self.ny_grids_cc_dn :
                                current_point.setPoint(coord_x, coord_y, dx_lipid, dy_lipid, 'LP', 'LP')
                                idx_y = 0
                                cc_subtype += 1
                            else :
                                current_point.setPoint(coord_x, coord_y, dx_lipid, dy_cc, 'LP', 'LP')
                        elif cc_subtype == 1 : # now within s
                            coord_y += dy_lipid
                            if idx_y == self.ny_grids_lipid :
                                current_point.setPoint(coord_x, coord_y, dx_lipid, dy_cc, 'LP', 'LP')
                                idx_y = 0
                                cc_subtype += 1
                            else :
                                current_point.setPoint(coord_x, coord_y, dx_lipid, dy_lipid, 'LP', 'LP')
                        elif cc_subtype == 2 : # now wtihin dm
                            coord_y += dy_cc
                            if idx_y == self.ny_grids_cc_dn*self.w :
                                current_point.setPoint(coord_x, coord_y, dx_lipid, dy_lipid, 'LP', 'LP')
                                idx_y = 0
                                cc_subtype += 1
                            else :
                                current_point.setPoint(coord_x, coord_y, dx_lipid, dy_cc, 'LP', 'LP')
                        elif cc_subtype == 3 : # now within the 2nd s
                            coord_y += dy_lipid
                            if idx_y == self.ny_grids_lipid :
                                current_point.setPoint(coord_x, coord_y, dx_lipid, dy_cc, 'LP', 'LP')
                                idx_y = 0
                                cc_subtype = 0
                            else :
                                current_point.setPoint(coord_x, coord_y, dx_lipid, dy_lipid, 'LP', 'LP')
                        else :
                            raise ValueError('Invalid cc_subtype')
                        
                    elif current_point.x_type == 'CC' : # current row is corneocyte
                    
                        if cc_subtype == 0 : # now within dh
                            coord_y += dy_cc
                            if idx_y == self.ny_grids_cc_dn :
                                if bOffset :
                                    current_point.setPoint(coord_x, coord_y, dx_cc, dy_lipid, 'CC', 'LP')
                                else :
                                    current_point.setPoint(coord_x, coord_y, dx_cc, dy_lipid, 'CC', 'CC')
                                idx_y = 0
                                cc_subtype += 1
                            else :
                                current_point.setPoint(coord_x, coord_y, dx_cc, dy_cc, 'CC', 'CC')
                        elif cc_subtype == 1 : # now within s
                            coord_y += dy_lipid
                            if idx_y == self.ny_grids_lipid :
                                current_point.setPoint(coord_x, coord_y, dx_cc, dy_cc, 'CC', 'CC')
                                idx_y = 0
                                cc_subtype += 1
                            else :
                                if bOffset :
                                    current_point.setPoint(coord_x, coord_y, dx_cc, dy_lipid, 'CC', 'LP')
                                else :
                                    current_point.setPoint(coord_x, coord_y, dx_cc, dy_lipid, 'CC', 'CC')
                        elif cc_subtype == 2 : # now wtihin dm
                            coord_y += dy_cc
                            if idx_y == self.ny_grids_cc_dn*self.w :
                                if bOffset :
                                    current_point.setPoint(coord_x, coord_y, dx_cc, dy_lipid, 'CC', 'CC')
                                else :
                                    current_point.setPoint(coord_x, coord_y, dx_cc, dy_lipid, 'CC', 'LP')
                                idx_y = 0
                                cc_subtype += 1
                            else :
                                current_point.setPoint(coord_x, coord_y, dx_cc, dy_cc, 'CC', 'CC')
                        elif cc_subtype == 3 : # now within the 2nd s
                            coord_y += dy_lipid
                            if idx_y == self.ny_grids_lipid :
                                current_point.setPoint(coord_x, coord_y, dx_cc, dy_cc, 'CC', 'CC')
                                idx_y = 0
                                cc_subtype = 0
                            else :
                                if bOffset :
                                    current_point.setPoint(coord_x, coord_y, dx_cc, dy_lipid, 'CC', 'CC')
                                else :
                                    current_point.setPoint(coord_x, coord_y, dx_cc, dy_lipid, 'CC', 'LP')
                        else :
                            raise ValueError('Invalid cc_subtype')
                            
                    else :
                        raise ValueError('Invalid current_point.x_type')     
            # for j
        # for i


    def setParDiff_paras(self, Kw_paras, D_paras):
        """ A wrapper interface to pass Kw/D options """
        self.setKwOptions(Kw_paras)
        self.setDOptions(D_paras)
        
        
    def compParDiff_lp(self, chem) :
        """ Compute the partition coefficient with respect to water
        and the diffusion coefficient for lipid   """
        rou_lipid = 0.9
        rou_water = 1               
        
        # Partition
        if self.Kw_paras.lp.option == 'VALE':
            #print(self.Kw_paras.lp.value)
            Kw = self.Kw_paras.lp.value[0]
        else:  # Some form of QSPR calcualtion is needed
            if self.Kw_paras.lp.option == 'DEFT':                
                a = 0.69
            elif self.Kw_paras.lp.option == 'QSPR':
                a = self.Kw_paras.lp.value[0]
            else:
                raise ValueError('Invalid self.Kw_paras.lp.option')
            Kw = rou_lipid / rou_water * (chem.K_ow ** a)
        
            
        # Diffusion
        if self.D_paras.lp.option == 'VALE':
            D = self.D_paras.lp.value[0]
        else:  # Some form of QSPR calcualtion is needed
            if self.D_paras.lp.option == 'DEFT':                
                a = 2 * 1E-9
                b = 0.46            
            elif self.D_paras.lp.option == 'QSPR':
                a, b = self.D_paras.lp.value
            else:
                raise ValueError('Invalid self.D_paras.lp.option')  
                
            if chem.mw <= 380.0:
                r_s_inA = chem.r_s*1e10 # unit in Angstrom
                D = a * np.exp(-b*r_s_inA*r_s_inA)
            else: 
                D = 3 * 1E-13
                
        #print("Klipid",Kw,"Dlipid",D)     
        return (Kw, D)        
        
        
    def compParDiff_cc(self, chem, Dw, r_f, phi_b, theta_b) :
        """ Compute the partition coefficient with respect to water
        and the diffusion coefficient for corneocyte   """
        rou_keratin = 1.37
        rou_water = 1               

        # Partition
        if self.Kw_paras.cc.option == 'VALE':
            K_kw = self.Kw_paras.cc.value[0]
        else:  # Some form of QSPR calcualtion is needed
            if self.Kw_paras.cc.option == 'DEFT':                
                a = 4.2
                b = 0.31
            elif self.Kw_paras.cc.option == 'QSPR':
                a, b = self.Kw_paras.cc.value
            else:
                raise ValueError('Invalid self.Kw_paras.cc.option')
            K_kw = rou_keratin / rou_water * a * (chem.K_ow**b)
            
        Kw = (1-phi_b) * K_kw + theta_b

            
        # Diffusion
        if self.D_paras.cc.option == 'VALE':
            D = self.D_paras.cc.value[0]
        else:  # Some form of QSPR calcualtion is needed
            if self.D_paras.cc.option == 'DEFT':                
                alpha = 9.47
                beta = 9.32e-8             
            elif self.D_paras.cc.option == 'QSPR':
                alpha, beta = self.D_paras.cc.value
            else:
                raise ValueError('Invalid self.D_paras.cc.option')                                
          
            lambdaa = 1.09
            gamma = -1.17
            
            r_s_inA = chem.r_s*1e10 # unit in A
            r_f_inA = r_f*1e10 # unit in A
            
            phi_f = 1 - theta_b
            k = beta*r_f_inA*r_f_inA* (phi_f**gamma)
            S = (r_s_inA+r_f_inA)/r_f_inA
            S = phi_f * S*S
            
            D = np.exp( -alpha*(S**lambdaa) ) / ( 1 + r_s_inA/np.sqrt(k) + r_s_inA*r_s_inA/3/k )
            D *= Dw
        #print("Kcorn",Kw,"Dcorn",D)         
               
        return (Kw, D)

                
    def compParDiff(self, name, chem, mass_frac_water, mass_frac_water_sat, 
                    V_mortar, V_brick, V_all, eta) :
        """ Compute the partition coefficient with respect to water
        and the diffusion coefficient   """
        rou_lipid = 0.9
        rou_keratin = 1.37
        rou_water = 1
        
        # calulcate volume fraction of water in CC based on
        #    mass fraction of water in SC
        theta_b = self.compVolFracWater_cc( mass_frac_water, rou_lipid, rou_keratin, rou_water,
				 V_mortar, V_brick, V_all)
        # do the same for saturated water
        phi_b = self.compVolFracWater_cc( mass_frac_water_sat, rou_lipid, rou_keratin, rou_water,
				 V_mortar, V_brick, V_all)


        # calculate diffusivity and partition coefficient
        
        r_f = 3.5e-9 # keratin microfibril radius, 3.5 nm
        Dw = comp.Comp.compDiff_stokes(self, eta, chem.r_s)
        
        if name == 'LP':            
            Kw, D = self.compParDiff_lp(chem)   
            #print(Kw, D)             
        elif name == 'CC' : # corneocyte
            Kw, D = self.compParDiff_cc(chem, Dw, r_f, phi_b, theta_b)
            
            
        #print(Kw, D)
        return (Kw, D)
        
        
    def compVolFracWater_cc(self, mass_frac_water, rou_lipid, rou_keratin, rou_water, 
                            V_mortar_geometry, V_brick_geometry, V_all_geometry):
        """ Compute the volume fraction of water in corneocyte
            based on the water content (mass fraction of water) of the stratum corneum
        """
        f_l = 0.125 # dry mass fraction of SC lipid and keratin  
        f_k = 1 - f_l
        
        # mass fraction of lipid and keratin
        mass_lipid = (1 - mass_frac_water) * f_l
        mass_keratin = (1 - mass_frac_water) * f_k  

        V_all = mass_lipid/rou_lipid + mass_keratin/rou_keratin + mass_frac_water/rou_water
        V_lipid = mass_lipid/rou_lipid / V_all
        V_keratin = mass_keratin/rou_keratin / V_all  
        
        V_water_mortar = V_mortar_geometry/V_all_geometry - V_lipid
        V_water_brick = V_brick_geometry/V_all_geometry - V_keratin  
        
        vol_frac_water_cc = V_water_brick / V_brick_geometry * V_all_geometry
  
        return vol_frac_water_cc


    def compODEdydt(self, t, y, args=None):
        """ The wrapper function for computing the right hand side of ODEs
        """
        return comp.Comp.compODEdydt_diffu (self, t, y, args)

    
        
    def saveCoord(self, fn_x, fn_y) :
        comp.Comp.saveCoord(self, fn_x, fn_y, '.sc')
        