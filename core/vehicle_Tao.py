# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 13:28:52 2017

@author: tc0008
"""
import importlib, sys
import numpy as np

from core import comp
importlib.reload(comp)


class Vehicle(comp.Comp):
    """Class definition for Vehicle
    which is the delivery vehicle, currently modelled as a homogenised media
    """
    
    def __init__(self, chem, xlen, ylen, dz_dtheta, nx, ny, init_conc, Kw, D,
                 coord_sys, bdy_cond, b_inf_source=False,
                 rho_solute=1e3, rho_solvent=1e3, mw_solvent=18, phase_solute='LIQUID',
                 k_evap_solvent=0, k_evap_solute=0, solubility=1e10):
        comp.Comp.__init__(self)
        comp.Comp.setup(self, xlen, ylen, dz_dtheta, nx, ny, coord_sys, bdy_cond)
        
        self.eta = 7.644E-4 # water viscosity at 305 K (32 deg C) (Pa s)
        self.b_inf_source = b_inf_source
        
        self.chem = chem
        
        # evaporative mass transfer coefficient for solvent and solute
        self.b_vary_vehicle = False #True
        self.k_evap_solvent = k_evap_solvent
        self.k_evap_solute = k_evap_solute
        self.solubility = solubility
        self.rho_solute = rho_solute
        self.rho_solvent = rho_solvent
        self.mw_solvent = mw_solvent
        self.phase_solute = phase_solute
        self.K_lip_water = None
        self.vehicle_dried = False
        
        #self.conc_solvent = rho_solvent
        A = self.compTotalArea(3)
        V =  A * xlen
        V_solute = V*init_conc/rho_solute
        self.conc_solvent =  (V-V_solute)*rho_solvent / V

        #print(V, V_solute, self.conc_solvent, init_conc)
        #sys.exit()
        
        self.depth_vehicle = xlen
        self.mass_out_evap = 0
        self.mass_out_phase = 0
        
                
        self.init_conc = init_conc
                
        comp.Comp.set_Kw(self, Kw)
        comp.Comp.set_D(self, D)
        
    def getMass_OutEvap(self):
        return self.mass_out_evap
    def getMass_OutPhase(self):
        return self.mass_out_phase
        
    def createMesh(self, chem, coord_x_start, coord_y_start) :
        """ Create mesh for this compartment
        Args:
                coord_x_start, coord_y_start: starting coordinates
        """
        self.compParDiff(chem)
        comp.Comp.createMeshHomo(self, 'VH', chem, self.init_conc, coord_x_start, coord_y_start)
        
        
    def compParDiff(self, chem) :
        """ Compute the partition coefficient with respect to water
        and the diffusion coefficient
        """
        if self.Kw < 0:
            Kw = 1 # caution: only placeholder and needs refining
            comp.Comp.set_Kw(self, Kw)
        
                    
        if self.D < 0: # calculation of diffusivity according to the Stoke-Eistein equation
            D = comp.Comp.compDiff_stokes(self, self.eta, chem.r_s)
            comp.Comp.set_D(self, D)
        
        
        #return (Kw, D)
                

    def compODEdydt(self, t, y, args=None):
        """ The wrapper function for computing the right hand side of ODEs
        """
        if self.b_vary_vehicle is False:
            dydt = comp.Comp.compODEdydt_diffu (self, t, y, args)
            
            # If infinite source, concentration doesn't change, but above dydt calculation 
            #   is still needed since calling compODEdydt_diffu will calculate the 
            #   flux across boundaries properly            
            if self.b_inf_source :
                dydt.fill(0)            

        else :
            # Evaporation of solvent and solute
            #   Currently a crude approximation and only implemented for a homogeneous vehicle compartment                
            assert(self.nx==1 and self.ny==1)
                        
            # y[0] - solute conc, y[1] - solvent conc, y[2] - h,
            # y[3] - solute mass out due to evarporation
            dim = self.nx*self.ny
            A = self.compTotalArea(3)            
            
            if y[0] < 0:
                y[0] = 0            
            
            h = y[2]  # vehicle thickness
            if h < 1e-12:  # nothing left in vehicle, now fix it to be a thin film
                self.vehicle_dried = True
            if self.vehicle_dried is True: # nothing left in vehicle, now fix it to be a thin film
                h = 1e-12
                dx = self.meshes[0].dx
                self.meshes[0].dx = h
                K = self.meshes[0].Kw
                K_vw = self.K_lip_water                
                self.setMeshes_Kw(K_vw)

                dydt = comp.Comp.compODEdydt_diffu (self, t, y, args)
                
                # mole fractions
                mw_lipid = 566 # Biophys J. 2007 Nov 1; 93(9): 3142–3155
                rho_lipid = 900
                x0 = y[0] / self.chem.mw
                x1 = rho_lipid / mw_lipid
                total = x0+x1
                x0 /= total                
                x1 /= total
                #if t>3600:
                #    print (y[0], self.chem.mw, rho_lipid, mw_lipid, x0, x1)
                #    sys.exit()
                     
                dy0dt = ( -self.k_evap_solute * x0 * self.rho_solute ) / h
                dy3dt = self.k_evap_solute * x0 * self.rho_solute * A
                
                dydt += np.array([dy0dt, 0, 0, dy3dt])
                
                self.meshes[0].dx = dx
                self.setMeshes_Kw(K)                

                return dydt

            V =  A * h
            
            # Vehicle could consist of solution, a separate phase of over-saturated solute (either liquid or solid)
            if y[1] < 1e-12:
                y[1] = 0

            V_solu = y[0]*V/self.rho_solute
            V_solv = y[1]*V/self.rho_solvent
            V_solu_1 = self.solubility*V_solv/(self.rho_solute-self.solubility)
            if V_solu > V_solu_1: # mass out of the solution phase
                V1 = V_solu - V_solu_1 # volume of the mass out phase
                V2 = V_solv + V_solu_1 # volume of the solution phase
                print(V1, V2)
            else:
                V1 = .0
                V2 = V
            # Update partiton coefficient for vehicle that could have two phases
            P1 = self.rho_solute/self.solubility
            P2 = self.Kw
            K_vw = P1*V1/(V1+V2) + P2*V2/(V1+V2)
            
            # Set mesh parameters 
            K = self.meshes[0].Kw    
            self.setMeshes_Kw(K_vw)
            dx = self.meshes[0].dx
            self.meshes[0].dx = h
            #if t > 1442:
            #    print(K, K_vw, dx, h)
                #sys.exit()
            #     and calculating diffusion mass flux
            dydt = comp.Comp.compODEdydt_diffu (self, t, y, args)
            flux = dydt[0]*V/A
            
            # mole fractions            
            x0 = y[0] / self.chem.mw
            x1 = y[1] / self.mw_solvent
            total = x0+x1
            if total < 1e-12: # no evaporation
                x0 = 0
                x1 = 0
            else:
                x0 /= total
                x1 /= total
            
            # Here we calculate reduction of vehicle due to evaporation (both solvent and solute)
            #   and due to solute diffusion into skin.
            #   We assume solvent doesn't diffuse into skin
                        
            dhdt = flux/self.rho_solute - self.k_evap_solvent*x1
            t = self.k_evap_solute * x0
                
            dhdt += - t    
            dy0dt = ( -t*self.rho_solute + flux - y[0]*dhdt ) / h
            dy3dt = t * self.rho_solute * A
            
            dy1dt = ( -self.rho_solvent*self.k_evap_solvent*x1 - y[1]*dhdt ) / h
            
            
            dydt = np.array([dy0dt, dy1dt, dhdt, dy3dt])
            
            self.setMeshes_Kw(K)
            self.meshes[0].dx = dx
            #print('dydt=', dydt)
            #sys.exit()           
        
        return dydt
        
    def saveCoord(self, fn_x, fn_y) :
        comp.Comp.saveCoord(self, fn_x, fn_y, '.vh')
        
        
    def getMeshConc(self) :
        """ This function name is a misnomer but meant to be consistent with 
        the same function in class comp
        The function returns the concentration from all meshes
        AND also the variables due to varying vehicle
        into a single numpy array
        """
        if self.b_vary_vehicle is False:
            return comp.Comp.getMeshConc(self)
        else:
            dim = self.dim
            y = np.zeros(self.get_dim())
            y[:dim] = comp.Comp.getMeshConc(self)
            y[dim:] = [self.conc_solvent, self.depth_vehicle,\
                       self.mass_out_evap]
            return y

    def setMeshConc_all(self, conc) :
        """ Similar to the above, this function name is a misnomer but meant to be 
        consistent with the same function in class comp
        The function sets the concentration for all meshes
        AND also the variables due to varying vehicle
        """
        if self.b_vary_vehicle is False:
            comp.Comp.setMeshConc_all(self,conc)
        else:
            if conc[2]<0:
                conc[2] = 1e-12
                self.vehicle_dried = True
            conc[np.where( conc<0 )] = 0
            dim = self.dim
            comp.Comp.setMeshConc_all(self,conc[:dim])
            self.conc_solvent, self.depth_vehicle, \
                self.mass_out_evap = conc[dim:]
                        
            assert(self.nx==1 and self.ny==1)
            self.x_length = self.depth_vehicle
            self.meshes[0].dx = self.x_length

    def get_dim(self):
        if self.b_vary_vehicle is False:
            return comp.Comp.get_dim(self)
        else:
            return comp.Comp.get_dim(self)+3

    def saveMeshConc(self, b_1st_time, fn) :
        """ Save mesh concentrations to file
        Args: b_1st_time -- if True, write to a new file; otherwise append to the existing file
        """
        comp.Comp.saveMeshConc(self, b_1st_time, fn)
        if self.b_vary_vehicle is True:            
            file = open(fn, 'a')
            file.write( "{:.6e}\n".format( self.conc_solvent ) )
            file.write( "{:.6e}\n".format( self.depth_vehicle ) )
            file.write( "{:.6e}\n".format( self.mass_out_evap ) )            
            file.close()
            