#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 15:17:39 2023

@author: bendeacon
"""

import importlib, sys
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy 
import math
from scipy.integrate import solve_ivp
from thermo import ChemicalConstantsPackage, PRMIX, CEOSLiquid, CEOSGas, FlashPureVLS
from thermo import ChemicalConstantsPackage, CEOSGas, CEOSLiquid, PRMIX, FlashVL
from thermo.interaction_parameters import IPDB
from thermo.chemical import Chemical
from thermo.chemical import Mixture
from thermo import unifac
import thermo.unifac
from thermo.unifac import UNIFAC_gammas
from thermo.unifac import PSRKIP, PSRKSG, UNIFAC

from core import comp
importlib.reload(comp)
from core import finite_dose_vehicle
importlib.reload(finite_dose_vehicle)

class Vehicle:
    """Class definition for Finite dose consideration of the Vehicle
    which is the delivery vehicle, currently modelled as a homogenised media
    """
    def Activity(T,nWater,nActive,Name_Active):
        #print(nw,nn)
        Water_Assignment = {16: 1}
        
        thermo.unifac.load_group_assignments_DDBST()
        from chemicals import search_chemical
        Chem_Inchi = search_chemical(Name_Active).InChI_key
        Chem_Assignment = thermo.unifac.DDBST_PSRK_assignments[Chem_Inchi]
        
        #print(nWater,nActive)
        # if nWater < 0 or nActive < 0:
        #     Activity_Water = 0
        #     Activity_Active = 0
        # else:
        Activity_Coeff = UNIFAC_gammas(T, [nWater, nActive],[{16:1}, Chem_Assignment],subgroup_data=PSRKSG, interaction_data=PSRKIP, modified=True)
        #print("actitvity_coeff",activity_coeff)
        Activity_Coeff_Water = Activity_Coeff[0]
        Activity_Coeff_Active = Activity_Coeff[1]
        
        return Activity_Coeff_Water, Activity_Coeff_Active
    
    def dried_vehicle(self, Volume, A, y, Temp, t, args):
        print("dried")
        h = 1e-12
        dx = self.meshes[0].dx
        self.meshes[0].dx = h
        K = self.meshes[0].Kw
        K_vw = self.K_lip_water                
        self.setMeshes_Kw(K_vw)
        Rg = 8.314 #gas constant
        RH = 33 #relative humidity
        ht = 0.02348-(Volume/A) #tube height m
    
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
        
        Volume_Active = self.chem.mw/self.rho_solute/scipy.constants.Avogadro #volume in cm3/particle
        Volume_Active = Volume_Active * (1e+21) #volume in nm3/particle
        r_Solute = (Volume_Active*(3/4)/scipy.constants.pi)**(1/3)
        P_Solute = Chemical("Resorcinol").VaporPressure.solve_property(1e5)  #0.065 #Vapor pressure of Active Pascal
        D_Solute =  scipy.constants.Boltzmann*Temp/(6*math.pi*0.0000000917*r_Solute)#0.1382e-4 #0.1382e-4 #Diffusion coefficient of Active in air m2/s   
        Conc_Solute = P_Solute * (x0) / Rg / Temp
        J_Solute = D_Solute * (Conc_Solute - 0) / ht  
        
             
        # dy0dt =  (J_Solute  * A * self.rho_solute)/h #reduction of solute
        # dy3dt = (J_Solute * A * self.rho_solute ) #mg/s/cm^2* cm^2 #evaporation of solute #this is good
         
        dy0dt =  -(J_Solute *Volume/A) #reduction of solute
        dy3dt = (J_Solute*Volume/A ) #mg/s/cm^2* cm^2 #evaporation of solute #this is good
        
        
    
        self.meshes[0].dx = dx
        self.setMeshes_Kw(K)  
        
        dydt += np.array([dy0dt, 0, 0, dy3dt])
        return dydt
    
    def Evaporation_Solution(self, Volume, A, y, Temp, x1, x0, Name_Solute, flux, h):
        print("solution")
        #print("were here")
        Rg = 8.314 #gas constant
        RH = 33 #relative humidity
        ht = 0.02348-(Volume/A) #tube height m
        h = y[0]
        
        Activity_Coeff_Solvent, Activity_Coeff_Solute = finite_dose_vehicle.Vehicle.Activity(Temp,x1,x0,"Resorcinol")
        Name_Solvent = "Water"
        
        #Water or Solvent
        if Name_Solvent == "Water":
            PW100 = 0.61094*math.exp(17.625*(Temp-273)/((Temp-273)+243.04))*1000 #Chemical('water').VaporPressure.solve_property(1e5) # #Chemical('water').VaporPressure.solve_property(1e5) #0.61094*math.exp(17.625*Temp/(Temp+243.04))*1000 
            D_Solvent = 22.5E-06 * (Temp/273.15)**(1.8)#Diffusion coefficient of water in air m2/s D = 22.5E-06 * (T/273.15K)^(1.8)   [m*m/s]
            PWRH = PW100 * RH / 100 #Calculating the partial vapor pressure in the environment
            CWRH = PWRH / Rg / Temp #ideal gas concentration
            Conc_Solvent = PW100 * (x1*Activity_Coeff_Solvent) / Rg / Temp
            #Conc_Solvent = PW100 * (x1) / Rg / Temp
            J_Solvent = D_Solvent * (Conc_Solvent-CWRH) / ht
        else:
            Volume_Sol= self.mw_solvent/self.rho_solvent/scipy.constants.Avogadro #volume in cm3/particle
            Volume_Sol = Volume_Active * (1e+21) #volume in nm3/particle
            r_Solvent = (Volume_Sol*(3/4)/scipy.constants.pi)**(1/3)
            P_Solvent = Chemical(Name_Solvent).VaporPressure.solve_property(1e5)  #Vapor pressure of Active Pascal
            D_Solvent =  scipy.constants.Boltzmann*Temp/(6*math.pi*0.0000000917*r_Solvent)#0.1382e-4 #0.1382e-4 #Diffusion
            Conc_Solvent = P_Solvent * (x1*Activity_Coeff_Solvent) / Rg / Temp
            #Conc_Solvent = P_Solvent * (x1) / Rg / Temp
            J_Solvent = D_Solvent * (Conc_Solvent-0) / ht
        
        #Solute
        Volume_Active = self.chem.mw/self.rho_solute/scipy.constants.Avogadro #volume in cm3/particle
        Volume_Active = Volume_Active * (1e+21) #volume in nm3/particle
        r_Solute = (Volume_Active*(3/4)/scipy.constants.pi)**(1/3)
        P_Solute = Chemical("Resorcinol").VaporPressure.solve_property(1e5) #0.065 # #Vapor pressure of Active Pascal
        D_Solute =  scipy.constants.Boltzmann*Temp/(6*math.pi*0.0000000917*r_Solute)#4.33E-12 0.1382e-4 #0.1382e-4 #Diffusion coefficient of Active in air m2/s   
        Conc_Solute = P_Solute * (x0*Activity_Coeff_Solute) / Rg / Temp
        #Conc_Solute = P_Solute * (x0) / Rg / Temp
        J_Solute = D_Solute * (Conc_Solute - 0) / ht  #mass per second
        
        # The original evaporation script
        # dhdt = flux/self.rho_solute - self.k_evap_solvent*x1 #reduction in vehicle thickness
        # t = self.k_evap_solute * x0
                
        # dhdt += - self.k_evap_solute * x0   
        # dy0dt = ( - self.k_evap_solute * x0 *self.rho_solute + flux - y[0]*dhdt ) / h #reduction of solute 
        # dy1dt = ( -self.rho_solvent*self.k_evap_solvent*x1 - y[1]*dhdt ) / h #evaporation of solvent
        # dy3dt = self.k_evap_solute * x0 * self.rho_solute * A #evaporation of solute
            
        #density is mass/cm3
        #The evaporation script written by me attempt 2
        dhdt =  (flux)/self.rho_solute/h - J_Solvent/self.rho_solvent/h #reduction in vehicle thickness this is good too
        t = J_Solute
        dhdt += - t/self.rho_solute/h
        dy0dt =  ((J_Solute*self.rho_solute/(Volume/A))+ flux*self.rho_solute/h)#reduction of solute
        dy1dt = (-J_Solvent*Volume/A- y[1]*dhdt) #evaporation of solvent
        dy3dt = (J_Solute*(Volume/A)) #mg/s/cm^2* cm^2 #evaporation of solute #this is good
        
        #attempt 1
        # dhdt =  (flux+J_Solute)/self.rho_solute - J_Solute/self.rho_solvent #reduction in vehicle thickness this is good too
        # t = J_Solute*ht
        # dhdt += - t 
        # dy0dt =  ((t * self.rho_solute)+ flux*self.rho_solute)#reduction of solute
        # dy1dt = (J_Solvent * Volume/A * self.rho_solvent ) #evaporation of solvent
        # dy3dt = (t * self.rho_solute ) #mg/s/cm^2* cm^2 #evaporation of solute #this is good
        
        #Theoretically equivilent evaporation script written by me
        # dhdt =  (flux+J_Solute)*A - J_Solvent*ht*A #reduction in vehicle thickness 
        # t = J_Solute*htcd 
        # dhdt += - t
        # dy0dt =  -((t * A /self.rho_solute)+ flux*A - y[0]*dhdt)/h #reduction of solute
        # dy1dt = (J_Solvent * ht * A * self.rho_solvent + y[1]*dhdt) #evaporation of solvent
        # dy3dt = (t * A * self.rho_solute )/h  #evaporation of solute 
        
        # dhdt =  (flux+J_Solute)/self.rho_solute - J_Solvent*ht #reduction in vehicle thickness 
        # t = J_Solute*ht
        # dhdt += - t 
        # dy0dt =  ((t * A *self.rho_solute)+ flux - y[0]*dhdt)/h #reduction of solute
        # dy1dt = -(J_Solvent * ht * A * self.rho_solvent + y[1]*dhdt) #evaporation of solvent
        # dy3dt = (t * A * self.rho_solute )/h  #evaporation of solute 
        

        dydt = np.array([dy0dt, dy1dt, dhdt, dy3dt])
        print(dydt)
        return dydt
    
    def Evaporation_Precipitate(self, V, A, y, Temp, x1, x0, Name_Solute, flux, h, V1, V2):
        print("precipitate")
        Rg = 8.314 #gas constant
        RH = 33 #relative humidity
        ht = 0.02348-(V/A) #tube height m
        
        x3 = (V1*self.rho_solute/V)/x0
        x4 = (V2*self.rho_solute/V)/x0
    
        
        Activity_Coeff_Solvent, Activity_Coeff_Solute = finite_dose_vehicle.Vehicle.Activity(Temp,x1,x0,"Resorcinol")
        Name_Solvent = "Water"
        
        #Water or Solvent
        if Name_Solvent == "Water":
            PW100 = 0.61094*math.exp(17.625*(Temp-273)/((Temp-273)+243.04))*1000 #Chemical('water').VaporPressure.solve_property(1e5) # #Chemical('water').VaporPressure.solve_property(1e5) #0.61094*math.exp(17.625*Temp/(Temp+243.04))*1000 
            D_Solvent = 22.5E-06 * (Temp/273.15)**(1.8)#Diffusion coefficient of water in air m2/s D = 22.5E-06 * (T/273.15K)^(1.8)   [m*m/s]
            PWRH = PW100 * RH / 100 #Calculating the partial vapor pressure in the environment
            CWRH = PWRH / Rg / Temp #ideal gas concentration
            Conc_Solvent = PW100 * (x1*Activity_Coeff_Solvent) / Rg / Temp
            J_Solvent = D_Solvent * (Conc_Solvent-CWRH) / ht
        else:
            Volume_Sol= self.mw_solvent/self.rho_solvent/scipy.constants.Avogadro #volume in cm3/particle
            Volume_Sol = Volume_Active * (1e+21) #volume in nm3/particle
            r_Solvent = (Volume_Sol*(3/4)/scipy.constants.pi)**(1/3)
            P_Solvent = Chemical(Name_Solvent).VaporPressure.solve_property(1e5)  #Vapor pressure of Active Pascal
            D_Solvent =  scipy.constants.Boltzmann*Temp/(6*math.pi*0.0000000917*r_Solvent)#0.1382e-4 #0.1382e-4 #Diffusion
            Conc_Solvent = P_Solvent * (x1*Activity_Coeff_Solvent) / Rg / Temp
            J_Solvent = D_Solvent * (Conc_Solvent-0) / ht

        #Solute
        Volume_Active = self.chem.mw/self.rho_solute/scipy.constants.Avogadro #volume in cm3/particle
        Volume_Active = Volume_Active * (1e+21) #volume in nm3/particle
        r_Solute = (Volume_Active*(3/4)/scipy.constants.pi)**(1/3)
        P_Solute = Chemical("Resorcinol").VaporPressure.solve_property(1e5)  #Vapor pressure of Active Pascal
        D_Solute =  scipy.constants.Boltzmann*Temp/(6*math.pi*0.0000000917*r_Solute)#0.1382e-4 #0.1382e-4 #Diffusion coefficient of Active in air m2/s   
        D_Solute_Water = scipy.constants.Boltzmann*Temp/(6*math.pi*0.00089*r_Solute)
        Diff_Sol = 1/(1/D_Solute+1/D_Solute_Water)
        Conc_Solute = P_Solute * (x4*Activity_Coeff_Solute) / Rg / Temp
        Conc_Solute_Precip = P_Solute * (x3) / Rg / Temp
        J_Solute = Diff_Sol * (Conc_Solute_Precip - 0) / ht + D_Solute*(Conc_Solute-0)/ht 

        # dhdt =  (flux+J_Solute)/self.rho_solute - J_Solvent*ht #reduction in vehicle thickness 
        # t = J_Solute*ht
        # dhdt += - t 
        # dy0dt =  ((t * A *self.rho_solute)+ flux - y[0]*dhdt)/h #reduction of solute
        # dy1dt = -(J_Solvent * ht * A * self.rho_solvent + y[1]*dhdt) #evaporation of solvent
        # dy3dt = (t * A * self.rho_solute )/h  #evaporation of solute 
        
        dhdt =  (flux+J_Solute) + J_Solvent #reduction in vehicle thickness this is good too
        t = J_Solute
        dhdt += - t
        dy0dt =  ((t*V/A)+ flux*self.rho_solute)#reduction of solute
        dy1dt = (-J_Solvent*V/A- y[1]*dhdt) #evaporation of solvent
        dy3dt = (t*V/A) #mg/s/cm^2* cm^2 #evaporation of solute #this is good
        dydt = np.array([dy0dt, dy1dt, dhdt, dy3dt])
        return dydt
    
    
    def main(self, params):
        #params = [y, V, t, A, h, args]
        y = params[0]
        V = params[1]
        t = params[2]
        A = params[3]
        args = params[4]
        Name_Solute = "Ibuprofen"
        Temp = 303.15
        
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

            dydt = finite_dose_vehicle.Vehicle.dried_vehicle(self, V, A, y, Temp, t, args)
                          
    
            return dydt
    
        V =  A * self.depth_vehicle
        
        # Vehicle could consist of solution, a separate phase of over-saturated solute (either liquid or solid)
        if y[1] < 1e-12:
            y[1] = 0
    
        V_solu = y[0]*V/self.rho_solute
        V_solv = y[1]*V/self.rho_solvent
        V_solu_1 = self.solubility*V_solv/(self.rho_solute-self.solubility)
        if V_solu > V_solu_1: # mass out of the solution phase
            V1 = V_solu - V_solu_1 # volume of the mass out phase
            V2 = V_solv + V_solu_1 # volume of the solution phase
            #print(V1, V2)
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
        flux = dydt[0]*self.depth_vehicle
        
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
        
        if V_solu > V_solu_1: # mass out of the solution phase
            dydt = finite_dose_vehicle.Vehicle.Evaporation_Precipitate(self, V, A, y, Temp, x1, x0, Name_Solute, flux, h, V1, V2)
        else:
            dydt = finite_dose_vehicle.Vehicle.Evaporation_Solution(self, V, A, y, Temp, x1, x0, Name_Solute, flux, h)
      
        
        self.setMeshes_Kw(K)
        self.meshes[0].dx = dx
        #print('dydt=', dydt)
        #sys.exit() 
        return dydt
    