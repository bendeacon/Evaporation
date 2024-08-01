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
from thermo.unifac import UFSG, UFIP, DOUFSG, DOUFIP2006

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
        Chem_Inchi = search_chemical("Nitrobenzene").InChI_key
        Chem_Assignment = thermo.unifac.DDBST_PSRK_assignments[Chem_Inchi]
 
        
        #print(nWater,nActive)
        # if nWater < 0 or nActive < 0:
        #     Activity_Water = 0
        #     Activity_Active = 0
        # else:
        Activity_Coeff = UNIFAC_gammas(T, [nWater, nActive],[{16:1}, Chem_Assignment],subgroup_data=PSRKSG, interaction_data=PSRKIP, modified=True)
        #print("actitvity_coeff",activity_coeff)
        Activity_Coeff_Water = Activity_Coeff[0]*nWater
        Activity_Coeff_Active = Activity_Coeff[1]*nActive
        #print("S", Activity_Coeff_Water, Activity_Coeff_Active)
        return Activity_Coeff_Water, Activity_Coeff_Active
    
    def dried_vehicle(self, Volume, A, y, Temp, t, args, h):
        print("Dried")
        #print("dried")
        phase = "LIQUID"
        #h = 1e-12
        dx = self.meshes[0].dx
        self.meshes[0].dx = h
        K = self.meshes[0].Kw
        K_vw = self.K_lip_water                
        self.setMeshes_Kw(K_vw)
        Rg = 8.314 #gas constant
        RH = 33 #relative humidity
        ht = 0.02348-(Volume/A) #tube height m
    
        dydt = comp.Comp.compODEdydt_diffu (self, t, y, args)
        flux = dydt[0]*h
        
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
        
        if phase == "SOLID":
            dy0dt =  0 - flux*A#reduction of solute
            dy3dt = 0 #mg/s/cm^2* cm^2 #evaporation of solute #this is good
        
        else: 
            Volume_Active = self.chem.mw/self.rho_solute/scipy.constants.Avogadro #volume in cm3/particle
            Volume_Active = Volume_Active * (1e+21) #volume in nm3/particle
            r_Solute = ( 0.9087 * self.chem.mw * 3/4/np.pi ) ** (1.0/3) * 1e-10
            #r_Solute = ( 0.9087 * self.chem.mw * 3/4/np.pi ) ** (1.0/3)
            P_Solute =  2.66 #Chemical("Acetophenone").VaporPressure.solve_property(1e5)  #0.065 #Vapor pressure of Active Pascal
            D_Solute =  scipy.constants.Boltzmann*Temp/(6*math.pi*0.0000000917*r_Solute)#0.1382e-4 #0.1382e-4 #Diffusion coefficient of Active in air m2/s   
            #=((1.86*10^(-3))*T^(3/2)*(1/M+1/M)^0.5)/(p*(2*r)^2*sigma)
            #D_Solute =  ((1.86*10**(-3))*Temp**(3/2)*(1/self.chem.mw+1/self.chem.mw)**(0.5))/(1*(2*r_Solute)**(2)*1)
            #D_Solute = D_Solute *10**(-4)
            Conc_Solute = P_Solute * (x0) / Rg / Temp
            J_Solute = D_Solute * (Conc_Solute - 0) / ht  
            
                 
            # dy0dt =  (J_Solute  * A * self.rho_solute)/h #reduction of solute
            # dy3dt = (J_Solute * A * self.rho_solute ) #mg/s/cm^2* cm^2 #evaporation of solute #this is good
             
            dy0dt =  -(J_Solute *Volume/A)-flux*A #reduction of solute
            dy3dt = (J_Solute*Volume/A ) #mg/s/cm^2* cm^2 #evaporation of solute #this is good
            
        
    
        self.meshes[0].dx = dx
        self.setMeshes_Kw(K)  
        
        dydt += np.array([dy0dt, 0, 0, dy3dt])
        return dydt
    
    def Evaporation_Solution(self, Volume, A, y, Temp, x1, x0, Name_Solute, flux, h):
        #print("solution")
        #print("were here")
        Rg = 8.314 #gas constant
        RH = 33 #relative humidity
        ht = 0.02348-(Volume/A) #tube height m
        h = y[0]
        Volume_Active = self.chem.mw/self.rho_solute/scipy.constants.Avogadro #volume in cm3/particle
        Volume_Active = Volume_Active * (1e+21) #volume in nm3/particle
        
        #Activity_Coeff_Solvent, Activity_Coeff_Solute = finite_dose_vehicle.Vehicle.Activity(Temp,x1,x0,"Diethanolamine")
        Name_Solvent = "Water"
        
        #Water or Solvent
        if Name_Solvent == "Water":
            PW100 = 0.61094*math.exp(17.625*(Temp-273)/((Temp-273)+243.04))*1000 #Chemical('water').VaporPressure.solve_property(1e5) # #Chemical('water').VaporPressure.solve_property(1e5) #0.61094*math.exp(17.625*Temp/(Temp+243.04))*1000 
            D_Solvent = 22.5E-06 * (Temp/273.15)**(1.8)#Diffusion coefficient of water in air m2/s D = 22.5E-06 * (T/273.15K)^(1.8)   [m*m/s]
            PWRH = PW100 * RH / 100 #Calculating the partial vapor pressure in the environment
            CWRH = PWRH / Rg / Temp #ideal gas concentration
            #Conc_Solvent = PW100 * (Activity_Coeff_Solvent) / Rg / Temp
            Conc_Solvent = PW100 * (x1) / Rg / Temp
            J_Solvent = D_Solvent * (Conc_Solvent-CWRH) / ht
      
        else:
            Volume_Sol= self.mw_solvent/self.rho_solvent/scipy.constants.Avogadro #volume in cm3/particle
            Volume_Sol = Volume_Sol * (1e+21) #volume in nm3/particle
            r_Solvent = ( 0.9087 * self.mw_solvent * 3/4/np.pi ) ** (1.0/3) * 1e-10
            P_Solvent = 6000 #Chemical(Name_Solvent).VaporPressure.solve_property(1e5)  #Vapor pressure of Active Pascal
            D_Solvent =  scipy.constants.Boltzmann*Temp/(6*math.pi*0.0000000917*r_Solvent)#0.1382e-4 #0.1382e-4 #Diffusion
            #D_Solvent = ((1.86*10**(-3))*Temp**(3/2)*(1/self.mw_solvent+1/self.mw_solvent)**(0.5))/(1*(2*r_Solvent)**(2)*1)
            #D_Solvent = D_Solvent *10**(-4)
            Conc_Solvent = P_Solvent * (x1) / Rg / Temp
            #Conc_Solvent = P_Solvent * (Activity_Coeff_Solvent) / Rg / Temp
            J_Solvent = D_Solvent * (Conc_Solvent-0) / ht
        
        #Active Ingredient Ibuprofen
        r_Solute = ( 0.9087 * self.chem.mw * 3/4/np.pi ) ** (1.0/3) * 1e-10
        #r_Solute = ( 0.9087 * self.chem.mw * 3/4/np.pi ) ** (1.0/3)
        P_Solute =  2.66 #Chemical("Acetophenone").VaporPressure.solve_property(1e5)  #0.065 #Vapor pressure of Active Pascal
        D_Solute =  scipy.constants.Boltzmann*Temp/(6*math.pi*0.0000000917*r_Solute)#0.1382e-4 #0.1382e-4 #Diffusion coefficient of Active in air m2/s   
        #=((1.86*10^(-3))*T^(3/2)*(1/M+1/M)^0.5)/(p*(2*r)^2*sigma)
        #D_Solute =  ((1.86*10**(-3))*Temp**(3/2)*(1/self.chem.mw+1/self.chem.mw)**(0.5))/(1*(2*r_Solute)**(2)*1)
        #D_Solute = D_Solute *10**(-4)
        #Conc_Solute = P_Solute * (Activity_Coeff_Solute) / Rg / Temp
        Conc_Solute = P_Solute * (x0) / Rg / Temp
        J_Solute = D_Solute * (Conc_Solute - 0) / ht  #mass per second


        # The original evaporation script
        #Taos equivalent
        dhdt =  (flux+J_Solute)*A - J_Solvent*ht*A #reduction in vehicle thickness 
        t = J_Solute*ht
        dhdt += - t
        dy0dt =  -((t * A /self.rho_solute)+ flux*A - y[0]*dhdt)/h #reduction of solute
        dy1dt = (J_Solvent * ht * A * self.rho_solvent + y[1]*dhdt) #evaporation of solvent
        dy3dt = (t * A * self.rho_solute )/h  #evaporation of solute 
         

        

        dydt = np.array([dy0dt, dy1dt, dhdt, dy3dt])
        #print(dydt)
        return dydt
    
    def Evaporation_Precipitate(self, V, A, y, Temp, x1, x0, Name_Solute, flux, h, V1, V2):
        #print("precipitate")
        Rg = 8.314 #gas constant
        RH = 33 #relative humidity
        ht = 0.02348-(V/A) #tube height m
        
        x3 = V1*self.rho_solute #mass in precipitate
        x4 = V2*self.rho_solute #mass in solution
    
        
        #Activity_Coeff_Solvent, Activity_Coeff_Solute = finite_dose_vehicle.Vehicle.Activity(Temp,x1,x0,"Ethoxycoumarin")
        Name_Solvent = "Water"
        
        #Water or Solvent
        if Name_Solvent == "Water":
            PW100 = 0.61094*math.exp(17.625*(Temp-273)/((Temp-273)+243.04))*1000 #Chemical('water').VaporPressure.solve_property(1e5) # #Chemical('water').VaporPressure.solve_property(1e5) #0.61094*math.exp(17.625*Temp/(Temp+243.04))*1000 
            D_Solvent = 22.5E-06 * (Temp/273.15)**(1.8)#Diffusion coefficient of water in air m2/s D = 22.5E-06 * (T/273.15K)^(1.8)   [m*m/s]
            PWRH = PW100 * RH / 100 #Calculating the partial vapor pressure in the environment
            CWRH = PWRH / Rg / Temp #ideal gas concentration
            #Conc_Solvent = PW100 * (x1*Activity_Coeff_Solvent) / Rg / Temp
            Conc_Solvent = PW100 * (x1) / Rg / Temp
            J_Solvent = D_Solvent * (Conc_Solvent-CWRH) / ht
        else:
            Volume_Sol= self.mw_solvent/self.rho_solvent/scipy.constants.Avogadro #volume in cm3/particle
            Volume_Sol = Volume_Sol * (1e+21) #volume in nm3/particle
            r_Solvent = ( 0.9087 * self.mw_solvent * 3/4/np.pi ) ** (1.0/3) * 1e-10
            P_Solvent = 6000 #Chemical(Name_Solvent).VaporPressure.solve_property(1e5)  #Vapor pressure of Active Pascal
            D_Solvent =  scipy.constants.Boltzmann*Temp/(6*math.pi*0.0000000917*r_Solvent)
            #D_Solvent = ((1.86*10**(-3))*Temp**(3/2)*(1/self.mw_solvent+1/self.mw_solvent)**(0.5))/(1*(2*r_Solvent)**(2)*1)
            #D_Solvent = D_Solvent *10**(-4)
            Conc_Solvent = P_Solvent * (x1) / Rg / Temp
            J_Solvent = D_Solvent * (Conc_Solvent-0) / ht

        #Solute
        Volume_Active = self.chem.mw/self.rho_solute/scipy.constants.Avogadro #volume in cm3/particle
        Volume_Active = Volume_Active * (1e+21) #volume in nm3/particle
        r_Solute = ( 0.9087 * self.chem.mw * 3/4/np.pi ) ** (1.0/3) * 1e-10
        #r_Solute = ( 0.9087 * self.chem.mw * 3/4/np.pi ) ** (1.0/3)
        P_Solute =  2.66 #Chemical("Acetophenone").VaporPressure.solve_property(1e5)  #0.065 #Vapor pressure of Active Pascal
        D_Solute =  scipy.constants.Boltzmann*Temp/(6*math.pi*0.0000000917*r_Solute)#0.1382e-4 #0.1382e-4 #Diffusion coefficient of Active in air m2/s   
        #=((1.86*10^(-3))*T^(3/2)*(1/M+1/M)^0.5)/(p*(2*r)^2*sigma)
        #D_Solute =  ((1.86*10**(-3))*Temp**(3/2)*(1/self.chem.mw+1/self.chem.mw)**(0.5))/(1*(2*r_Solute)**(2)*1)
        #D_Solute = D_Solute *10**(-4)
        D_Solute_Water = scipy.constants.Boltzmann*Temp/(6*math.pi*0.00089*r_Solute)
        Diff_Sol = 1/(1/D_Solute+1/D_Solute_Water)
        #print("Diffusion coeffs", Diff_Sol, D_Solute)
        #Conc_Solute = P_Solute * (x4*Activity_Coeff_Solute) / Rg / Temp
        Conc_Solute = P_Solute * (x4) / Rg / Temp
        Conc_Solute_Precip = P_Solute * (x3) / Rg / Temp
        #print("amounts of solute", Conc_Solute, Conc_Solute_Precip)
        J_Solute_liquid = Diff_Sol * (Conc_Solute_Precip - 0) / ht + D_Solute*(Conc_Solute-0)/ht
        J_Solute_solid = D_Solute*(Conc_Solute-0)/ht
        #J_Solute = Diff_Sol * (Conc_Solute_Precip - 0) / ht + 0 

        # dhdt =  (flux+J_Solute)*A - J_Solvent*ht*A #reduction in vehicle thickness 
        # t = J_Solute*ht
        # dhdt += - t
        # dy0dt =  -((t * A /self.rho_solute)+ flux*A - y[0]*dhdt)/h #reduction of solute
        # dy1dt = (J_Solvent * ht * A * self.rho_solvent + y[1]*dhdt) #evaporation of solvent
        # dy3dt = (t * A * self.rho_solute )/h  #evaporation of solute 
         
        phase = "LIQUID";
        
        if phase == "SOLID":
            dhdt =  (flux+J_Solute_solid)*A - J_Solvent*ht*A #reduction in vehicle thickness 
            t = J_Solute_solid*ht
            dhdt += - t
            dy0dt =  -( t * A/self.rho_solute + flux*A - y[0]*dhdt)/h #reduction of solute
            dy1dt = (J_Solvent * ht * A * self.rho_solvent + y[1]*dhdt) #evaporation of solvent
            dy3dt = (t * A * self.rho_solute )/h  #evaporation of solute 


            dydt = np.array([dy0dt, dy1dt, dhdt, dy3dt])
            
        else:
            dhdt =  (flux+J_Solute_liquid)*A - J_Solvent*ht*A #reduction in vehicle thickness 
            t = J_Solute_liquid*ht
            dhdt += - t
            dy0dt =  -((t * A /self.rho_solute) + flux*A - y[0]*dhdt)/h #reduction of solute
            dy1dt = (J_Solvent * ht * A * self.rho_solvent + y[1]*dhdt) #evaporation of solvent
            dy3dt = (t * A * self.rho_solute )/h  #evaporation of solute 


            dydt = np.array([dy0dt, dy1dt, dhdt, dy3dt])
        return dydt
    
    
    def main(self, params):
        #params = [y, V, t, A, h, args]
        y = params[0]
        V = params[1]
        t = params[2]
        A = params[3]
        args = params[4]
        Name_Solute = "Benzyl Bromide"
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
        #h = 1e-13
        if h < 1e-12:  # nothing left in vehicle, now fix it to be a thin film
            self.vehicle_dried = True #this is the real one
            #self.vehicle_dried = False #setup to see if my code works
        if self.vehicle_dried is True: # nothing left in vehicle, now fix it to be a thin film

            dydt = finite_dose_vehicle.Vehicle.dried_vehicle(self, V, A, y, Temp, t, args, h)
                          
    
            return dydt
    
        V =  A * self.depth_vehicle
        
        # Vehicle could consist of solution, a separate phase of over-saturated solute (either liquid or solid)
        if y[1] < 1e-12:
            y[1] = 0
    
        # V_solu = y[0]*V/self.rho_solute #volume of solute
        # V_solv = y[1]*V/self.rho_solvent #volume of solvent
        # #V_solu_1 = self.solubility*V_solu/(self.rho_solute) #volume of solute possible in solvent
        # V_solu_1 = (self.solubility/1000)*V_solv
        # if V_solu > V_solu_1: # mass out of the solution phase
        #     V1 = V_solu - V_solu_1 # volume of the mass out phase
        #     V2 = V_solv + V_solu_1 # volume of the solution phase
        # else:
        #     V1 = .0
        #     V2 = V
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
    