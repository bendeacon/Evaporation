# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:57:38 2017

@author: tc0008
"""
import numpy as np
import importlib
#import scipy as sp
from scipy.integrate import ode

#import chemical
from core import comp
importlib.reload(comp)

class Skin:
    """ Class definition for Skin
    which can contain a number of compartments
    """
    
    def __init__(self, coord_sys = None, dz_dtheta = None, bdy_cond = None):
        """      """        
        self.comps = None # List of compartments
        self.dz_dtheta = 0.01
        self.coord_sys = 'Cartesian'
        self.nxComp = 0
        self.nyComp = 0
        self.dim_all = 0
        self.nSpecies = 0
        
    def createComps(self, nrow, ncol):
        self.nxComp = nrow
        self.nyComp = ncol
        self.comps = [comp.Comp() for i in range(nrow*ncol)]
        
    def set_n_species(self, n):
        self.nSpecies = n
    def set_dim_all(self, dim):
        self.dim_all = dim
    def get_dim_all(self):
        return self.dim_all
    def setComp(self, comp, idx_comp_x, idx_comp_y) :
        self.comps[ idx_comp_x*self.nyComp + idx_comp_y ] = comp
    def getComp(self, idx_comp_x, idx_comp_y) :
        return self.comps[ idx_comp_x*self.nyComp + idx_comp_y ]
        
    ### (START OF) Class methods dealing with ODE computation ###
    
    def compODEdydt (self, t, y, args=None):
        """Compute the right-hand side of the ODEs, i.e. dydt
        """
        f = np.zeros(y.shape) #
        #print(y)
        
        for k in range(self.nSpecies) :
            idx = k*self.dim_all
            
            ## 1. reset mass transfer between compartments
            for i in range(self.nxComp) :
                for j in range(self.nyComp) :
                    idx_comp = i*self.nyComp+j
                    current_comp = self.comps[idx_comp]
                    current_comp.setBdyMassInOutZero()
                    
            ## 2. Calculate the right hand side of the differential equations
            for i in range(self.nxComp) :
                for j in range(self.nyComp) :
                    idx_comp = i*self.nyComp+j
                    current_comp = self.comps[idx_comp]
                    current_dim = current_comp.get_dim()
                    current_y = y[idx:idx+current_dim]
                    #print('current_y\n', current_y)

                    # 2.1. Update the concentration in boundary meshes according to y[]                    
                    idx_compBdyRight, concBdyRight = self.getBdyRight(current_comp, y, idx, i, j)
                    idx_compBdyDown, concBdyDown = self.getBdyDown(current_comp, y, idx, i, j)
                    current_comp.setBdyConc(concBdyRight, concBdyDown)
                    
                    # 2.2. Call the compartment specific ODE functions
                    #   for certain compartments special treatment is needed
                    #if type(current_comp).__name__ == 'Dermis' :
                        #if (self.b_has_blood) # y[ k*m_dim_all+m_dim_all-1 ] contains the blood concentration, i.e. the last term in the differential equations
                        #((Dermis *)pComp)->updateBlood( y[ k*m_dim_all+m_dim_all-1 ] );
                    tmp_f = current_comp.compODEdydt(t, current_y)
                    f[idx:idx+current_dim] = tmp_f
                    #print('f\n', f)

                    self.passBdyMassOut(current_comp, idx_compBdyRight, idx_compBdyDown)
                    idx += current_dim
                # for j
            # for i
            
            # Simulation is for a small skin area, but needs to multiple
            #  the mass transport due to blood flow by the actual topical application area
            #if (m_b_has_blood){
          
            #double factor = m_Vehicle_area / m_Dermis[k].compTotalArea(0);

            #todo: when more than one dermis compartments are involved, need to collate mass in-out of all dermis compartments

            #m_Blood[k].updateMassInOutDermis(m_Dermis[k].m_mass_into_dermis, m_Dermis[k].m_mass_outof_dermis, factor);
            #m_Blood[k].compODE_dydt(t, y+idx, f+idx);
        # for k
        
        return f
    
    def solveMoL(self, t_start, t_end) :
        """ Solving PDE using method of lines (MoL)
        """
  
        ## get current concentration, and set as initial conditions
        y0 = np.zeros(self.dim_all*self.nSpecies)
        for k in range(self.nSpecies) :
            idx = k*self.dim_all
            
            for i in range(self.nxComp) :
                for j in range(self.nyComp) :
                    idx_comp = i*self.nyComp+j
                    current_comp = self.comps[idx_comp]
                    dim = current_comp.get_dim()
                    y0[ idx : idx+dim ] = current_comp.getMeshConc()
                    idx += current_comp.get_dim()
            #if (m_b_has_blood)
            #  m_Blood[k].getGridsConc(y+idx, m_Blood[k].m_dim);
        # for k, each species

        ## Integration
        
        r = ode(self.compODEdydt).set_integrator('vode', method='bdf', 
            nsteps=5000, with_jacobian=True,atol=1e-10,rtol=1e-8)
        r.set_initial_value(y0, t_start).set_f_params(None)
        r.integrate( r.t + t_end-t_start )
        
        ## Extract calculated values 
        for k in range(self.nSpecies) :
            idx = k*self.dim_all
            
            for i in range(self.nxComp) :
                for j in range(self.nyComp) :
                    idx_comp = i*self.nyComp+j
                    current_comp = self.comps[idx_comp]
                    dim = current_comp.get_dim()
                    current_comp.setMeshConc_all( r.y[ idx : idx+dim ] )
                    idx += current_comp.get_dim()
            #    if (m_b_has_blood)
            # m_Blood[k].setGridsConc(pNVs+idx, m_Blood[k].m_dim);
        # for k, each species
    
    ### (END OF) Class methods dealing with ODE computation ###
    
    
    ### (START OF) Class methods dealing with boundaries ###
    
    def getBdyRight(self, compThis, y, idx_up2now, cIdx_i, cIdx_j):
        """ Get the boundary to the right of this compartment
        using values in <y>
        """
        if (cIdx_j == self.nyComp-1) : # rightmost compartment, no right boundary
            assert( compThis.get_bdyCond(2) != 'FromOther' )
            #compBdyRight = None
            idx_compBdyRight = None
            size = 0            
            concBdyRight = None
        else :            
            idx_compBdyRight = cIdx_i*self.nyComp + cIdx_j + 1
            compBdyRight = self.comps[ idx_compBdyRight ]
            size = compBdyRight.get_nx()            

            idx = idx_up2now + compThis.get_dim()
            concBdyRight = np.zeros(size)
            for i in range(size):
                concBdyRight[i] = y[ idx + i*compBdyRight.get_ny() ]            
            
        return (idx_compBdyRight, concBdyRight)
    
        
    def getBdyDown(self, compThis, y, idx_up2now, cIdx_i, cIdx_j):
        """ Get the boundary to the down of this compartment
        using values in y[]
        """
        if cIdx_i == self.nxComp-1 : # downmost compartment, no down boundary
            assert( compThis.get_bdyCond(3) != 'FromOther' )
            #compBdyDown = None
            idx_compBdyDown = None
            size = 0
            concBdyDown = None
        else :
            idx_compBdyDown = (cIdx_i+1)*self.nyComp + cIdx_j
            compBdyDown = self.comps[ idx_compBdyDown ]
            size = compBdyDown.get_ny()
            
            # work out the index for the down boundary
            idx = idx_up2now
            
            i = cIdx_i
            j = cIdx_j
            while j < self.nyComp :
                idx += self.comps[ i*self.nyComp + j ].get_dim()
                j += 1
                
            i = cIdx_i+1
            j = 0
            while j < cIdx_j :
                idx += self.comps[ i*self.nyComp + j ].get_dim()
                j += 1
            
            # Fill in concBdyDown from y[]
            concBdyDown = np.zeros(size)
            for j in range(size):
                concBdyDown[j] = y[ idx + j ]            
            
        return (idx_compBdyDown, concBdyDown)   
           
    def passBdyMassOut(self, compThis, idx_bdyRight, idx_bdyDown):
        """Pass the outflow mass into the two boundary meshes
        idx_bdyRight, idx_bdyDown -- index of the boundary classes Comp
        """
        if idx_bdyRight is not None :
            self.comps[idx_bdyRight].setMassIn_left(compThis.massOut_right)
            #print(compThis.massOut_right, self.comps[idx_bdyRight].massIn_left)
        if idx_bdyDown is not None:
            self.comps[idx_bdyDown].setMassIn_up(compThis.massOut_down)
            #print(compThis.massOut_down, self.comps[idx_bdyDown].massIn_up)
            
    ### (END OF) Class methods dealing with boundaries ###
    
    ### (START OF) Class methods for post processing / computation ###
    
    def compFlux(self, idx_compThis, direction):
        """Compute flux, i.e mass transfer rate per unit surface area,
        from this compartment (indexed by idx_compThis) to a neighbour (or boundary)
        Args:
            idx_compThis: [i, j], a list indicating this compartment
            direction: 0 - up, 1 - left, 2 - right, 3 - down
        Returns:
            flux, area
        """
        i = idx_compThis[0]
        j = idx_compThis[1]
        idx_comp = i*self.nyComp+j
        compThis = self.comps[idx_comp]

        b_cross_bdy = False
        
        if direction == 0: # up
            if i == 0 : # already at top
                b_cross_bdy = True
            else :
                idx_other = (i-1)*self.nyComp+j
                
        elif direction == 1: # left
            if j == 0: # already left-most
                b_cross_bdy = True
            else : 
                idx_other = idx_comp-1
                
        elif direction == 2: # right
            if j == self.nyComp-1: # already right-most
                b_cross_bdy = True
            else : 
                idx_other = idx_comp+1
                
        elif direction == 3: # down
            if i == self.nxComp-1: # already at bottom
                b_cross_bdy = True
            else : 
                idx_other = (i+1)*self.nyComp+j
        else :
            raise ValueError('Invalid direction')
            
        if b_cross_bdy :
            return compThis.compFluxBdy(direction)
        else :
            compOther = self.comps[idx_other]
            flux, area = self.compFlux_btw_comps(compThis, compOther, direction)
            return [flux, area]
            
            
    def compFlux_btw_comps(self, compThis, compOther, direction):
        """Computer the flux from compThis to compOther
        """
        compThis.setBdyMassInOutZero()
        
        if direction == 3: # down is compOther
            concOther = compOther.getMeshConc()
            concBdy = concOther[0:compOther.get_ny()]
            compThis.setBdyConc(None, concBdy)

            idx = (compThis.get_nx()-1)*compThis.get_ny()
            mass_tf_rate = 0
            for j in range(compThis.get_ny()):
                mass_tf_rate += \
                compThis.compMassIrregMeshDown( compThis.meshes[idx+j], compThis.meshes[idx+j].getConc() )
                #print('mass_tf_rate = {:.6e}', mass_tf_rate)
                                
            area = compThis.compTotalArea(direction)
            flux = mass_tf_rate / area
            return [flux, area]

        else :
            raise ValueError('TODO: direction value not implemented yet')
            
    def compMass_comps(self) :
        """Compute the total mass in compartments
        The computed mass is stored both in self.comps[].mass and as return from this function
        """
        mass = np.zeros(self.nxComp*self.nyComp)
        for i in range(self.nxComp) :
            for j in range(self.nyComp) :
                idx_comp = i*self.nyComp+j
                current_comp = self.comps[idx_comp]
                mass[idx_comp] = current_comp.compTotalMass()

        return mass
        
        
    ### (START OF) Class methods for post processing / computation ###