# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 10:34:18 2017

@author: tc0008
"""

import importlib
import numpy as np

from core import mesh
importlib.reload(mesh)
from core import point
importlib.reload(point)


class Comp:
    """Class definition for Comp
    which are compartments for modelling dermato-pharmacokinetics.
    This is the parent class that contains computational meshes.
    From this class the compartments of stratum corneum, viable epidermis,
    dermis, blood, hair follicle, sebum layer on top of skin, vehicle etc.
    will be derived as daughter classes."""

    def __init__(self, T=305):
        """ A generic method to create the instance """
        self.coord_sys = None        
        self.x_length = 0  # compartment size in the x (verticle, starting from top) and y (lateral, starting from left) directions
        self.y_length = 0  #    if cylindrical coordinate (axisymmetric), y starts from centre to right
        self.dz_dtheta = 0  # compartment size in the z (width) direction, either dz (Cartesian, metre) or dtheta (Cylindrical, degree)
  
        self.nx = 0 # number of meshes at x and y directions of the compartment
        self.ny = 0 #   only support rectangular meshes
        self.dim = 0
        
        self.chem = None # the solute chemical
        
        ### partition and diffusion coefficients ###
        self.Kw = 0
        self.D = 0
        
        ### Parameters used in the QSPR models for calculating Kw & D
        self.Kw_paras = None
        self.D_paras = None
        
        self.T = T # default is 305 (32 deg C), temperature in Kelvin
        self.mass = 0 # total mass in this compartment
        
        ### compartment boundary parameters ###
        
        self.n_meshBdyRight = 0 # number of meshes of the (other) compartment that connects with
        self.n_meshBdyDown = 0  #    the right and down boundaries of 'self'
        
        # allowable options: 'ZeroFlux', 'ZeroConc', 'Periodic', 'FromOther', 'ODE'
        #   order: up, left, right, down
        self.bdyCond = ['ZeroFlux']*4
        
        self.meshBdyRight = None # meshes of the 2 boundaries for which OUTGOING flux is calculated
        self.meshBdyDown = None
        
        # To avoid duplicate calculations, only record mass-in from up and left
        #   and mass-out to right and down
        self.massIn_up = None 
        self.massIn_left = None
        self.massOut_right = None
        self.massOut_down = None
        
        self.meshSink = None # sink with zero concentration
        self.meshes = None # The main meshes

        # Ad hoc settings for solids
        self.hasSolid = False
        self.massSolid = 0
        
    def setup(self, xlen, ylen, dz_dtheta, nx, ny, coord_sys, bdy_cond):       
        """Define instance variables and assign default values.
        Geometric parameters will be setup at the compartment-specific initialisation
        Thus here most default values are set to non-meaning values
        """
        
        ### geometric parameters ###
        
        # allowable options: 'Cartesian', 'Cylindrical'
        self.coord_sys = coord_sys if coord_sys is not None else 'Cartesian' 
        
        self.x_length = xlen  # compartment size in the x (verticle, starting from top) and y (lateral, starting from left) directions
        self.y_length = ylen  #    if cylindrical coordinate (axisymmetric), y starts from centre to right
        self.dz_dtheta = dz_dtheta if dz_dtheta is not None else 1 # compartment size in the z (width) direction, either dz (Cartesian, metre) or dtheta (Cylindrical, degree)
  
        self.nx = nx # number of meshes at x and y directions of the compartment
        self.ny = ny #   only support rectangular meshes
        self.dim = self.nx * self.ny
        
        ### partition and diffusion coefficients ###
        self.Kw = None
        self.D = None
        
        ### compartment boundary parameters ###
        
        self.n_meshBdyRight = 0 # number of meshes of the (other) compartment that connects with
        self.n_meshBdyDown = 0  #    the right and down boundaries of 'self'
        
        # allowable options: 'ZeroFlux', 'ZeroConc', 'Periodic', 'FromOther', 'ODE'
        #   order: up, left, right, down
        self.bdyCond = bdy_cond if bdy_cond is not None else ['ZeroFlux']*4
        
        self.meshBdyRight = None # meshes of the 2 boundaries for which OUTGOING flux is calculated
        self.meshBdyDown = None
        
        # To avoid duplicate calculations, only record mass-in from up and left
        #   and mass-out to right and down
        self.massIn_up = None 
        self.massIn_left = None
        self.massOut_right = None
        self.massOut_down = None
        
        self.meshSink = mesh.MeshSink() # sink with zero concentration
        self.meshes = None # The main meshes

        # Ad hoc settings for solids
        self.hasSolid = False
        self.massSolid = 0

    def get_dim(self):
        return self.dim
    def get_x_length(self):
        return self.x_length
    def get_y_length(self):
        return self.y_length
    def get_nx(self):
        return self.nx
    def get_ny(self):
        return self.ny
    def get_bdyCond(self, direction):
        return self.bdyCond[direction]
    def set_Kw(self, Kw):
        self.Kw = Kw
    def set_D(self, D):
        self.D = D
    
    def compDiff_stokes(self, eta, r_s) :
        """ Compute the diffusivity in solution using the Stokes-Einstein equation """
        K = 1.3806488 * 1E-23 # Boltzmann constant, Kg m^2 s^{-2}
        D = K*self.T/6/np.pi/eta/r_s
        return D
        
    ### (START OF) Class methods dealing with boundaries ###
    
    def createBdy(self, nMeshBdyRight, nMeshBdyDown):
        """Create the meshes and mass arrays to hold boundary information
        """
        
        if self.bdyCond[0] == 'FromOther': # up
            self.massIn_up = np.zeros(self.ny)
            
        if self.bdyCond[1] == 'FromOther': # left
            self.massIn_left = np.zeros(self.nx)

        if self.bdyCond[2] == 'FromOther': # right
            self.n_meshBdyRight = nMeshBdyRight
            self.massOut_right = np.zeros(nMeshBdyRight)
            self.meshBdyRight = [mesh.Mesh() for i in range(nMeshBdyRight)]

        if self.bdyCond[3] == 'FromOther': # down
            self.n_meshBdyDown = nMeshBdyDown
            self.massOut_down = np.zeros(nMeshBdyDown)
            self.meshBdyDown =  [mesh.Mesh() for i in range(nMeshBdyDown)]


    def setBdyMesh(self, meshBdyRight, meshBdyDown):
        """Set boundary meshes to the argument values
        meshBdyRight, meshBdydown -- list of instances of class Mesh
        """
        if self.n_meshBdyRight > 0 :
            self.meshBdyRight = meshBdyRight

        if self.n_meshBdyDown > 0:
            self.meshBdyDown = meshBdyDown

            
    def setBdyConc(self, concBdyRight, concBdyDown):
        """Set boundary concentration to the argument values
        concBdyRight, concBdyDown -- numpy array
        """
        for i in range(self.n_meshBdyRight):
            self.meshBdyRight[i].setConc(concBdyRight[i])
        for i in range(self.n_meshBdyDown):
            self.meshBdyDown[i].setConc(concBdyDown[i])
            

    def setBdyMassInOutZero(self):
        if self.massIn_up is not None:
            self.massIn_up.fill(0)
        if self.massIn_left is not None:
            self.massIn_left.fill(0)
        if self.massOut_right is not None:
            self.massOut_right.fill(0)
        if self.massOut_down is not None:
            self.massOut_down.fill(0)
        
    def setMassIn_left(self, massIn_left):
        self.massIn_left = np.copy(massIn_left)
    
    def setMassIn_up(self, massIn_up):
        self.massIn_up = np.copy(massIn_up)
    
    def compFluxBdy(self, direction):
        """Compute flux to boundary
        Args:            
            direction: 0 - up, 1 - left, 2 - right, 3 - down
        Returns:
            flux, area
        """
        assert(direction==3) #only down direction is implemented so far
        
        if self.bdyCond[direction] == 'ZeroFlux':
            mass_tf_rate = 0
        elif self.bdyCond[direction] == 'ZeroConc':
            idx = (self.get_nx()-1) * self.get_ny()
            mass_tf_rate = 0

            for j in range(self.get_ny()) : # y direction left to right
                meshThis = self.meshes[idx+j]
                conc_this = meshThis.getConc()
                f = meshThis.compFlux_diffu(self.meshSink, conc_this, 0, \
                                            meshThis.get_dx()/2, 0)
                area_j = meshThis.compInterArea(direction)
                mass_tf_rate += f*area_j
	
        else:
            raise ValueError('Invalid boundary condition')
            
        area = self.compTotalArea(direction)
        flux = mass_tf_rate / area
        return [flux, area]
        
    ### (END OF) Class methods dealing with boundaries ###
    
    
    ### (START OF) Class methods dealing with geometries ###

    def compTotalArea(self, direction):
        """Compute self's total area to a certain direction
        direction: [0] = up; [1] = left; [2] = right; [3] = down 
        """
        #print(self.coord_sys)
        if self.coord_sys == 'Cartesian' :
            if direction==0 or direction==3 :
                area = self.y_length * self.dz_dtheta
            elif direction==1 or direction==2 :
                area = self.x_length * self.dz_dtheta
            else:
                raise ValueError('direction must be one of: 0, 1, 2, 3')
                
        elif self.coord_sys == 'Cylindrical' :
            pi_alpha_360 = np.pi * self.dz_dtheta / 360
            if direction==0 or direction==3 :
                area = pi_alpha_360 * self.y_length * self.y_length
            elif direction==1 :
                area = 0
            elif direction==2 :
                area = self.x_length * pi_alpha_360 * 2 * self.y_length
            else:
                raise ValueError('direction must be one of: 0, 1, 2, 3')
        else :
            raise ValueError('Coordinate system not implemented')
            
        return area


    def compTotalVolume(self):
        volume = 0
        for i in range(self.nx) :
            for j in range(self.ny) :
                volume += self.meshes[i*self.ny+j].compVolume()

        return volume
        
    ### (END OF) Class methods dealing with geometries ###
    

    ### (START OF) Class methods dealing with meshes ###
    
    def createMeshHomo(self, name, chem, init_conc, coord_x_start, coord_y_start) :
        """ Create mesh for this compartment with homogeneous properties,
            i.e. the partition and diffusion properties are all uniform
        Args:
            name: two-letter string to indicate the name of this compartment
            coord_x_start, coord_y_start: starting coordinates
        """
        dx = self.x_length / self.nx
        dy = self.y_length / self.ny
        
        self.meshes = [mesh.Mesh() for i in range(self.nx*self.ny)] # organised in row dominant
	
        coord_x = coord_x_start
        coord_y = coord_y_start
        current_point = point.Point(coord_x, coord_y, dx, dy, name, name)       
        
        for i in range(self.nx) : # verticle direction up to down
            for j in range(self.ny): # lateral direction left to right	
                idx = i*self.ny + j

                self.meshes[idx].setup(name, chem, init_conc, self.Kw, self.D,
                    current_point.x_coord, current_point.y_coord, current_point.dx, current_point.dy, 
                    self.dz_dtheta)

                if j==self.ny-1 : # last element in the lateral direction, move down
                    coord_x += dx
                    coord_y = coord_y_start
                else : # not the last element in the lateral direction, thus move to the right
                    coord_y += dy
                current_point.setPoint(coord_x, coord_y, dx, dy, name, name)
            # for j
        # for i
        
        
    def compMassIrregMeshRight(self, meshThis, conc_this):
        """Compute mass transfer between meshThis and neighbouring meshes to the right
        whose meshing does not match exactly to meshThis, e.g. 
        meshThis may interface with multiple meshess with self.meshBdyRight
        """
        bDone = False
        massIntoThis = 0
        
        # keep a record that needs to be passed back to meshThis
        meshThis_x_coord = meshThis.get_x_coord()
        meshThis_dx = meshThis.get_dx()
        
        z_length = meshThis.compInterArea(2) / meshThis.get_dx() # interfacial z_length

        for i in range(self.n_meshBdyRight) :
            x1_this = meshThis.get_x_coord()
            x2_this = x1_this + meshThis.get_dx()

            currentX = self.meshBdyRight[i].get_x_coord()  # x coordinates of the neighbouring mesh
            nextX = currentX + self.meshBdyRight[i].get_dx()

            # to compare whether two real numbers are the same
            if meshThis.get_dx()<self.meshBdyRight[i].get_dx():
                thd = meshThis.get_dx()  
            else :
                thd = self.meshBdyRight[i].get_dx()
            thd *= 1e-3

    
            if x1_this > currentX-thd and x1_this < nextX-thd :

                if x2_this < nextX+thd : # meshThis is contained between currentX & nextX
                    x_length = x2_this - x1_this
                    bDone = True
                else : # meshThis extends beyond nextX
                    x_length = nextX - x1_this
                     
                meshOther = self.meshBdyRight[i]
                conc_other = meshOther.getConc()
             
                flux = meshThis.compFlux_diffu(meshOther, conc_this, conc_other, 
                                         meshThis.get_dy()/2, meshOther.get_dy()/2)
            
                meshThis.set_dx(x_length) # this is needed to calculate the correct interfacial area, especially for cylindrical coordinate
                area = meshThis.compInterArea(2) # interfacial area to the right
                mass = area * flux
            
                massIntoThis += mass
                self.massOut_right[i] += -mass; # flux into meshThis is negative flux into the neighbour
            
                if bDone:
                    break
                else:
                    meshThis.set_x_coord(nextX)
                    meshThis.set_dx(x2_this - x1_this - x_length)
        
        # Pass saved record back to meshThis
        meshThis.set_x_coord(meshThis_x_coord)
        meshThis.set_dx(meshThis_dx)
            
        return massIntoThis


    def compMassIrregMeshDown(self, meshThis, conc_this):
        """Compute mass transfer between meshThis and neighbouring meshes downward
        whose meshing does not match exactly to meshThis, e.g. 
        meshThis may interface with multiple meshess with self.meshBdyDown
        """
        bDone = False 
        massIntoThis = 0
        
        # keep a record that needs to be passed back to meshThis
        meshThis_y_coord = meshThis.get_y_coord()
        meshThis_dy = meshThis.get_dy()

        for i in range(self.n_meshBdyDown) :

            y1_this = meshThis.get_y_coord()
            y2_this = y1_this + meshThis.get_dy()

            currentY = self.meshBdyDown[i].get_y_coord() # y coordinates of the neighbouring mesh
            nextY = currentY + self.meshBdyDown[i].get_dy()

            # to compare whether two real numbers are the same
            if meshThis.get_dy()<self.meshBdyDown[i].get_dy() :
                thd = meshThis.get_dy() 
            else :
                thd = self.meshBdyDown[i].get_dy()
            thd *= 1e-3
    
            if y1_this > currentY-thd and y1_this < nextY-thd :

                if  y2_this < nextY+thd : # meshThis is contained between currentY & nextY
                    y_length = y2_this - y1_this
                    bDone = True
                else : # meshThis extends beyond nextY
                    y_length = nextY - y1_this

                meshOther = self.meshBdyDown[i]
                conc_other = meshOther.getConc()
     
                flux = meshThis.compFlux_diffu(meshOther, conc_this, conc_other, 
                                         meshThis.get_dx()/2, meshOther.get_dx()/2)

                meshThis.set_dy(y_length) # this is needed to calculate the correct interfacial area, especially for cylindrical coordinate
                area = meshThis.compInterArea(3) # interfacial area downward
                mass = area * flux

                massIntoThis += mass
                self.massOut_down[i] += -mass # flux into meshThis is negative flux into the neighbour

                if bDone:
                    break
                else:
                    meshThis.set_y_coord(nextY)
                    meshThis.set_dy(y2_this - y1_this - y_length)

        # Pass saved record back to meshThis
        meshThis.set_y_coord(meshThis_y_coord)
        meshThis.set_dy(meshThis_dy)
        
        return massIntoThis;

    ### (END OF) Class methods dealing with meshes ###
    
    ### (START OF) Class methods dealing with ODE computation ###

    def compODEdydt_diffu (self, t, y, args=None):
        """Compute the right-hand side of the ODEs, i.e. dydt, due to diffusion
        """
        
        f = np.zeros(y.shape) # f contains dy/dt to be returned
        
        #print(self.nx)
        #print(self.ny)
        # Calculate mass flux
        for i in range(self.nx) : #  x direction up to down
            for j in range(self.ny) : # y direction left to right
            
                mass_transfer_rate = 0
                idx_this = i*self.ny+j
    			
                meshThis = self.meshes[idx_this]
                conc_this = y[idx_this]
                volume_this = meshThis.compVolume()
    			
                # Setup the neighbouring grids and calculate the mass transfer rate.
                #   in the following order: up, left, (self), right, down.
    			
                # diffusion from up
                if i==0 : # topmost layer, its top is up boundary
                    if self.bdyCond[0] == 'ZeroFlux': 
                        pass
                    elif self.bdyCond[0] == 'FromOther': 
                        mass_transfer_rate += self.massIn_up[j]
                    else :
                        raise ValueError('Invalid boundary condition')
                else : # not topmost layer
                    idx_other = (i-1)*self.ny+j
                    meshUp = self.meshes[idx_other]
                    conc_other = y[idx_other]                
                    flux = meshThis.compFlux_diffu(meshUp, conc_this, conc_other,
                                             meshThis.get_dx()/2, meshUp.get_dx()/2)
                    area = meshThis.compInterArea(0) # interfacial area
                    mass_transfer_rate += area * flux
                    #print(mass_transfer_rate)
      
                # diffusion from left
                area = meshThis.compInterArea(1) # interfacial area
                #print(mass_transfer_rate)

                if j==0 : # leftmost layer, its left is left boundary
                    if self.bdyCond[1] == 'ZeroFlux' :
                        pass
                    elif self.bdyCond[1] == 'ZeroConc' :
                        flux = meshThis.compFlux_diffu(self.meshSink, conc_this, .0, 
                                                 meshThis.get_dy()/2, 0)
                        mass_transfer_rate += area * flux
                    elif self.bdyCond[1] == 'Periodic' :
                        idx_other = (i+1)*self.ny-1
                        meshLeft = self.meshes[idx_other]
                        conc_other = y[idx_other]
                        flux = meshThis.compFlux_diffu(meshLeft, conc_this, conc_other,
                                                 meshThis.get_dy()/2, meshLeft.get_dy()/2)
                        mass_transfer_rate += area * flux
                    elif self.bdyCond[1] == 'FromOther' :
                        mass_transfer_rate += self.massIn_left[i]
                    else :
                        raise ValueError('Invalid boundary condition')
                        
                else : # not leftmost layer
                    idx_other = idx_this-1
                    meshLeft = self.meshes[idx_other]
                    conc_other = y[idx_other]
                    flux = meshThis.compFlux_diffu(meshLeft, conc_this, conc_other,
                                             meshThis.get_dy()/2, meshLeft.get_dy()/2)	
                    mass_transfer_rate += area * flux

                #print(mass_transfer_rate)
                # diffusion from right
                area = meshThis.compInterArea(2) # interfacial area
                
                if j==self.ny-1 : # rightmost layer
                    if self.bdyCond[2] == 'ZeroFlux' : 
                        pass
                    elif self.bdyCond[2] == 'ZeroConc' :
                        flux = meshThis.compFlux_diffu(self.meshSink, conc_this, .0, 
                                                 meshThis.get_dy()/2, 0)
                        mass_transfer_rate += area * flux
                    elif self.bdyCond[2] == 'Periodic' :
                        idx_other = i*self.ny
                        meshRight = self.meshes[idx_other]
                        conc_other = y[idx_other]
                        flux = meshThis.compFlux_diffu(meshRight, conc_this, conc_other, 
                                                 meshThis.get_dy()/2, meshRight.get_dy()/2)
                        mass_transfer_rate += area * flux
                    elif self.bdyCond[2] == 'FromOther' :
                        mass = self.compMassIrregMeshRight(meshThis, conc_this)
                        mass_transfer_rate += mass
                    else :
                        raise ValueError('Invalid boundary condition')
                
                else : # not rightmost layer
                    idx_other = idx_this + 1
                    meshRight = self.meshes[idx_other]
                    conc_other = y[idx_other]
                    flux = meshThis.compFlux_diffu(meshRight, conc_this, conc_other, 
                                             meshThis.get_dy()/2, meshRight.get_dy()/2)
                    mass_transfer_rate += area * flux

                # diffusion from down
                area = meshThis.compInterArea(3) #  interfacial area
                
                if i==self.nx-1 : # // bottom layer
                    if self.bdyCond[3] == 'ZeroFlux' : 
                        pass
                    elif self.bdyCond[3] == 'ZeroConc' :
                        flux = meshThis.compFlux_diffu(self.meshSink, conc_this, .0, 
                                                 meshThis.get_dx()/2, 0)
                        mass_transfer_rate += area * flux
                    elif self.bdyCond[3] == 'FromOther' :
                        mass = self.compMassIrregMeshDown(meshThis, conc_this)
                        mass_transfer_rate += mass
                    else :
                        raise ValueError('Invalid boundary condition')
	 
                else : # not downest layer
                    idx_other = (i+1)*self.ny+j
                    meshDown = self.meshes[idx_other]
                    conc_other = y[idx_other]
                    flux = meshThis.compFlux_diffu(meshDown, conc_this, conc_other, 
                                             meshThis.get_dx()/2, meshDown.get_dx()/2)
                    mass_transfer_rate += area * flux
      
		
                f[idx_this] = mass_transfer_rate / volume_this
                #print(mass_transfer_rate)
            # for j
        # for i
        return f

    ### (END OF) Class methods dealing with ODE computation ###
    
    ### (START OF) Class methods dealing with I/O ###

    def displayMesh(self) :
        """        """
        print('Number of meshes: [x] ', self.nx, ' [y] ', self.ny, '\n')
        for i in range(self.nx) : # verticle direction up to down
            for j in range(self.ny): # lateral direction left to right	
                idx = i*self.ny + j
                if self.meshes[idx].name == 'VH' :
                    print('V ')
                elif self.meshes[idx].name == 'LP' :
                    print('L ')
                elif self.meshes[idx].name == 'CC' :
                    print('C ')
                elif self.meshes[idx].name == 'VE' :
                    print('E ')
                elif self.meshes[idx].name == 'DE' :
                    print('D ')
                elif self.meshes[idx].name == 'HF' :
                    print('H ')
                else :
                    raise ValueError('Invalid mesh name')
            print("\n");


    def compTotalMass(self) :
        """ """
        mass = .0
        for i in range(self.nx) : # verticle direction up to down
            for j in range(self.ny): # lateral direction left to right	
                idx = i*self.ny + j
                volume = self.meshes[idx].compVolume()
                #print('volume = ', volume)
                mass += self.meshes[idx].getConc() * volume
                #print('i=', i, 'j=',j, 'volume=', volume, 'conc=', self.meshes[idx].getConc())
        self.mass = mass
        
        return mass

        
    def getMeshConc(self) :
        """ Return the concentration from all meshes into a single numpy array
        """
        nSpecies = self.meshes[0].get_no_species()
        if nSpecies == 1:
            conc = np.zeros( self.nx*self.ny )
        else :
            conc = np.zeros( (self.nx*self.ny, nSpecies) )
        for i in range(self.nx) : # verticle direction up to down
            for j in range(self.ny): # lateral direction left to right	
                idx = i*self.ny + j
                conc[idx,] = self.meshes[idx].getConc()
        return conc
        
    def setMeshConc_all(self, conc) :
        """ Set the concentration of meshes to conc
        """
        nSpecies = self.meshes[0].get_no_species()        
        for i in range(self.nx) : # verticle direction up to down
            for j in range(self.ny): # lateral direction left to right	
                idx = i*self.ny + j
                self.meshes[idx].setConc(conc[idx,])

    def setMeshConc(self, conc, idx_x, idx_y) :
        """ Set the concentration of ONE mesh, indexed by [idx_x, idx_y], to conc
        """
        idx = idx_x*self.ny + idx_y
        self.meshes[idx].setConc(conc)

    def saveMeshConc(self, b_1st_time, fn) :
        """ Save mesh concentrations to file
        Args: b_1st_time -- if True, write to a new file; otherwise append to the existing file
        """
        if b_1st_time :
            file = open(fn, 'w')
        else :
            file = open(fn, 'a')
            
        for i in range(self.nx) : # verticle direction up to down
            for j in range(self.ny): # lateral direction left to right	
                idx = i*self.ny + j                
                file.write( "{:.6e}".format( np.ndarray.tolist(self.meshes[idx].getConc()) ) )
                file.write('\t')
            file.write('\n')
            
        file.close()

        
    def getXCoord(self) :
        """ Get x-coordinates and return a numpy array
        """
        coord_x = np.zeros((self.nx, self.ny))
        for i in range(self.nx) : # verticle direction up to down
            for j in range(self.ny): # lateral direction left to right	
                idx = i*self.ny + j
                coord_x[i,j] = self.meshes[idx].get_x_coord()
                
        return coord_x

        
    def getYCoord(self) :
        """ Get y-coordinates and return a numpy array
        """
        coord_y = np.zeros((self.nx, self.ny))
        for i in range(self.nx) : # verticle direction up to down
            for j in range(self.ny): # lateral direction left to right	
                idx = i*self.ny + j
                coord_y[i,j] = self.meshes[idx].get_y_coord()
                
        return coord_y

        
    def saveCoord(self, fn_x, fn_y, fn_suffix) :
        """ Save coordinates to two files, fn_x and fn_y followed by fn_suffix
        """
        file_x = open(fn_x+fn_suffix, 'w')
        file_y = open(fn_y+fn_suffix, 'w')
            
        for i in range(self.nx) : # verticle direction up to down
            for j in range(self.ny): # lateral direction left to right	
                idx = i*self.ny + j
                file_x.write( format(self.meshes[idx].get_x_coord(), '.5e\t') )
                file_y.write( format(self.meshes[idx].get_y_coord(), '.5e\t') )
            file_x.write('\n')
            file_y.write('\n')
            
        file_x.close()
        file_y.close()

    def setMeshes_D(self, D):
        """ Set all meshes' diffusivity to D """
        for i in range(self.nx) : # verticle direction up to down
            for j in range(self.ny): # lateral direction left to right	
                idx = i*self.ny + j
                self.meshes[idx].set_D(D)
                
    def setMeshes_Kw(self, Kw):
        """ Set all meshes' partition to Kw """
        for i in range(self.nx) : # verticle direction up to down
            for j in range(self.ny): # lateral direction left to right	
                idx = i*self.ny + j
                self.meshes[idx].set_Kw(Kw)       
                
    def setMeshes_Kw_fromOther(self, compOther, multiplier):
        """ Set meshes' partition coefficient based on the concentrations in the other component
        This is mainly used to model the effect of penetration enhancers (solvents) on partition of the main ingredients
        Kw_new = Kw * ( 1 + compOther.meshes[idx].getConc() * multiplier )
        """
        for i in range(self.nx) : # verticle direction up to down
            for j in range(self.ny): # lateral direction left to right	
                idx = i*self.ny + j
                Kw = self.meshes[idx].get_Kw() * ( 1.0 + compOther.meshes[idx].getConc() * multiplier )
                self.meshes[idx].set_Kw(Kw)
                
                
### (END OF) Class methods dealing with I/O ###