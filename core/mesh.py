# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 15:00:12 2017

@author: tc0008
"""

#import importlib
import numpy as np

#import chemical
#chemical = importlib.import_module('chemical')

class Mesh:
    """Class definition for Mesh.
    """

    # define member variables and assign default values
    #
    def __init__(self, coord_sys='Cartesian'):
        """ A generic method to create the instance """
        self.name = None        
        self.chem = None
        self.nSpecies = None
        self.conc = np.array([0])
        self.Kw = None
        self.D = None        
        self.x_coord = None
        self.y_coord = None
        self.dx = None
        self.dy = None
        self.dz = None
        self.coord_sys = coord_sys
        
    def setup(self, name, chem, conc, Kw, D, x_coord, y_coord, dx, dy, dz):       
        """ A separate setup method to modify instance's attributes
        Args:
            Kw, D - partition and diffusion coefficients in this mesh
        """        
        self.name = name
        
        self.chem = chem # a list of objectives of the class Chemical
        if chem is None:
            self.nSpecies = 0
        elif type(chem) is not list :
            self.nSpecies = 1
        else :
            self.nSpecies = len(chem) # Number of chemcial species
        self.conc = conc # The dependent variable (named 'conc') of interest
        self.Kw = Kw
        self.D = D
        
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.dx = dx
        self.dy = dy
        self.dz = dz
                
    def get_dx(self):
        return self.dx
    def set_dx(self, dx):
        self.dx = dx
    def get_dy(self):
        return self.dy
    def set_dy(self, dy):
        self.dy = dy        
    def get_x_coord(self):
        return self.x_coord
    def set_x_coord(self, x_coord):
        self.x_coord = x_coord
    def get_y_coord(self):
        return self.y_coord
    def set_y_coord(self, y_coord):
        self.y_coord = y_coord
    def getConc(self):
        return np.array(self.conc)
    def setConc(self, conc):
        self.conc = np.copy(conc)
    def get_no_species(self):
        return self.nSpecies
    def set_no_species(self, n):
        self.nSpecies = n
    def get_Kw(self):
        return self.Kw
    def set_Kw(self, Kw):
        self.Kw = Kw
    def get_D(self):
        return self.D
    def set_D(self, D):
        self.D = D
        
  
    ### (START OF) Class methods dealing with geometries ###
        
    def compInterArea(self, direction):
        """Calculate the interfacial area between self and a neighbouring mesh
        direction: [0] = up; [1] = left; [2] = right; [3] = down
        """  
        if self.coord_sys == 'Cartesian' :
            if direction==0 or direction==3 :
                area = self.dy * self.dz
            elif direction==1 or direction==2 :
                area = self.dx * self.dz
            else :
                raise ValueError('direction must be one of: 0, 1, 2, 3')
                
        elif self.coord_sys == 'Cylindrical' :
            r1 = self.y_coord
            r2 = self.y_coord + self.dy;
            pi_alpha_360 = np.pi * self.m_dz / 360
            if direction==0 or direction==3 :
                area = pi_alpha_360 * (r2*r2 - r1*r1)
            elif direction==1 :
                area = self.dx * pi_alpha_360 * 2 * r1
            elif direction==2 :
                area = self.dx * pi_alpha_360 * 2 * r2
            else :
                raise ValueError('direction must be one of: 0, 1, 2, 3')
                
        else :
            raise ValueError('Coordinate system not implemented')
        
        return area

    
    def compVolume(self) :  
        if self.coord_sys == 'Cartesian' :
            volume = self.dx * self.dy * self.dz
        elif self.coord_sys == 'Cylindrical' :
            r1 = self.y_coord
            r2 = self.y_coord + self.dy
            volume = self.dx * (np.pi*self.dz/360) * (r2*r2 - r1*r1)
        else :
            raise ValueError('Coordinate system not implemented')
        return volume

    ### (END OF) Class methods dealing with geometries ###  
    

    def compFlux_diffu(self, meshOther, conc_this, conc_other, half_dis_this, half_dis_other):
        """ Compute the flux, due to diffusion, of solute from meshOther to <self> mesh
        However, do not use concentration values in the mesh objects,
        instead use <conc_this> and <conc_other>
        """
        K_other2this = meshOther.get_Kw() / self.get_Kw()
        flux = conc_other - K_other2this*conc_this
        #print(flux)
        tmp1 = half_dis_other/meshOther.get_D() + K_other2this*half_dis_this/self.get_D()
        #print(tmp1)
        flux /= tmp1
        #print(flux)
        #raise ValueError('stop')
        return flux


class MeshSink(Mesh):
    """ Derive a Sink class from Mesh
    """
    def __init__(self):
        Mesh.__init__(self)
        conc = 0
        Kw = 1 # arbitrary value, doesn't matter in flux calculation
        D = 1  # very big value, essentially means anything in the sink can be quickly removed
        x_coord = 0
        y_coord = 0
        dx = 0
        dy = 0
        dz = 0
        Mesh.setup(self, 'SK', None, conc, Kw, D, x_coord, y_coord, dx, dy, dz)