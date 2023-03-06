# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 13:45:49 2017

@author: tc0008
"""


class Point:
    """Class definition for Point
    which is a specific point in space
    """
    
    def __init__(self, x_coord, y_coord, dx, dy, x_type, y_type):
        self.setPoint(x_coord, y_coord, dx, dy, x_type, y_type)
        
    def setPoint(self, x_coord, y_coord, dx, dy, x_type, y_type):
        """
        """
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.dx = dx
        self.dy = dy
        self.x_type = x_type
        self.y_type = y_type

    def cpyPoint(self, source_point):
        """      """
        self.x_coord = source_point.x_coord
        self.y_coord = source_point.y_coord
        self.dx = source_point.dx
        self.dy = source_point.dy
        self.x_type = source_point.x_type
        self.y_type = source_point.y_type
