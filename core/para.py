# -*- coding: utf-8 -*-
"""
Class definition for parameters that can be
    passed to compartment classes for calculations of
    e.g. partitoin and diffusion coefficients
"""

class optval():
    """ Class definition for option-value"""
    def __init__(self):
        self.option = None
        self.value = None
        