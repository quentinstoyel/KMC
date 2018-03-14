# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:36:34 2017

@author: qstoyel
Object oriented KMC code:
"""


import numpy as np
import matplotlib.pyplot as plt
import random
import time
import sys
import math
import re
from itertools import combinations
import pickle
import itertools  # to hide all my nested for loop shenannigans inside of functions

atom_types = {'Li': -1, 'Vac': 1}


class mc_lattice(object):

    def _init_(self, mc_lattice_size, atom_types):
        # atom_types is now a dictionary
        self.dimensions = [mc_lattice_size[0],
                           mc_lattice_size[1], mc_lattice_size[2]]
        self.lattice = np.zeros(self.dimensions) - atom_types{'Vac'}

    def fill_lattice(self, concentration):
        number_of_li = concentration * np.product(self.dimensions)
        print number_of_li
