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


def kmc_evolve(mc_lattice, hop_probability_matrix, startpoint, endpoint, use_initialize_mode):
    """function that updates the hop_probability_matrix after a hop has occured, updating only the effected probabilities, startpoint,endpoint are passed as lists: [i,j,k], if use_initialize_mode is True, it does it for the full lattice"""

    if use_initialize_mode == True:
        hop_probability_matrix = np.tile(
            np.zeros(6), (self.dimensions[0], self.dimensions[1], self.dimensions[2], 1))
        max_i = self.dimensions[0]
        max_j = self.dimensions[1]
        max_k = self.dimensions[2]
        min_i = 0
        min_j = 0
        min_k = 0
    elif use_initialize_mode == False:
        # which atoms are affected by a hop, 2,3's defined by the local lattice size
        max_i = max([startpoint[0], endpoint[0]]) + 2
        max_j = max([startpoint[1], endpoint[1]]) + 2
        min_i = min([startpoint[0], endpoint[0]]) - 2
        min_j = min([startpoint[1], endpoint[1]]) - 2
        max_k = 1
        min_k = 0  # 2d case

    for i, j, k in itertools.product(range(min_i, max_i), range(min_j, max_j), range(min_k, max_k)):

        i = i % self.dimensions[0]  # periodic boundary conditions
        j = j % self.dimensions[1]

        hop_start_point = int(mc_lattice[i][j][k])

        # is there an ion on the site at the start of the cycle?
        if hop_start_point == int(atom_types[1][0]):
            hop_probability_matrix[i][j][k] = get_hex_probability([i, j, k])
        # if there is no atom to move,hop probabilities are zero
        elif hop_start_point == int(atom_types[1][1]):
            hop_probability_matrix[i][j][k] = np.zeros(6)

    return hop_probability_matrix


class mc_lattice:
    "general parameters of mc_lattice"
    atom_types = {'Li': '1', 'Vac': '0'}
    lattice_constant = 2.8334 * 10**(-8)  # in cm
    hex_dict = pickle.load(open("hex_dict.p", "rb"))

    def __init__(self, mc_lattice_size):
        # generates the latice of given dimension, zeros the HH counter
        # atom_types is now a dictionary
        self.dimensions = [mc_lattice_size[0],
                           mc_lattice_size[1], mc_lattice_size[2]]
        self.hop_histogram = np.zeros(6)

    def get_hex_probability(self, lattice_point):
        """
        function that gets hop probabilities for all endpoints, given an atom's location in the mc_lattice
        """

        i, j, k = lattice_point[0], lattice_point[1], lattice_point[2]
        if self.lattice[i][j][k] == self.atom_types['Li']:

            if any([i > (self.dimensions[0] - 3), j > self.dimensions[1] - 3, i < 2, j < 2]):
                hex_dict_key = self.lattice[i][(j + 2) % self.dimensions[1]][k] + self.lattice[i - 1][(j + 2) % self.dimensions[1]][k] + self.lattice[i - 2][(j + 2) % self.dimensions[1]][k] + self.lattice[(i + 1) % self.dimensions[0]][(j + 1) % self.dimensions[1]][k] + self.lattice[i][(j + 1) % self.dimensions[1]][k] + self.lattice[i - 1][(j + 1) % self.dimensions[1]][k] + self.lattice[i - 2][(j + 1) % self.dimensions[1]][k] + self.lattice[(i + 2) %
                                                                                                                                                                                                                                                                                                                                                                                                                                                            self.dimensions[0]][j][k] + self.lattice[(i + 1) % self.dimensions[0]][j][k] + self.lattice[i - 1][j][k] + self.lattice[i - 2][j][k] + self.lattice[(i + 2) % self.dimensions[0]][j - 1][k] + self.lattice[(i + 1) % self.dimensions[0]][j - 1][k] + self.lattice[i][j - 1][k] + self.lattice[i - 1][j - 1][k] + self.lattice[(i + 2) % self.dimensions[0]][j - 2][k] + self.lattice[(i + 1) % self.dimensions[0]][j - 2][k] + self.lattice[i][j - 2][k]

                hex_dict_key = ''.join(str(hex_dict_key))

                # box_slice = np.roll(
                #    np.roll(self.lattice, shift=-i + 2, axis=0), shift=-j + 2, axis=1)[:5, :5, 0]
            else:
                """
                    box_slice = self.lattice[i - 2:i + 3, j - 2:j + 3, :][:, :, 0]
                hex_dict_key = ''.join(np.concatenate(
                    (box_slice[2:, 0], box_slice[1:, 1], box_slice[:2, 2], box_slice[3:, 2], box_slice[:-1, 3], box_slice[:-2, 4])))[::-1]
                """
                self.working_lattice = self.lattice.reshape(
                    1, self.dimensions[0] * self.dimensions[1])[0]
                # first row: self.

                hex_dict_key = ''.join(np.concatenate((self.working_lattice[(i) + self.dimensions[1] * (j + 2):(i - 3) + self.dimensions[1] * (j + 2):-1], self.working_lattice[(i + 1) + self.dimensions[1] * (j + 1):(i - 3) + self.dimensions[1] * (j + 1):-1], self.working_lattice[(i + 2) + self.dimensions[1] * (j):(i) + self.dimensions[1] * (
                    j):-1], self.working_lattice[(i - 1) + self.dimensions[1] * (j):(i - 3) + self.dimensions[1] * (j):-1], self.working_lattice[(i + 2) + self.dimensions[1] * (j - 1):(i - 2) + self.dimensions[1] * (j - 1):-1], self.working_lattice[(i + 2) + self.dimensions[1] * (j - 2):(i - 1) + self.dimensions[1] * (j - 2):-1])))
                """

            hex_dict_key = self.lattice[i][(j + 2) % self.dimensions[1]][k] + self.lattice[i - 1][(j + 2) % self.dimensions[1]][k] + self.lattice[i - 2][(j + 2) % self.dimensions[1]][k] + self.lattice[(i + 1) % self.dimensions[0]][(j + 1) % self.dimensions[1]][k] + self.lattice[i][(j + 1) % self.dimensions[1]][k] + self.lattice[i - 1][(j + 1) % self.dimensions[1]][k] + self.lattice[i - 2][(j + 1) % self.dimensions[1]][k] + self.lattice[(i + 2) %
                                                                                                                                                                                                                                                                                                                                                                                                                                                        self.dimensions[0]][j][k] + self.lattice[(i + 1) % self.dimensions[0]][j][k] + self.lattice[i - 1][j][k] + self.lattice[i - 2][j][k] + self.lattice[(i + 2) % self.dimensions[0]][j - 1][k] + self.lattice[(i + 1) % self.dimensions[0]][j - 1][k] + self.lattice[i][j - 1][k] + self.lattice[i - 1][j - 1][k] + self.lattice[(i + 2) % self.dimensions[0]][j - 2][k] + self.lattice[(i + 1) % self.dimensions[0]][j - 2][k] + self.lattice[i][j - 2][k]

            hex_dict_key = ''.join(str(hex_dict_key))
            """
            return self.hex_dict[hex_dict_key]
        else:
            return np.zeros(6)

    def fill_lattice(self, concentration):

        # given a concentration, this guy initializes the mc_lattice and probability matrix
        # sets use_presimulation_mode = True
        self.lattice = np.tile(self.atom_types['Vac'], self.dimensions)

        self.number_of_li = int(concentration * np.product(self.dimensions))
        Li_atoms = 0
        self.probability_matrix = np.tile(
            np.zeros(6), (self.dimensions[0], self.dimensions[1], self.dimensions[2], 1))
        while Li_atoms < self.number_of_li:
            i = np.random.randint(0, self.dimensions[0])
            j = np.random.randint(0, self.dimensions[1])
            k = np.random.randint(0, self.dimensions[2])
            if self.lattice[i][j][k] == self.atom_types['Vac']:
                self.lattice[i][j][k] = (self.atom_types['Li'])
                Li_atoms += 1

        for i, j, k in itertools.product(range(0, self.dimensions[0]), range(0, self.dimensions[1]), range(0, self.dimensions[2])):
            # is there an ion on the site at the start of the cycle?
            if self.lattice[i][j][k] == (self.atom_types['Li']):
                self.probability_matrix[i][j][k] = self.get_hex_probability([
                    i, j, k])
            # if there is no atom to move,hop probabilities are zero
            elif self.lattice[i][j][k] == (self.atom_types['Vac']):
                self.probability_matrix[i][j][k] = np.zeros(6)
        self.use_presimulation_mode = True

    def hop(self):
        # the hopping function, moves a single atom, and reocords the hop in the HH, and (if presim mode ==false) the distance lattice
        k = 0
        gamma = np.sum(self.probability_matrix)
        normalized_hop_probabilities = np.cumsum(
            self.probability_matrix) / gamma  # normalizing the probabilities
        rho = random.random()  # the random number
        hop_master_index = np.searchsorted(
            normalized_hop_probabilities, rho)  # get master index of hop
        # get which endpoint the ion hopped to (number between 0-5)
        endpoint_index = hop_master_index % 6
        hopping_ion_j = (hop_master_index // 6) % self.dimensions[1]
        hopping_ion_i = (hop_master_index // 6) // self.dimensions[1]
        hop_endpoints = np.array([[hopping_ion_i + 1, hopping_ion_j], [hopping_ion_i, hopping_ion_j + 1], [hopping_ion_i - 1, hopping_ion_j + 1], [hopping_ion_i - 1,
                                                                                                                                                   hopping_ion_j], [hopping_ion_i, hopping_ion_j - 1], [hopping_ion_i + 1, hopping_ion_j - 1]])  # all of the possible endpoint indices for the hop
        hop_endpoint_i = hop_endpoints[endpoint_index][0]
        hop_endpoint_j = hop_endpoints[endpoint_index][1]
        # move vacancy in
        # print [hop_master_index, endpoint_index, hopping_ion_i, hopping_ion_j, hop_endpoint_i,
        #       hop_endpoint_j, self.probability_matrix[hopping_ion_i][hopping_ion_j][0]]
        self.lattice[hopping_ion_i][hopping_ion_j][k] = self.atom_types['Vac']
        self.lattice[(hop_endpoint_i % self.dimensions[0])][hop_endpoint_j % self.dimensions[1]
                                                            ][0] = self.atom_types['Li']  # moving ion to appropriate endpoint mc_lattice

        if self.use_presimulation_mode == False:
            # moving the ion's coordinates (eg [2,3,0]) to the appropriate site,
            self.distance_lattice[hop_endpoint_i % self.dimensions[0]][hop_endpoint_j %
                                                                       self.dimensions[1]][k] = self.distance_lattice[hopping_ion_i][hopping_ion_j][k]
            self.distance_lattice[hop_endpoint_i % self.dimensions[0]][hop_endpoint_j % self.dimensions[1]][k][2] = self.distance_lattice[hop_endpoint_i % self.dimensions[0]][hop_endpoint_j % self.dimensions[1]][k][2] + 100 * (
                hop_endpoint_i // self.dimensions[0]) + hop_endpoint_j // self.dimensions[1]  # adds index to distance lattice for periodic boundary condition purposes, 3rd value is 100 times a loop in i, +single times loops in j
            # clears the coordiates from the old site ->sticks in a vaccancy
            self.distance_lattice[hopping_ion_i][hopping_ion_j][k] = [-1, 0, 0]
            # time of the hop, as defined in VdV paper
            self.time_step_per_hop = (-1. / gamma) * \
                np.log(random.random())
            self.time_step_per_kmcstep = self.time_step_per_kmcstep + \
                self.time_step_per_hop  # total time of the step
        time1 = time.time()
        for i, j, k in itertools.product(range(hopping_ion_i - 2, hopping_ion_i + 3), range(hopping_ion_j - 2, hopping_ion_j + 4), range(k, k + 1)):
            self.probability_matrix[i % self.dimensions[0]][j % self.dimensions[1]][k % self.dimensions[2]] = self.get_hex_probability(
                [i % self.dimensions[0], j % self.dimensions[1], k % self.dimensions[2]])  # update the probability lattice after the hop
        time2 = time.time()
        # print time2 - time1
        self.hop_histogram[endpoint_index] += 1

    def initialize_KMC(self):
        # function that creates the distance lattice
        # sets use_presimulation_mode to False
        # needs to be run after presim
        self.distance_lattice = np.tile([-1, 0, 0], (self.dimensions + [1]))
        for i, j, k in itertools.product(range(0, self.dimensions[0]), range(0, self.dimensions[1]), range(0, self.dimensions[2])):

            if self.lattice[i][j][k] == self.atom_types['Li']:
                self.distance_lattice[i][j][k] = [i, j, k]
        self.use_presimulation_mode = False
        self.time_step_per_kmcstep = 0
        self.total_time = 0

    def KMC_step(self):
        # runs the correct number of hops
        for number_of_hop in range(self.number_of_li):
            self.hop()
            if self.use_presimulation_mode == False:
                self.total_time = self.total_time + self.time_step_per_kmcstep

    def calculate_diffusion_coefficients(self):
        # if initialize KMC was run before KMC step, this guy calculates the diffusion coeficcients
        # this is a list which contains (delta i, delta j) for each atom:
        self.distances_travelled_list = np.array([[[0, 0]]])
        for i, j, k in itertools.product(range(0, self.dimensions[0]), range(0, self.dimensions[1]), range(0, self.dimensions[2])):
            # vacancies are [-1,0,0], converting distance lattice values into a list of deltax/y for each atoms
            if self.distance_lattice[i][j][k][0] >= 0:
                i_initial = self.distance_lattice[i][j][k][0]
                j_innitial = self.distance_lattice[i][j][k][1]
                i_final = i + self.dimensions[0] * \
                    self.distance_lattice[i][j][k][2] // 100
                j_final = j + self.dimensions[1] * (self.distance_lattice[i][j][k][2] - round(
                    self.distance_lattice[i][j][k][2] / 100) * 100)
                self.distances_travelled_list = np.append(self.distances_travelled_list, [[[
                    (i_final - i_initial), (j_final - j_innitial)]]], axis=1)
        self.distances_travelled_list = self.distances_travelled_list * self.lattice_constant
        # d_j is the vector sum of all the hex distances traveled for each atom.

        d_j_sum = np.sum(self.distances_travelled_list, axis=1)[0]
        d_j_convert_and_square = d_j_sum[0]**2 + \
            d_j_sum[1]**2 + d_j_sum[0] * d_j_sum[1]
        d_star_convert_and_sqaure = np.transpose(self.distances_travelled_list)[0]**2 + np.transpose(self.distances_travelled_list)[
            1]**2 + np.transpose(self.distances_travelled_list)[0] * np.transpose(self.distances_travelled_list)[1]
        d_star_sum = np.sum(d_star_convert_and_sqaure)
        self.d_j = d_j_convert_and_square / \
            (2 * 2 * self.total_time * self.number_of_li)
        self.d_star = d_star_sum / \
            (2 * 2 * self.total_time * self.number_of_li)

    def colorsquare(self):  # to plot the lattices as you go, and save as png's
        """Figure animation stuff"""
        plt.ion()
        plt.imshow(np.transpose(self.lattice)[
                   0].astype(float), cmap='gray_r', interpolation='none')
        plt.show()
        plt.pause(1)
        return

    def dimension_check(self):
        # checks the dimension dependance of the KMC loops
        for dimension in range(10, 60, 10):
            print dimension
            for averaging_iter in range(10):
                x = mc_lattice([dimension, dimension, 1])
                x.fill_lattice(0.2)
                for z in range(200):
                    x.KMC_step()

                x.initialize_KMC()
                for z in range(500):
                    x.KMC_step()
                x.calculate_diffusion_coefficients()
                print x.d_j
            x.hop_histogram = np.zeros(6)

    def HH_check(self):
        # checks whether each hop direction is equally likely, confirms that dictionary construction is valid.
        for dimension in range(10, 50, 10):
            print dimension
            for averaging_iter in range(10):
                x = mc_lattice([dimension, dimension, 1])
                x.fill_lattice(0.2)
                for z in range(500):
                    x.KMC_step()
                print(x.hop_histogram[1] / x.hop_histogram[0])
            print "averaged HH: " + str(x.hop_histogram[1] / x.hop_histogram[0])
            x.hop_histogram = np.zeros(6)

    def counts_check(self):
        x = mc_lattice([50, 50, 1])
        x.fill_lattice(0.2)
        self.diffusion_list = []
        x.initialize_KMC()
        for z in range(100000):
            x.KMC_step()
            x.calculate_diffusion_coefficients()
            self.diffusion_list = np.append(self.diffusion_list, x.d_j)
        plt.scatter(range(len(self.diffusion_list)), self.diffusion_list)

    def concentration_run(self):
        for concentration in np.array(range(90, 100, 2)) / 100.:
            self.fill_lattice(concentration)
            self.averaging_d_j = []
            self.averaging_d_star = []

            for averaging_iter in range(10):
                for z in range(500):
                    self.KMC_step()
                self.initialize_KMC()
                for z in range(750):
                    self.KMC_step()

                self.calculate_diffusion_coefficients()
                self.averaging_d_j = np.append(self.averaging_d_j, self.d_j)
                self.averaging_d_star = np.append(
                    self.averaging_d_star, self.d_star)
            print str(concentration) + " " + str(np.mean(self.averaging_d_j)) + " " + str(np.std(self.averaging_d_j))
            print str(concentration) + " " + str(np.mean(self.averaging_d_star)) + " " + str(np.std(self.averaging_d_star))
