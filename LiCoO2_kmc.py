# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:36:34 2017

@author: qstoyel
KMC code:
"""


import numpy as np
import matplotlib.pyplot as plt
import random
import time
import sys
import math
from itertools import combinations
import pickle
import itertools  # to hide all my for loop shenannigans inside of functions

"""Start Timer"""
start_timer = time.time()

"""Initiallizing useful variables"""
atom_types = ([["Li", "Vac"], [-1, 1]]
              )  # what values are assigned to which atoms in the matrix
kb_t = 0.025  # kb=~8.6*10^-5
prefactor = 10**(13)  # entropic factor from VderV paper
# in ev, constant, hop energy when the two sites on each side are occupied
dumbell_hop_energy = 1.48
# the hop probability associated with that energy
dumbell_hop_probability = prefactor * math.exp(-(dumbell_hop_energy) / kb_t)
lattice_constant = 2.8334 * 10**(-8)  # in cm
hop_histogram = np.zeros(6)

"""Building the Dictionary"""

"""Building the Dictionary"""

def get_excited_energy(lattice):
    """
    lattice values MUST BE FLOATS!
    lattice must be a 4x4x1,rotated so the hop is going from 1,1 to 2,1 endpoint, with the 2,2 nearset neighbour endpoint empty
    uses the local cluster expansion, returns the excited state energy
    """

    # -25.5985 #magic number, yay!, from difference between 10 and 16 site cluster expansions, in ev
    Li_offsite_unit_energy = -25.832
    # magic number, contribution of each of the "on site" Lithiums, in ev
    Li_unit_energy = -25.05

    if lattice[2][1][0] != atom_types[1][1]:
        sys.exit("Error: endpoint not empty!")
    if lattice[1][1][0] != atom_types[1][0]:
        sys.exit("Error: no ion to move!")
    if lattice[2][2][0] != atom_types[1][1]:
        sys.exit("Error: nearest neighbour not empty!")
    eci = [-89292.2076392, 12.0246248871, 0.5321464472, 0.6013549835, 0.6062609396, 0.5901284127, 0.5864268761,
           0.1392609219, 0.0912246106, 0.0125126163, 0.040332075]  # from LCE, energy contribution of each cluster
    # hard coded, depends on which clusters are chosen in the LCE,
    multiplicities = np.array([1, 10, 2, 2, 2, 2, 1, 1, 2, 2, 1])
    sites = np.array([lattice[2][2][0], lattice[0][0][0], lattice[1][0][0], lattice[2][0][0], lattice[0][1][0], lattice[1][2][0],
                      lattice[2][3][0], lattice[3][1][0], lattice[3][2][0], lattice[3][3][0]])  # defines the sites 0-9, as indicated in notebook
    number_of_sites = len(sites)
    number_of_clusters = len(eci)
    # the product bit of the cluster expansion, eci*expted occupancy
    product = np.zeros(number_of_clusters)
    # Li sites, not included in the cluster expansion
    li_sites_not_in_lce = list(
        [lattice[3][0][0], lattice[1][3][0], lattice[0][3][0], lattice[0][2][0]])
    # corrects for the 4 Li sites that are always full in the LCE, in ev.
    correction_for_sites_not_in_lce = (li_sites_not_in_lce.count(
        atom_types[1][1])) * (-Li_offsite_unit_energy)
    # clusters:
    # energy contribution from clusters, dot product with eci is done on the way:
    product[0] = eci[0]
    product[1] = eci[1] * sum(sites)
    product[2] = eci[2] * (sites[1] + sites[3])
    product[3] = eci[3] * (sites[4] + sites[7])
    product[4] = eci[4] * (sites[5] + sites[8])
    product[5] = eci[5] * (sites[6] + sites[9])
    product[6] = eci[6] * (sites[2])
    product[7] = eci[7] * (sites[6] * sites[9])
    product[8] = eci[8] * (sites[4] * sites[1] + sites[3] * sites[7])
    product[9] = eci[9] * (sites[4] * sites[5] *
                           sites[6] + sites[7] * sites[8] * sites[9])
    product[10] = eci[10] * (sites[1] * sites[2] * sites[3])
    excited_energy = sum(product) + correction_for_sites_not_in_lce
    return excited_energy  # returns total excited state energy, on order 90keV


def get_config_energy(input_lattice):
    """
    input_lattice values MUST BE FLOATS!
    input_lattice must be a 4x4x1, function takes input_lattice, uses the global cluster expansion to return the configurational energy of the state
    """
    lattice = np.array(
        input_lattice)  # because reinitializing, otherwise it still points to the big lattice
    # it had better be, otherwise the CE won't work,, hard coded from the cluster expansion.
    CE_lattice_size = (4, 4, 1)
    eci = [-2.333258, 0.001058, 0.045567, 0.002186, 0.000803, 0.000000, -0.000329, -0.000050, -0.000001, 0.000009, -0.000008, 0.000001, -0.000003,
           0.000000, 0.000000, -0.000000, 0.000000, 0.000000, 0.000027, 0.000001, 0.000004, 0.000001, 0.000002, 0.000003, 0.000001]  # eci, from maps
    multiplicities = np.array([1, 16, 48, 48, 48, 96, 32, 96, 32, 48, 192, 32, 192,
                               192, 96, 192, 96, 64, 48, 32, 192, 96, 48, 96, 96])  # multiplicities from maps
    # reference_energy=[-5590.22,-5564.88] #also from maps, ref.out
    # energy contribution from every Li atom in the lattice
    Li_unit_energy = -25.337714498
    # product is the the expectation values of the product of every site in a cluster
    product = np.zeros(len(eci))
    product[0] = 1.0  # by definition
    product[1] = np.average(lattice)  # also by definition
    k = 0
    # defines the product with all of the expectation values of the occupancies
    for i, j in itertools.product(range(CE_lattice_size[0]), range(CE_lattice_size[1])):
        # for k in range(CE_lattice_size[2]):#always zero, no need to loop
        # for speedup, by minimizing memory reads into lattice
        cluster_location = lattice[i][j][k]
        nearest_neighbour1 = lattice[(i + 1) % CE_lattice_size[0]][j][k]
        nearest_neighbour2 = lattice[i][(j + 1) % CE_lattice_size[1]][k]
        nearest_neighbour3 = lattice[(
            i + 1) % CE_lattice_size[0]][(j + 1) % CE_lattice_size[1]][k]
        neighbour1 = lattice[(i + 2) % CE_lattice_size[0]
                             ][(j + 1) % CE_lattice_size[1]][k]
        neighbour2 = lattice[(i + 1) % CE_lattice_size[0]
                             ][(j + 2) % CE_lattice_size[1]][k]
        # pairs
        product[2] = product[2] + (cluster_location * nearest_neighbour1 + cluster_location *
                                   nearest_neighbour2 + cluster_location * nearest_neighbour3) / multiplicities[2]
        product[3] = product[3] + (cluster_location * neighbour1 + cluster_location * neighbour2 + cluster_location *
                                   lattice[(i + 1) % CE_lattice_size[0]][(j + 3) % CE_lattice_size[1]][k]) / multiplicities[3]
        product[4] = product[4] + (cluster_location * lattice[(i + 2) % CE_lattice_size[0]][j][k] + cluster_location * lattice[i][(j + 2) %
                                                                                                                                  CE_lattice_size[1]][k] + cluster_location * lattice[(i + 2) % CE_lattice_size[0]][(j + 2) % CE_lattice_size[1]][k]) / multiplicities[4]
        product[5] = 0  # eci=0, atoms are too far away
        # triplets:
        product[6] = product[6] + (cluster_location * nearest_neighbour1 * nearest_neighbour3 +
                                   cluster_location * nearest_neighbour2 * nearest_neighbour3) / multiplicities[6]
        product[7] = product[7] + (cluster_location * nearest_neighbour1 * neighbour1 + cluster_location * nearest_neighbour3 * neighbour1 + cluster_location * nearest_neighbour3 * neighbour2 + cluster_location * nearest_neighbour2 * neighbour2 + cluster_location *
                                   nearest_neighbour2 * lattice[(i + 3) % CE_lattice_size[0]][(j + 1) % CE_lattice_size[1]][k] + cluster_location * lattice[(i + 3) % CE_lattice_size[0]][j][k] * lattice[(i + 3) % CE_lattice_size[0]][(j + 1) % CE_lattice_size[1]][k]) / multiplicities[7]
        # product[8]+(cluster_location*neighbour1*neighbour2+cluster_location*lattice[(i+3)%CE_lattice_size[0]][(j+1)%CE_lattice_size[1]][k]*neighbour2)/multiplicities[8]
        product[8] = 0
        product[9] = product[9] + (cluster_location * nearest_neighbour1 * lattice[(i + 2) % CE_lattice_size[0]][j][k] + cluster_location * nearest_neighbour2 * lattice[i][(
            j + 2) % CE_lattice_size[1]][k] + cluster_location * nearest_neighbour3 * lattice[(i + 2) % CE_lattice_size[0]][(j + 2) % CE_lattice_size[1]][k]) / multiplicities[9]
        # quartics
        product[18] = product[18] + (cluster_location * nearest_neighbour1 * nearest_neighbour3 * neighbour1 + cluster_location * nearest_neighbour3 * nearest_neighbour2 * neighbour2 +
                                     cluster_location * nearest_neighbour2 * lattice[(i + 3) % CE_lattice_size[0]][j][k] * lattice[(i + 3) % CE_lattice_size[0]][(j + 1) % CE_lattice_size[1]][k]) / multiplicities[18]
    energy_no_li = -89038.093349139890911  # the groundstate energy, in ev
    # contribution from having n li, ev
    energy_from_li = np.count_nonzero(
        lattice == atom_types[1][0]) * Li_unit_energy
    # tiny, energy per cite (order meVs), contribution from configuration, in ev
    energy_from_CE = np.dot(multiplicities * product, eci)
    # print "CE energy: " +str(energy_from_CE)
    lattice_energy = energy_no_li + energy_from_li + energy_from_CE
    return lattice_energy  # total energy of 4x4 lattice (order keV) ~=90kev


def lattice_convert(lattice):
    """function that takes a lattice and converts it into a 16 digit int """

    lattice = np.reshape(lattice, 16).astype(int).astype(str)
    s = "".join(lattice)
    return int(s)


def dictionary_builder():  # creates a dictionary that contains the hop probailities for all of the lattice combinations, must be rerun for each temperature
    master_dictionary = {}
    dict_lattice = np.array([[1.],  [1.],  [1.],  [1.], [
                            1.],  [-1.],  [1.],  [1.], [1.],  [1.],  [1.],  [1.], [1.],  [1.],  [1.],  [1.]]) + 2
    dict_values = [1, 3]
    for a0 in dict_values:  # ignore some sites (start (5), end(9) and NN(10))
        dict_lattice[0] = a0
        for a1 in dict_values:
            dict_lattice[1] = a1
            for a2 in dict_values:
                dict_lattice[2] = a2
                for a3 in dict_values:
                    dict_lattice[3] = a3
                    for a4 in dict_values:
                        dict_lattice[4] = a4
                        for a7 in dict_values:
                            dict_lattice[7] = a7
                            for a8 in dict_values:
                                dict_lattice[8] = a8
                                for a6 in dict_values:  # there is a rogue transpose in here somewhere, which screwed up the numbering
                                    dict_lattice[6] = a6
                                    for a11 in dict_values:
                                        dict_lattice[11] = a11
                                        for a12 in dict_values:
                                            dict_lattice[12] = a12
                                            for a13 in dict_values:
                                                dict_lattice[13] = a13
                                                for a14 in dict_values:
                                                    dict_lattice[14] = a14
                                                    for a15 in dict_values:
                                                        dict_lattice[15] = a15
                                                        dictionary_key = lattice_convert(
                                                            dict_lattice)  # the tag for each structure
                                                        working_lattice = (np.reshape(
                                                            dict_lattice, (4, 4, 1))) - 2
                                                        master_dictionary[dictionary_key] = get_hop_probability(
                                                            working_lattice, prefactor, kb_t)  # write the probailities to the dicitonary
    pickle.dump(master_dictionary, open("master_dictionary.p", "wb+"),
                pickle.HIGHEST_PROTOCOL)  # saves dictionary to pickle


def get_hop_probability(local_lattice, prefactor, kb_t):
    """
    given an input lattice with a hoping ion on from the 1,1 site, to the 1,2 site,
    this function returns the probability of the hop
        """
    use_dictionary_mode = True  # toggles dicitonary option
    if use_dictionary_mode == False:
        drift_potential = 0  # in V, to apply an external field
        # dot products the potential
        drift_directions = np.array(
            [1, 0.5, -0.5, -1, -0.5, 0.5]) * drift_potential
        # calculates excited and groundstate energies to get barrier
        hop_energy = get_excited_energy(
            local_lattice) - get_config_energy(local_lattice)
        probability = prefactor * \
            math.exp(-(hop_energy) / kb_t)  # the stat mech bit
        # dictionary_builder() #build the master_dictionary and save to file
    else:
        # deals with negatives/0 in lattice, make it a list of 1 and 3's instead of 1's and -1's
        local_lattice = local_lattice + 2
        # collapse the lattice into the dictionary key
        lattice_key = lattice_convert(local_lattice)
        # get the probability from the dictionary
        probability = master_dictionary[lattice_key]
    return probability


"""To make a KMC step"""


def kmc_evolve(mc_lattice, hop_probability_matrix, startpoint, endpoint, use_initialize_mode):
    """function that updates the hop_probability_matrix after a hop has occured, updating only the effected probabilities, startpoint,endpoint are passed as lists: [i,j,k], if use_initialize_mode is True, it does it for the full lattice"""

    if use_initialize_mode == True:
        hop_probability_matrix = np.tile(
            np.zeros(6), (mc_lattice_size[0], mc_lattice_size[1], mc_lattice_size[2], 1))
        max_i = mc_lattice_size[0]
        max_j = mc_lattice_size[1]
        max_k = mc_lattice_size[2]
        min_i = 0
        min_j = 0
        min_k = 0
    elif use_initialize_mode == False:
        # which atoms are affected by a hop, 2,3's defined by the local lattice size
        max_i = max([startpoint[0], endpoint[0]]) + 3
        max_j = max([startpoint[1], endpoint[1]]) + 3
        min_i = min([startpoint[0], endpoint[0]]) - 2
        min_j = min([startpoint[1], endpoint[1]]) - 2
        max_k = 1
        min_k = 0  # 2d case

    for i, j, k in itertools.product(range(min_i, max_i), range(min_j, max_j), range(min_k, max_k)):

        i = i % mc_lattice_size[0]  # periodic boundary conditions
        j = j % mc_lattice_size[1]

        hop_start_point = float(mc_lattice[i][j][k])
        # list of probabilities for atom to go to each endpoint, reset for each atom
        hop_probability_by_site = np.zeros(6)

        # is there an ion on the site at the start of the cycle?
        if int(hop_start_point) == int(atom_types[1][0]):
            endpoint_mc_lattice_indices = np.array([[i + 1, j], [i, j + 1], [i - 1, j + 1], [i - 1, j], [
                                                   i, j - 1], [i + 1, j - 1]])  # indices in the mc_lattice where this atom can hop
            endpoint_occupancy = np.array([mc_lattice[endpoint_mc_lattice_indices[0][0] % mc_lattice_size[0]][endpoint_mc_lattice_indices[0][1] % mc_lattice_size[1]][0], mc_lattice[endpoint_mc_lattice_indices[1][0] % mc_lattice_size[0]][endpoint_mc_lattice_indices[1][1] % mc_lattice_size[1]][0],
                                           mc_lattice[endpoint_mc_lattice_indices[2][0] % mc_lattice_size[0]][endpoint_mc_lattice_indices[2][1] % mc_lattice_size[1]][0], mc_lattice[endpoint_mc_lattice_indices[3][0] % mc_lattice_size[0]
                                                                                                                                                                                     ][endpoint_mc_lattice_indices[3][1] % mc_lattice_size[1]][0], mc_lattice[endpoint_mc_lattice_indices[4][0] % mc_lattice_size[0]][endpoint_mc_lattice_indices[4][1] % mc_lattice_size[1]][0],
                                           mc_lattice[endpoint_mc_lattice_indices[5][0] % mc_lattice_size[0]][endpoint_mc_lattice_indices[5][1] % mc_lattice_size[1]][0]])
            # loop to define local lattice, based on endpoint and nearest neighhbour occupancy (direction indepenance), creates hop_probability_by_site a vector with probability/per site of hops
            for (endpoint_index, endpoint) in list(enumerate(endpoint_mc_lattice_indices)):
                # some endpoints have 2 valid paths, they will have increased probabilities
                probability_multiplier = 1
                hop_end_point = float(
                    mc_lattice[endpoint[0] % mc_lattice_size[0]][endpoint[1] % mc_lattice_size[1]][0])
                # is this endpoint clear? if not probability of a hop to it is zero
                if int(hop_end_point) == int(atom_types[1][0]):
                    hop_probability_by_site[endpoint_index] = 0
                    # vaccancy_count=vaccancy_count-1
                # local lattice is the 4x4x1 lattice that the cluster expansions know, the if statements are to make it direction independent, hop goes from 1,1 to 2,1 endpoint
                elif int(hop_end_point) == int(atom_types[1][1]):
                    # if both adjacent sites are empty, probability of going here is doubled
                    if endpoint_occupancy[endpoint_index - 1] == atom_types[1][1] and endpoint_occupancy[(endpoint_index + 1) % len(endpoint_occupancy)] == atom_types[1][1]:
                        probability_multiplier = 2
                    # if both sites adjacent to the hop are occupied, our atom follows a straight line path (dumbell) with a fixed probability
                    if endpoint_occupancy[endpoint_index - 1] == atom_types[1][0] and endpoint_occupancy[(endpoint_index + 1) % len(endpoint_occupancy)] == atom_types[1][0]:
                        hop_probability_by_site[endpoint_index] = dumbell_hop_probability
                        # dumbell_path_count+=1
                    else:
                        if endpoint_index == 0:
                            if endpoint_occupancy[5] == atom_types[1][1]:
                                local_lattice = np.array([[[mc_lattice[i - 1][(j + 1) % mc_lattice_size[1]][k]], [mc_lattice[i][(j + 1) % mc_lattice_size[1]][k]], [mc_lattice[(i + 1) % mc_lattice_size[0]][(j + 1) % mc_lattice_size[1]][k]], [mc_lattice[(i + 2) % mc_lattice_size[0]][(j + 1) % mc_lattice_size[0]][k]]], [[mc_lattice[i - 1][j][k]], [hop_start_point], [hop_end_point], [mc_lattice[(
                                    i + 2) % mc_lattice_size[0]][j][k]]], [[mc_lattice[i - 1][j - 1][k]], [mc_lattice[i][j - 1][k]], [mc_lattice[(i + 1) % mc_lattice_size[0]][j - 1][k]], [mc_lattice[(i + 2) % mc_lattice_size[0]][j - 1][k]]], [[mc_lattice[i - 1][j - 2][k]], [mc_lattice[i][j - 2][k]], [mc_lattice[(i + 1) % mc_lattice_size[0]][j - 2][k]], [mc_lattice[(i + 2) % mc_lattice_size[0]][j - 2][k]]]])  # good
                            elif endpoint_occupancy[1] == atom_types[1][1]:
                                local_lattice = np.array([[[mc_lattice[i][j - 1][k]], [mc_lattice[(i + 1) % mc_lattice_size[0]][j - 1][k]], [mc_lattice[(i + 2) % mc_lattice_size[0]][j - 1][k]], [mc_lattice[(i + 3) % mc_lattice_size[0]][j - 1][k]]], [[mc_lattice[i - 1][j][k]], [hop_start_point], [hop_end_point], [mc_lattice[(i + 2) % mc_lattice_size[0]][j][k]]], [[mc_lattice[i - 2][(j + 1) % mc_lattice_size[1]][k]], [
                                                         mc_lattice[i - 1][(j + 1) % mc_lattice_size[1]][k]], [mc_lattice[i][(j + 1) % mc_lattice_size[1]][k]], [mc_lattice[(i + 1) % mc_lattice_size[0]][(j + 1) % mc_lattice_size[1]][k]]], [[mc_lattice[i - 3][(j + 2) % mc_lattice_size[1]][k]], [mc_lattice[i - 2][(j + 2) % mc_lattice_size[1]][k]], [mc_lattice[i - 1][(j + 2) % mc_lattice_size[1]][k]], [mc_lattice[i][(j + 2) % mc_lattice_size[1]][k]]]])  # good
                        elif endpoint_index == 1:
                            if endpoint_occupancy[2] == atom_types[1][1]:
                                local_lattice = np.array([[[mc_lattice[(i + 1) % mc_lattice_size[0]][j - 1][k]], [mc_lattice[(i + 1) % mc_lattice_size[0]][j][k]], [mc_lattice[(i + 1) % mc_lattice_size[0]][(j + 1) % mc_lattice_size[1]][k]], [mc_lattice[(i + 1) % mc_lattice_size[0]][(j + 2) % mc_lattice_size[1]][k]]], [[mc_lattice[i][j - 1][k]], [hop_start_point], [hop_end_point], [mc_lattice[i][(
                                    j + 2) % mc_lattice_size[1]][k]]], [[mc_lattice[i - 1][j - 1][k]], [mc_lattice[i - 1][j][k]], [mc_lattice[i - 1][(j + 1) % mc_lattice_size[1]][k]], [mc_lattice[i - 1][(j + 2) % mc_lattice_size[1]][k]]], [[mc_lattice[i - 2][j - 1][k]], [mc_lattice[i - 2][j][k]], [mc_lattice[i - 2][(j + 1) % mc_lattice_size[1]][k]], [mc_lattice[i - 2][(j + 2) % mc_lattice_size[1]][k]]]])  # good
                            elif endpoint_occupancy[0] == atom_types[1][1]:
                                local_lattice = np.array([[[mc_lattice[i - 1][j][k]], [mc_lattice[i - 1][(j + 1) % mc_lattice_size[1]][k]], [mc_lattice[i - 1][(j + 2) % mc_lattice_size[1]][k]], [mc_lattice[i - 1][(j + 3) % mc_lattice_size[1]][k]]], [[mc_lattice[i][j - 1][k]], [hop_start_point], [hop_end_point], [mc_lattice[i][(j + 2) % mc_lattice_size[1]][k]]], [[mc_lattice[(i + 1) % mc_lattice_size[0]][j - 2][k]], [mc_lattice[(
                                    i + 1) % mc_lattice_size[0]][j - 1][k]], [mc_lattice[(i + 1) % mc_lattice_size[0]][j][k]], [mc_lattice[(i + 1) % mc_lattice_size[0]][(j + 1) % mc_lattice_size[1]][k]]], [[mc_lattice[(i + 2) % mc_lattice_size[0]][j - 3][k]], [mc_lattice[(i + 2) % mc_lattice_size[0]][j - 2][k]], [mc_lattice[(i + 2) % mc_lattice_size[0]][j - 1][k]], [mc_lattice[(i + 2) % mc_lattice_size[0]][j][k]]]])  # good
                        elif endpoint_index == 2:
                            if endpoint_occupancy[1] == atom_types[1][1]:
                                local_lattice = np.array([[[mc_lattice[i][j - 1][k]], [mc_lattice[i - 1][j][k]], [mc_lattice[i - 2][(j + 1) % mc_lattice_size[1]][k]], [mc_lattice[i - 3][(j + 2) % mc_lattice_size[1]][k]]], [[mc_lattice[(i + 1) % mc_lattice_size[0]][j - 1][k]], [hop_start_point], [hop_end_point], [mc_lattice[i - 2][(j + 2) % mc_lattice_size[1]][k]]], [[mc_lattice[(i + 2) % mc_lattice_size[0]][j - 1][k]], [mc_lattice[(
                                    i + 1) % mc_lattice_size[0]][j][k]], [mc_lattice[i][(j + 1) % mc_lattice_size[1]][k]], [mc_lattice[i - 1][(j + 2) % mc_lattice_size[1]][k]]], [[mc_lattice[(i + 2) % mc_lattice_size[0]][j][k]], [mc_lattice[(i + 1) % mc_lattice_size[0]][(j + 1) % mc_lattice_size[1]][k]], [mc_lattice[i][(j + 2) % mc_lattice_size[1]][k]], [mc_lattice[(i + 2) % mc_lattice_size[0]][(j + 2) % mc_lattice_size[1]][k]]]])  # good
                            elif endpoint_occupancy[3] == atom_types[1][1]:
                                local_lattice = np.array([[[mc_lattice[(i + 1) % mc_lattice_size[0]][j][k]], [mc_lattice[i][(j + 1) % mc_lattice_size[1]][k]], [mc_lattice[i - 1][(j + 2) % mc_lattice_size[1]][k]], [mc_lattice[i - 2][(j + 3) % mc_lattice_size[1]][k]]], [[mc_lattice[(i + 1) % mc_lattice_size[0]][j - 1][k]], [hop_start_point], [hop_end_point], [mc_lattice[i - 2][(
                                    j + 2) % mc_lattice_size[1]][k]]], [[mc_lattice[(i + 1) % mc_lattice_size[0]][j - 2][k]], [mc_lattice[i][j - 1][k]], [mc_lattice[i - 1][j][k]], [mc_lattice[i - 2][(j + 1) % mc_lattice_size[1]][k]]], [[mc_lattice[(i + 1) % mc_lattice_size[0]][j - 3][k]], [mc_lattice[i][j - 2][k]], [mc_lattice[i - 1][j - 1][k]], [mc_lattice[i - 2][j][k]]]])  # good
                        elif endpoint_index == 3:
                            if endpoint_occupancy[2] == atom_types[1][1]:
                                local_lattice = np.array([[[mc_lattice[(i + 1) % mc_lattice_size[0]][j - 1][k]], [mc_lattice[i][j - 1][k]], [mc_lattice[i - 1][j - 1][k]], [mc_lattice[i - 2][j - 1][k]]], [[mc_lattice[(i + 1) % mc_lattice_size[0]][j][k]], [hop_start_point], [hop_end_point], [mc_lattice[i - 2][j][k]]], [[mc_lattice[(i + 1) % mc_lattice_size[0]][(j + 1) % mc_lattice_size[1]][k]], [mc_lattice[i][(
                                    j + 1) % mc_lattice_size[1]][k]], [mc_lattice[i - 1][(j + 1) % mc_lattice_size[1]][k]], [mc_lattice[i - 2][(j + 1) % mc_lattice_size[1]][k]]], [[mc_lattice[(i + 1) % mc_lattice_size[0]][(j + 2) % mc_lattice_size[1]][k]], [mc_lattice[i][(j + 2) % mc_lattice_size[1]][k]], [mc_lattice[i - 1][(j + 2) % mc_lattice_size[1]][k]], [mc_lattice[i - 2][(j + 2) % mc_lattice_size[1]][k]]]])  # good
                            elif endpoint_occupancy[4] == atom_types[1][1]:
                                local_lattice = np.array([[[mc_lattice[i][(j + 1) % mc_lattice_size[1]][k]], [mc_lattice[i - 1][(j + 1) % mc_lattice_size[1]][k]], [mc_lattice[i - 2][(j + 1) % mc_lattice_size[1]][k]], [mc_lattice[i - 3][(j + 1) % mc_lattice_size[1]][k]]], [[mc_lattice[(i + 1) % mc_lattice_size[0]][j][k]], [hop_start_point], [hop_end_point], [mc_lattice[i - 2][j][k]]], [
                                                         [mc_lattice[(i + 2) % mc_lattice_size[0]][j - 1][k]], [mc_lattice[(i + 1) % mc_lattice_size[0]][j - 1][k]], [mc_lattice[i][j - 1][k]], [mc_lattice[i - 1][j - 1][k]]], [[mc_lattice[(i + 3) % mc_lattice_size[0]][j - 2][k]], [mc_lattice[(i + 2) % mc_lattice_size[0]][j - 2][k]], [mc_lattice[(i + 1) % mc_lattice_size[0]][j - 2][k]], [mc_lattice[i][j - 2][k]]]])  # good
                        elif endpoint_index == 4:
                            if endpoint_occupancy[3] == atom_types[1][1]:
                                local_lattice = np.array([[[mc_lattice[(i + 1) % mc_lattice_size[0]][j][k]], [mc_lattice[(i + 1) % mc_lattice_size[0]][j - 1][k]], [mc_lattice[(i + 1) % mc_lattice_size[0]][j - 2][k]], [mc_lattice[(i + 1) % mc_lattice_size[0]][j - 3][k]]], [[mc_lattice[i][(j + 1) % mc_lattice_size[1]][k]], [hop_start_point], [hop_end_point], [mc_lattice[i][j - 2][k]]], [
                                                         [mc_lattice[i - 1][(j + 2) % mc_lattice_size[1]][k]], [mc_lattice[i - 1][(j + 1) % mc_lattice_size[0]][k]], [mc_lattice[i - 1][j][k]], [mc_lattice[i - 1][j - 1][k]]], [[mc_lattice[i - 2][(j + 3) % mc_lattice_size[1]][k]], [mc_lattice[i - 2][(j + 2) % mc_lattice_size[1]][k]], [mc_lattice[i - 2][(j + 1) % mc_lattice_size[1]][k]], [mc_lattice[i - 2][j][k]]]])  # good
                            elif endpoint_occupancy[5] == atom_types[1][1]:
                                local_lattice = np.array([[[mc_lattice[i - 1][(j + 1) % mc_lattice_size[1]][k]], [mc_lattice[i - 1][j][k]], [mc_lattice[i - 1][j - 1][k]], [mc_lattice[i - 2][j - 2][k]]], [[mc_lattice[i][(j + 1) % mc_lattice_size[1]][k]], [hop_start_point], [hop_end_point], [mc_lattice[i][j - 2][k]]], [[mc_lattice[(i + 1) % mc_lattice_size[0]][(j + 1) % mc_lattice_size[1]][k]], [mc_lattice[(
                                    i + 1) % mc_lattice_size[0]][j][k]], [mc_lattice[(i + 1) % mc_lattice_size[0]][j - 1][k]], [mc_lattice[(i + 1) % mc_lattice_size[0]][j - 2][k]]], [[mc_lattice[(i + 2) % mc_lattice_size[0]][(j + 1) % mc_lattice_size[1]][k]], [mc_lattice[(i + 2) % mc_lattice_size[0]][j][k]], [mc_lattice[(i + 2) % mc_lattice_size[0]][j - 1][k]], [mc_lattice[(i + 2) % mc_lattice_size[0]][j - 2][k]]]])  # good
                        elif endpoint_index == 5:
                            if endpoint_occupancy[4] == atom_types[1][1]:
                                local_lattice = np.array([[[mc_lattice[i][(j + 1) % mc_lattice_size[1]][k]], [mc_lattice[(i + 1) % mc_lattice_size[0]][j][k]], [mc_lattice[(i + 2) % mc_lattice_size[0]][j - 1][k]], [mc_lattice[(i + 3) % mc_lattice_size[0]][j - 2][k]]], [[mc_lattice[i - 1][(j + 1) % mc_lattice_size[1]][k]], [hop_start_point], [hop_end_point], [mc_lattice[(
                                    i + 2) % mc_lattice_size[0]][j - 2][k]]], [[mc_lattice[i - 2][(j + 1) % mc_lattice_size[1]][k]], [mc_lattice[i - 1][j][k]], [mc_lattice[i][j - 1][k]], [mc_lattice[(i + 1) % mc_lattice_size[0]][j - 2][k]]], [[mc_lattice[i - 3][(j + 1) % mc_lattice_size[1]][k]], [mc_lattice[i - 2][j][k]], [mc_lattice[i - 1][j - 1][k]], [mc_lattice[i][j - 2][k]]]])  # good
                            elif endpoint_occupancy[0] == atom_types[1][1]:
                                local_lattice = np.array([[[mc_lattice[i - 1][j][k]], [mc_lattice[i][j - 1][k]], [mc_lattice[(i + 1) % mc_lattice_size[0]][j - 2][k]], [mc_lattice[(i + 2) % mc_lattice_size[0]][j - 3][k]]], [[mc_lattice[i - 1][(j + 1) % mc_lattice_size[1]][k]], [hop_start_point], [hop_end_point], [mc_lattice[(i + 2) % mc_lattice_size[0]][j - 2][k]]], [[mc_lattice[i - 1][(j + 2) % mc_lattice_size[1]][k]], [
                                                         mc_lattice[i][(j + 1) % mc_lattice_size[1]][k]], [mc_lattice[(i + 1) % mc_lattice_size[0]][j][k]], [mc_lattice[(i + 2) % mc_lattice_size[0]][j - 1][k]]], [[mc_lattice[i - 1][(j + 3) % mc_lattice_size[1]][k]], [mc_lattice[i][(j + 2) % mc_lattice_size[1]][k]], [mc_lattice[(i + 1) % mc_lattice_size[0]][(j + 1) % mc_lattice_size[1]][k]], [mc_lattice[(i + 2) % mc_lattice_size[0]][j][k]]]])  # good

                        hop_probability_by_site[endpoint_index] = probability_multiplier * get_hop_probability(np.transpose(
                            local_lattice, (1, 0, 2)), prefactor, kb_t)  # gets the themal probability of the hop in question

                        # tetrahedral_path_count+=1*probability_multiplier
                endpoint_index = endpoint_index + 1
            hop_probability_matrix[i][j][k] = hop_probability_by_site[:]

        # if there is no atom to move,hop probabilities are zero
        elif int(hop_start_point) == int(atom_types[1][1]):
            hop_probability_matrix[i][j][k] = np.zeros(6)
    return hop_probability_matrix


def kmc_step(mc_lattice, input_distance_lattice, input_hop_probability_matrix, use_presimulation_mode):
    """main monte carlo loop step
    runs KMC evolve as many times as there are atoms, records all hops in distance_lattice, and updates the distance_lattice as it goes. if use_presimulation_mode is true, the distances and times will not be updated
    """
    distance_lattice = np.array(
        input_distance_lattice)  # the distance_lattice from the previous run, lattice filled with the starting indices of each atom located in the position they are now
    # the hop probability lattice associated with this position
    hop_probability_matrix = np.array(input_hop_probability_matrix)
    time_step_per_hop = 0  # how long for individual hop
    time_step_per_kmcstep = 0  # how long for this step
    k = 0  # 2d case

    # do as many steps as there are ions
    for hop in range(int(mc_lattice_size[0] * mc_lattice_size[1] * mc_lattice_size[2] * Li_concentration)):
        # gamma as defined in VDVen paper,
        gamma = np.sum(hop_probability_matrix)
        # a list of probabilities, 6 for each atom one for each possible endpoint.  Monotonically increasing (cumsum)from 0-1 (normalized by the gamma)
        normalized_hop_probabilities = np.cumsum(
            hop_probability_matrix) / gamma
        rho = np.random.uniform(0, 1)  # the random number
        hop_master_index = np.searchsorted(
            normalized_hop_probabilities, rho)  # get master index of hop
        # get which endpoint the ion hopped to (number between 0-5)
        endpoint_index = hop_master_index % 6
        # mc_lattice indices of hop start point
        hopping_ion_i = (hop_master_index // 6) // mc_lattice_size[0]
        hopping_ion_j = (hop_master_index // 6) % mc_lattice_size[0]
        hop_endpoints = np.array([[hopping_ion_i + 1, hopping_ion_j], [hopping_ion_i, hopping_ion_j + 1], [hopping_ion_i - 1, hopping_ion_j + 1], [hopping_ion_i - 1,
                                                                                                                                                   hopping_ion_j], [hopping_ion_i, hopping_ion_j - 1], [hopping_ion_i + 1, hopping_ion_j - 1]])  # all of the possible endpoint indices for the hop
        hop_endpoint_i = hop_endpoints[endpoint_index][0]
        hop_endpoint_j = hop_endpoints[endpoint_index][1]
        # moving vaccancy to initial ion location on the mc_lattice
        mc_lattice[hopping_ion_i][hopping_ion_j][k] = atom_types[1][1]
        mc_lattice[hop_endpoint_i % mc_lattice_size[0]][hop_endpoint_j % mc_lattice_size[1]
                                                        ][0] = atom_types[1][0]  # moving ion to appropriate endpoint mc_lattice
        if use_presimulation_mode == False:
            # moving the ion's coordinates (eg [2,3,0]) to the appropriate site,
            distance_lattice[hop_endpoint_i % mc_lattice_size[0]][hop_endpoint_j %
                                                                  mc_lattice_size[1]][k] = distance_lattice[hopping_ion_i][hopping_ion_j][k]
            distance_lattice[hop_endpoint_i % mc_lattice_size[0]][hop_endpoint_j % mc_lattice_size[1]][k][2] = distance_lattice[hop_endpoint_i % mc_lattice_size[0]][hop_endpoint_j % mc_lattice_size[1]][k][2] + 100 * (
                hop_endpoint_i // mc_lattice_size[0]) + hop_endpoint_j // mc_lattice_size[1]  # adds index to distance lattice for periodic boundary condition purposes, 3rd value is 100 times a loop in i, +single times loops in j
            # clears the coordiates from the old site ->sticks in a vaccancy
            distance_lattice[hopping_ion_i][hopping_ion_j][k] = [-1, 0, 0]
            # time of the hop, as defined in VdV paper
            time_step_per_hop = (-1. / gamma) * np.log(np.random.uniform(0, 1))
            time_step_per_kmcstep = time_step_per_kmcstep + \
                time_step_per_hop  # total time of the step
        hop_probability_matrix = kmc_evolve(mc_lattice, hop_probability_matrix, [hopping_ion_i, hopping_ion_j, 0], [
                                            hop_endpoint_i, hop_endpoint_j, 0], use_initialize_mode)  # update the probability lattice after the hop
        # hop_histogram[endpoint_index]+=1        return (mc_lattice,hop_probability_matrix)
    if use_presimulation_mode == False:
        return (mc_lattice, distance_lattice, time_step_per_kmcstep, hop_probability_matrix)
    elif use_presimulation_mode == True:
        return(mc_lattice, hop_probability_matrix)


def colorsquare(s, filename):  # to plot the lattices as you go, and save as png's
    """Figure animation stuff"""
    plt.ion()
    plt.imshow(s, cmap='gray', interpolation='none')
    plt.show()
    plt.savefig(filename)
    plt.pause(1)
    return


"""To get the diffusion coeffficient once simulation is done"""


def get_distance(input_lattice, mc_lattice_size, lattice_constant):
    """function that takes thefull input/distance_lattice of coordinates at the end of the monte carlo and returns an array with values of how far all of the ions have travelled
    """
    distances_travelled_matrix = np.zeros([mc_lattice_size[0], mc_lattice_size[1], mc_lattice_size[2]]
                                          )  # array the same size as the mc_lattice, to be filled with the starting coordinates of each atom, in the location it currently occupies
    # list of the net distances travelled, loses starting point info
    distances_travelled_list = []
    # sum of deltat i and delta j coordinates of all atoms
    d_j_distance_ij = np.array([0., 0.])
    for i, j, k in itertools.product(range(mc_lattice_size[0]), range(mc_lattice_size[1]), range(mc_lattice_size[2])):
        atom_starting_point_indices = input_lattice[i][j][k]
        # is there an atom in the site?  for vaccancies the starting point index will be [-1,0,0]
        if atom_starting_point_indices[0] >= 0:
            # adding to deal with atoms that have looped around the periodic boundaries
            i_index = i + \
                mc_lattice_size[0] * \
                round(float(atom_starting_point_indices[2]) / 100)
            j_index = j + mc_lattice_size[1] * (atom_starting_point_indices[2] - round(
                float(atom_starting_point_indices[2]) / 100) * 100)
            distances_travelled_matrix[i][j][k] = np.sqrt(np.abs((atom_starting_point_indices[0] - i_index)**2 + (atom_starting_point_indices[1] - j_index)**2 + (
                atom_starting_point_indices[0] - i_index) * (atom_starting_point_indices[1] - j_index)))  # hexagonal distance between atom start point and where it is now, matrix form
            # distance travelled per atom, in list form, no order to it though
            distances_travelled_list = np.append(
                distances_travelled_list, distances_travelled_matrix[i][j][k])
            d_j_distance_ij += [(atom_starting_point_indices[0] - i_index),
                                (atom_starting_point_indices[1] - j_index)]
    d_j_distance = np.sqrt(np.abs(d_j_distance_ij[0]**2 + (d_j_distance_ij[1]) **
                                  2 + d_j_distance_ij[0] * d_j_distance_ij[1]))  # vector sum of all distances
    return distances_travelled_matrix * lattice_constant, distances_travelled_list * lattice_constant, d_j_distance * lattice_constant


def get_diffusion_coefficient(distance_lattice, mc_lattice_size, time_diffusing, lattice_constant, Li_concentration):
    """function that takes input distance_lattice and time fromm KMC cycle and returns the diffusion Coefficient"""
    number_of_atoms = mc_lattice_size[0] * \
        mc_lattice_size[1] * mc_lattice_size[2] * Li_concentration
    if number_of_atoms == 0:
        sys.exit("concentration is zero, no atoms to move!")
    distances_travelled_matrix, distances_travelled_list, d_j_distance = get_distance(
        distance_lattice, mc_lattice_size, lattice_constant)

    # need to divide time by number of atoms, 2*2 is for the fact that this thing is 2D, diffusion_star because that is what it is in the paper D*
    diffusion_star = 1 / (2 * 2 * time_diffusing / number_of_atoms) * \
        np.average(np.square(distances_travelled_list))
    diffusion_j = 1 / (2 * 2 * time_diffusing / number_of_atoms) * \
        (d_j_distance**2) / number_of_atoms
    return diffusion_star, distances_travelled_list, diffusion_j


# play_lattice=np.transpose(np.array([[[-1,-1,-1,1],[-1,1,-1,1],[1,-1,1,1],[1,1,-1,-1]]])*-1.)
# play_lattice=np.transpose(np.array([[[1.,1.,1.,1.],[1.,-1.,1.,1.],[1.,1.,1.,1.],[1.,1.,1.,1.]]])*1)
#x= get_config_energy(play_lattice)
#y= get_excited_energy(play_lattice)
# print x
# print y
# print y-x


# loads the dictionary
master_dictionary = pickle.load(open("master_dictionary.p", "rb"))


diffusion_coefficient_vs_concentration = []
diffusion_coefficient_j_vs_concentration = []
# for averaging_iterations in [1]:#range(1, 20,2):
averaging_iterations = 100  # loop over this
mc_lattice_size = [0, 0, 0]
dimensions = [50]  # loop over this
simulation_iterations = 2000  # how many steps
presimulation_iterations = 1000  # how many pre simulation steps

concentrations = np.array(range(50,150,2))/500. # looping over these too
average_path_counts = []

output_file = open('LiCoO2_kmc.out', 'w')  # the results file
output_file.write('Size:' + str(mc_lattice_size) + '\n kbT:' + str(kb_t) + '\n Simulation parameters: \n Presimulation cycles: ' +
                  str(presimulation_iterations) + '\n Simulation Cycles: ' + str(simulation_iterations) + '\n')
output_file.write("\n RESULTS \n")

# range(5,65,8): #to check size dependence of parameters
for dimension in dimensions:
    # initializing lattice sizes, other size dependent parameters
    mc_lattice_size[0] = dimension
    mc_lattice_size[1] = dimension
    mc_lattice_size[2] = 1
    hop_probability_matrix = np.tile(
        np.zeros(6), (mc_lattice_size[0], mc_lattice_size[1], mc_lattice_size[2], 1))
    n_sites = mc_lattice_size[0] * mc_lattice_size[1] * mc_lattice_size[2]

    for Li_concentration in concentrations:
        print ("concentration is: " + str(Li_concentration))
        averageing_iteration = 0
        # where the diffusion_coefficients that will be averaged go, one list per concentration with as many values as averaging cycles
        diffusion_coefficient_averaging = []
        diffusion_coefficient_j_averaging = []  # same thing but for other coefficient

        while averageing_iteration <= averaging_iterations:
            print "concentration, averaging step is: " + str(Li_concentration) + ", " + str(averageing_iteration)
            distance_lattice = np.tile(
                [-1, 0, 0], (mc_lattice_size[0], mc_lattice_size[1], mc_lattice_size[2], 1))
            averaging_step = 0
            simulation = 0
            total_time = 0
            """populate the lattice with the right concentration"""
            mc_lattice = np.zeros(
                [mc_lattice_size[0], mc_lattice_size[1], mc_lattice_size[2]])  # creates the mc_lattice
            # puts vacancies all over the lattice
            mc_lattice = mc_lattice + np.float32(atom_types[1][1])
            Li_atoms = 0
            while Li_atoms <= Li_concentration * n_sites - 1:
                i = np.random.randint(0, mc_lattice_size[0])
                j = np.random.randint(0, mc_lattice_size[1])
                k = np.random.randint(0, mc_lattice_size[2])
                if mc_lattice[i][j][k] == atom_types[1][1]:
                    mc_lattice[i][j][k] = np.float32(atom_types[1][0])
                    Li_atoms += 1
            use_initialize_mode = True
            # initialize_hop_probability_matrix(mc_lattice) #initializing the hop probabilities
            hop_probability_matrix = kmc_evolve(
                mc_lattice, hop_probability_matrix, 0, 0, use_initialize_mode)
            use_initialize_mode = False
            use_presimulation_mode = True
            # initializes the mc_lattice for the simulation to avoid starting artifacts
            while averaging_step < presimulation_iterations:
                (mc_lattice, hop_probability_matrix) = kmc_step(
                    mc_lattice, distance_lattice, hop_probability_matrix, use_presimulation_mode)
                if averaging_step % 50 == 0:
                    print "presimulation step:" + str(averaging_step)
                # colorsquare(np.transpose(mc_lattice)[0],"lattice_pictures/"+str(averaging_step)+".png") #to see the lattice
                averaging_step += 1
            use_presimulation_mode = False

            # need to initialize the distance_lattice after presimulation
            for i, j, k in itertools.product(range(mc_lattice_size[0]), range(mc_lattice_size[1]), range(mc_lattice_size[2])):
                if mc_lattice[i][j][k] == atom_types[1][0]:
                    # put in a starting coordinate tag for every ion
                    distance_lattice[i][j][k] = list([i, j, k])
                else:
                    # puts a void location at every vaccancy
                    distance_lattice[i][j][k] = [-1, 0, 0]
            # r_squared_vs_time=[]
            # x_axis_times=[]

            """THE KMC LOOP"""
            while simulation < simulation_iterations:  # how many kmc steps to take
                # colorsquare(np.transpose(mc_lattice)[0],"lattice_pictures/"+str(simulation+10)+".png")

                (mc_lattice, distance_lattice, time_step, hop_probability_matrix) = kmc_step(
                    mc_lattice, distance_lattice, hop_probability_matrix, use_presimulation_mode)  # perform 1 KMC step
                if simulation % 50 == 0:  # show how far we've gone in the simulation
                    print "simulation step: " + str(simulation)
                #colorsquare(np.transpose(mc_lattice)[0], "lattice_pictures/" + str(simulation + 10) + ".png")
                total_time = total_time + time_step  # the total time taken
                # diffusion_cycle,distances_travelled_list,diffusion_j=get_diffusion_coefficient(distance_lattice,mc_lattice_size,total_time,lattice_constant,Li_concentration) #getting the diffusion Coefficient, and the distances travelled by each atom
                # r_squared_vs_time=np.append(r_squared_vs_time,np.average(distances_travelled_list))
                # x_axis_times=np.append(x_axis_times,total_time)
                simulation += 1
            print "total time is: " + str(total_time)
            (diffusion_coefficient_cycle, tmp, diffusion_j_cycle) = get_diffusion_coefficient(distance_lattice, mc_lattice_size,
                                                                                              total_time, lattice_constant, Li_concentration)  # diffusion_coefficient_cycle is obtained for every averaging step
            diffusion_coefficient_averaging = np.append(
                diffusion_coefficient_averaging, diffusion_coefficient_cycle)
            diffusion_coefficient_j_averaging = np.append(
                diffusion_coefficient_j_averaging, diffusion_j_cycle)
            # plt.figure()
            # averaging_value=25
            #average_distances=np.mean(r_squared_vs_time.reshape(-1, averaging_value), axis=1)
            #distances_errors=np.std(r_squared_vs_time.reshape(-1, averaging_value), axis=1)
            # plt.errorbar(x_axis_times[::averaging_value],average_distances,distances_errors)
            # plt.plot(range(simulation_iterations)[::averaging_value],average_distances)
            # plt.show()
            # plt.pause(10000)
            averageing_iteration += 1
        output_file.write("Li concentration: " + str(Li_concentration) +
                          ". Diffusion_coefficients at this concentration: " + str(diffusion_coefficient_averaging) + "\n")
        output_file.write("Li concentration: " + str(Li_concentration) +
                          ". Diffusion_coefficient J at this concentration: " + str(diffusion_coefficient_j_averaging) + "\n")
        final_diffusion_coefficient = [np.average(
            diffusion_coefficient_averaging), np.std(diffusion_coefficient_averaging)]
        final_diffusion_coefficient_j = [np.average(
            diffusion_coefficient_j_averaging), np.std(diffusion_coefficient_j_averaging)]

        diffusion_coefficient_vs_concentration = np.append(diffusion_coefficient_vs_concentration, [
                                                           final_diffusion_coefficient[0], Li_concentration, final_diffusion_coefficient[1]])  # need to put in masterloop variable here
        diffusion_coefficient_j_vs_concentration = np.append(diffusion_coefficient_j_vs_concentration, [
                                                             final_diffusion_coefficient_j[0], Li_concentration, final_diffusion_coefficient_j[1]])  # need to put in masterloop variable here


"""Stop timer"""
stop_timer = time.time()

output_file.write('Averaged Diffusion star Coefficients:' + str(diffusion_coefficient_vs_concentration[::3]) + '\n Variables: ' + str(
    diffusion_coefficient_vs_concentration[1::3]) + '\n Errors on D_star: ' + str(diffusion_coefficient_vs_concentration[2::3]))
output_file.write('Averaged Diffusion J Coefficients:' + str(diffusion_coefficient_j_vs_concentration[::3]) + '\n Variables: ' + str(
    diffusion_coefficient_j_vs_concentration[1::3]) + '\n Errors on D_star: ' + str(diffusion_coefficient_j_vs_concentration[2::3]))
output_file.write("\n\n Total Time: " + str(stop_timer - start_timer))

output_file.write("\n\n\n END\n")
output_file.close()


print "diffusion_coefficient_vs_concentration is: " + str(diffusion_coefficient_vs_concentration)
print "diffusion_coefficient_j_vs_concentration is: " + str(diffusion_coefficient_j_vs_concentration)


plt.figure()
# plt.plot(concentrations,average_path_counts[::3])
# plt.plot(concentrations,average_path_counts[1::3])
# plt.plot(concentrations,average_path_counts[2::3])
# plt.plot(range(0,averaging_iterations+1),diffusion_coefficient)
plt.errorbar(diffusion_coefficient_vs_concentration[1::3], np.log(
    diffusion_coefficient_vs_concentration[::3]), diffusion_coefficient_vs_concentration[2::3] / diffusion_coefficient_vs_concentration[::3])
plt.errorbar(diffusion_coefficient_j_vs_concentration[1::3], np.log(
    diffusion_coefficient_j_vs_concentration[::3]), diffusion_coefficient_j_vs_concentration[2::3] / diffusion_coefficient_vs_concentration[::3])
plt.xlabel("Li Concentration")
plt.ylabel("diffusion coefficient (cm^2/s)")
plt.savefig("LiCoO2_kmc.png")

#


"""
def initialize_hop_probability_matrix(mc_lattice):
    function that creates the hop_probability_matrix for the entire lattice
    hop_probability_matrix=np.tile(np.zeros(6),(size[0],size[1],size[2],1))

    for i,j,k in itertools.product(range(size[0]),range(size[1]),range(size[2])):#looping over elements in the array to calculate all of the probabilities. write them into hop_probability_matrix
        hop_start_point=float(mc_lattice[i][j][k])
        hop_probability_by_site=np.zeros(6)
        if int(hop_start_point)==int(atom_types[1][0]):#is there an ion on the site at the start of the cycle?
            endpoints=np.array([[i+1,j],[i,j+1],[i-1,j+1],[i-1,j],[i,j-1],[i+1,j-1]]) #endpoint that the ion can hop to
            endpoint_occupancy=np.array([mc_lattice[endpoints[0][0]%size[0]][endpoints[0][1]%size[1]][0],mc_lattice[endpoints[1][0]%size[0]][endpoints[1][1]%size[1]][0],mc_lattice[endpoints[2][0]%size[0]][endpoints[2][1]%size[1]][0],mc_lattice[endpoints[3][0]%size[0]][endpoints[3][1]%size[1]][0],mc_lattice[endpoints[4][0]%size[0]][endpoints[4][1]%size[1]][0],mc_lattice[endpoints[5][0]%size[0]][endpoints[5][1]%size[1]][0]])
            #dumbell_path_count=0
            #tetrahedral_path_count=0
            #vaccancy_count=len(endpoints)
            for (endpoint_index, endpoint) in list(enumerate(endpoints)): #this loop defines local lattice, based on endpoint (direction independent), creates hop_probability_by_site a vector for each site with the probability of the ion jumping to each endpoint
                probability_multiplier=1#there are two paths to some of the endpoints...
                hop_end_point=float(mc_lattice[endpoint[0]%size[0]][endpoint[1]%size[1]][0])

                if int(hop_end_point)==int(atom_types[1][0]):#is this endpoint clear? if not probability of a hop is zero
                    hop_probability_by_site[endpoint_index]=0
                    #vaccancy_count=vaccancy_count-1
                elif hop_end_point==atom_types[1][1]: #local lattice is the 4x4x1 lattice that the cluster expansions know, the if statements are to make it direction independent, hop goes from 1,1 to 2,1 endpoint
                    if endpoint_occupancy[endpoint_index-1]==atom_types[1][1] and endpoint_occupancy[(endpoint_index+1)%len(endpoint_occupancy)]==atom_types[1][1]:#if both adjacent sites are empty, probability of going here is doubled
                        probability_multiplier=2
                    if endpoint_occupancy[endpoint_index-1]==atom_types[1][0] and endpoint_occupancy[(endpoint_index+1)%len(endpoint_occupancy)]==atom_types[1][0]:  #if both sites adjacent to the hop are occupied, our atom follows a straight line path (dumbell) with a fixed probability
                        hop_probability_by_site[endpoint_index]=dumbell_hop_probability
                        #dumbell_path_count+=1
                    else:
                        if endpoint_index==0:
                            if endpoint_occupancy[5]==atom_types[1][1]:
                                local_lattice=np.array([[[mc_lattice[i-1][(j+1)%size[1]][k]],[mc_lattice[i][(j+1)%size[1]][k]],[mc_lattice[(i+1)%size[0]][(j+1)%size[1]][k]],[mc_lattice[(i+2)%size[0]][(j+1)%size[0]][k]]],[[mc_lattice[i-1][j][k]],[hop_start_point],[hop_end_point],[mc_lattice[(i+2)%size[0]][j][k]]],[[mc_lattice[i-1][j-1][k]],[mc_lattice[i][j-1][k]],[mc_lattice[(i+1)%size[0]][j-1][k]],[mc_lattice[(i+2)%size[0]][j-1][k]]],[[mc_lattice[i-1][j-2][k]],[mc_lattice[i][j-2][k]],[mc_lattice[(i+1)%size[0]][j-2][k]],[mc_lattice[(i+2)%size[0]][j-2][k]]]]) #good
                            elif endpoint_occupancy[1]==atom_types[1][1]:
                                local_lattice=np.array([[[mc_lattice[i][j-1][k]],[mc_lattice[(i+1)%size[0]][j-1][k]],[mc_lattice[(i+2)%size[0]][j-1][k]],[mc_lattice[(i+3)%size[0]][j-1][k]]],[[mc_lattice[i-1][j][k]],[hop_start_point],[hop_end_point],[mc_lattice[(i+2)%size[0]][j][k]]],[[mc_lattice[i-2][(j+1)%size[1]][k]],[mc_lattice[i-1][(j+1)%size[1]][k]],[mc_lattice[i][(j+1)%size[1]][k]],[mc_lattice[(i+1)%size[0]][(j+1)%size[1]][k]]],[[mc_lattice[i-3][(j+2)%size[1]][k]],[mc_lattice[i-2][(j+2)%size[1]][k]],[mc_lattice[i-1][(j+2)%size[1]][k]],[mc_lattice[i][(j+2)%size[1]][k]]]]) #good
                        elif endpoint_index==1:
                            if endpoint_occupancy[2]==atom_types[1][1]:
                                local_lattice=np.array([[[mc_lattice[(i+1)%size[0]][j-1][k]],[mc_lattice[(i+1)%size[0]][j][k]],[mc_lattice[(i+1)%size[0]][(j+1)%size[1]][k]],[mc_lattice[(i+1)%size[0]][(j+2)%size[1]][k]]],[[mc_lattice[i][j-1][k]],[hop_start_point],[hop_end_point],[mc_lattice[i][(j+2)%size[1]][k]]],[[mc_lattice[i-1][j-1][k]],[mc_lattice[i-1][j][k]],[mc_lattice[i-1][(j+1)%size[1]][k]],[mc_lattice[i-1][(j+2)%size[1]][k]]],[[mc_lattice[i-2][j-1][k]],[mc_lattice[i-2][j][k]],[mc_lattice[i-2][(j+1)%size[1]][k]],[mc_lattice[i-2][(j+2)%size[1]][k]]]]) #good
                            elif endpoint_occupancy[0]==atom_types[1][1]:
                                local_lattice=np.array([[[mc_lattice[i-1][j][k]],[mc_lattice[i-1][(j+1)%size[1]][k]],[mc_lattice[i-1][(j+2)%size[1]][k]],[mc_lattice[i-1][(j+3)%size[1]][k]]],[[mc_lattice[i][j-1][k]],[hop_start_point],[hop_end_point],[mc_lattice[i][(j+2)%size[1]][k]]],[[mc_lattice[(i+1)%size[0]][j-2][k]],[mc_lattice[(i+1)%size[0]][j-1][k]],[mc_lattice[(i+1)%size[0]][j][k]],[mc_lattice[(i+1)%size[0]][(j+1)%size[1]][k]]],[[mc_lattice[(i+2)%size[0]][j-3][k]],[mc_lattice[(i+2)%size[0]][j-2][k]],[mc_lattice[(i+2)%size[0]][j-1][k]],[mc_lattice[(i+2)%size[0]][j][k]]]]) #good
                        elif endpoint_index==2:
                            if endpoint_occupancy[1]==atom_types[1][1]:
                                local_lattice=np.array([[[mc_lattice[i][j-1][k]],[mc_lattice[i-1][j][k]],[mc_lattice[i-2][(j+1)%size[1]][k]],[mc_lattice[i-3][(j+2)%size[1]][k]]],[[mc_lattice[(i+1)%size[0]][j-1][k]],[hop_start_point],[hop_end_point],[mc_lattice[i-2][(j+2)%size[1]][k]]],[[mc_lattice[(i+2)%size[0]][j-1][k]],[mc_lattice[(i+1)%size[0]][j][k]],[mc_lattice[i][(j+1)%size[1]][k]],[mc_lattice[i-1][(j+2)%size[1]][k]]],[[mc_lattice[(i+2)%size[0]][j][k]],[mc_lattice[(i+1)%size[0]][(j+1)%size[1]][k]],[mc_lattice[i][(j+2)%size[1]][k]],[mc_lattice[(i+2)%size[0]][(j+2)%size[1]][k]]]])#good
                            elif endpoint_occupancy[3]==atom_types[1][1]:
                                local_lattice=np.array([[[mc_lattice[(i+1)%size[0]][j][k]],[mc_lattice[i][(j+1)%size[1]][k]],[mc_lattice[i-1][(j+2)%size[1]][k]],[mc_lattice[i-2][(j+3)%size[1]][k]]],[[mc_lattice[(i+1)%size[0]][j-1][k]],[hop_start_point],[hop_end_point],[mc_lattice[i-2][(j+2)%size[1]][k]]],[[mc_lattice[(i+1)%size[0]][j-2][k]],[mc_lattice[i][j-1][k]],[mc_lattice[i-1][j][k]],[mc_lattice[i-2][(j+1)%size[1]][k]]],[[mc_lattice[(i+1)%size[0]][j-3][k]],[mc_lattice[i][j-2][k]],[mc_lattice[i-1][j-1][k]],[mc_lattice[i-2][j][k]]]]) #good
                        elif endpoint_index==3:
                            if endpoint_occupancy[2]==atom_types[1][1]:
                                local_lattice=np.array([[[mc_lattice[(i+1)%size[0]][j-1][k]],[mc_lattice[i][j-1][k]],[mc_lattice[i-1][j-1][k]],[mc_lattice[i-2][j-1][k]]],[[mc_lattice[(i+1)%size[0]][j][k]],[hop_start_point],[hop_end_point],[mc_lattice[i-2][j][k]]],[[mc_lattice[(i+1)%size[0]][(j+1)%size[1]][k]],[mc_lattice[i][(j+1)%size[1]][k]],[mc_lattice[i-1][(j+1)%size[1]][k]],[mc_lattice[i-2][(j+1)%size[1]][k]]],[[mc_lattice[(i+1)%size[0]][(j+2)%size[1]][k]],[mc_lattice[i][(j+2)%size[1]][k]],[mc_lattice[i-1][(j+2)%size[1]][k]],[mc_lattice[i-2][(j+2)%size[1]][k]]]]) #good
                            elif endpoint_occupancy[4]==atom_types[1][1]:
                                local_lattice=np.array([[[mc_lattice[i][(j+1)%size[1]][k]],[mc_lattice[i-1][(j+1)%size[1]][k]],[mc_lattice[i-2][(j+1)%size[1]][k]],[mc_lattice[i-3][(j+1)%size[1]][k]]],[[mc_lattice[(i+1)%size[0]][j][k]],[hop_start_point],[hop_end_point],[mc_lattice[i-2][j][k]]],[[mc_lattice[(i+2)%size[0]][j-1][k]],[mc_lattice[(i+1)%size[0]][j-1][k]],[mc_lattice[i][j-1][k]],[mc_lattice[i-1][j-1][k]]],[[mc_lattice[(i+3)%size[0]][j-2][k]],[mc_lattice[(i+2)%size[0]][j-2][k]],[mc_lattice[(i+1)%size[0]][j-2][k]],[mc_lattice[i][j-2][k]]]]) #good
                        elif endpoint_index==4:
                            if endpoint_occupancy[3]==atom_types[1][1]:
                                local_lattice=np.array([[[mc_lattice[(i+1)%size[0]][j][k]],[mc_lattice[(i+1)%size[0]][j-1][k]],[mc_lattice[(i+1)%size[0]][j-2][k]],[mc_lattice[(i+1)%size[0]][j-3][k]]],[[mc_lattice[i][(j+1)%size[1]][k]],[hop_start_point],[hop_end_point],[mc_lattice[i][j-2][k]]],[[mc_lattice[i-1][(j+2)%size[1]][k]],[mc_lattice[i-1][(j+1)%size[0]][k]],[mc_lattice[i-1][j][k]],[mc_lattice[i-1][j-1][k]]],[[mc_lattice[i-2][(j+3)%size[1]][k]],[mc_lattice[i-2][(j+2)%size[1]][k]],[mc_lattice[i-2][(j+1)%size[1]][k]],[mc_lattice[i-2][j][k]]]])#good
                            elif endpoint_occupancy[5]==atom_types[1][1]:
                                local_lattice=np.array([[[mc_lattice[i-1][(j+1)%size[1]][k]],[mc_lattice[i-1][j][k]],[mc_lattice[i-1][j-1][k]],[mc_lattice[i-2][j-2][k]]],[[mc_lattice[i][(j+1)%size[1]][k]],[hop_start_point],[hop_end_point],[mc_lattice[i][j-2][k]]],[[mc_lattice[(i+1)%size[0]][(j+1)%size[1]][k]],[mc_lattice[(i+1)%size[0]][j][k]],[mc_lattice[(i+1)%size[0]][j-1][k]],[mc_lattice[(i+1)%size[0]][j-2][k]]],[[mc_lattice[(i+2)%size[0]][(j+1)%size[1]][k]],[mc_lattice[(i+2)%size[0]][j][k]],[mc_lattice[(i+2)%size[0]][j-1][k]],[mc_lattice[(i+2)%size[0]][j-2][k]]]])#good
                        elif endpoint_index==5:
                            if endpoint_occupancy[4]==atom_types[1][1]:
                                local_lattice=np.array([[[mc_lattice[i][(j+1)%size[1]][k]],[mc_lattice[(i+1)%size[0]][j][k]],[mc_lattice[(i+2)%size[0]][j-1][k]],[mc_lattice[(i+3)%size[0]][j-2][k]]],[[mc_lattice[i-1][(j+1)%size[1]][k]],[hop_start_point],[hop_end_point],[mc_lattice[(i+2)%size[0]][j-2][k]]],[[mc_lattice[i-2][(j+1)%size[1]][k]],[mc_lattice[i-1][j][k]],[mc_lattice[i][j-1][k]],[mc_lattice[(i+1)%size[0]][j-2][k]]],[[mc_lattice[i-3][(j+1)%size[1]][k]],[mc_lattice[i-2][j][k]],[mc_lattice[i-1][j-1][k]],[mc_lattice[i][j-2][k]]]]) #good
                            elif endpoint_occupancy[0]==atom_types[1][1]:
                                local_lattice=np.array([[[mc_lattice[i-1][j][k]],[mc_lattice[i][j-1][k]],[mc_lattice[(i+1)%size[0]][j-2][k]],[mc_lattice[(i+2)%size[0]][j-3][k]]],[[mc_lattice[i-1][(j+1)%size[1]][k]],[hop_start_point],[hop_end_point],[mc_lattice[(i+2)%size[0]][j-2][k]]],[[mc_lattice[i-1][(j+2)%size[1]][k]],[mc_lattice[i][(j+1)%size[1]][k]],[mc_lattice[(i+1)%size[0]][j][k]],[mc_lattice[(i+2)%size[0]][j-1][k]]],[[mc_lattice[i-1][(j+3)%size[1]][k]],[mc_lattice[i][(j+2)%size[1]][k]],[mc_lattice[(i+1)%size[0]][(j+1)%size[1]][k]],[mc_lattice[(i+2)%size[0]][j][k]]]])#good

                        hop_probability_by_site[endpoint_index]=probability_multiplier*get_hop_probability(np.transpose(local_lattice,(1,0,2)),prefactor,kb_t)#gets the themal probability of the hop in question
                        #tetrahedral_path_count+=1*probability_multiplier

                endpoint_index=endpoint_index+1
            #path_counter=np.append(path_counter,(dumbell_path_count,tetrahedral_path_count,vaccancy_count))
            hop_probability_matrix[i][j][k]=hop_probability_by_site[:]

    return hop_probability_matrix
    """
