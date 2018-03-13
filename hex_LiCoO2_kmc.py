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
import re
from itertools import combinations
import pickle
import itertools  # to hide all my nested for loop shenannigans inside of functions

"""Start Timer"""
start_timer = time.time()

"""Initiallizing useful variables"""
atom_types = ([["Li", "Vac"], [-1, 1]]
              )  # what values are assigned to which atoms in the matrix
kb_t = 0.025  # kb=~8.6*10^-5, t~300
prefactor = 10**(13)  # entropic factor from VderV paper
lattice_constant = 2.8334 * 10**(-8)  # in cm
hop_histogram = np.zeros(6)


def get_LCE_energy(input_lattice):
    """
    defunct!
    input_lattice values MUST BE FLOATS!
    input_lattice must be a 4x4x1, function takes input_lattice, uses the global cluster expansion to return the configurational energy of the state
    """
    lattice = np.array(input_lattice)

    if lattice[2][1][0] != atom_types[1][1]:
        sys.exit("Error: endpoint not empty!")
    if lattice[1][1][0] != atom_types[1][0]:
        sys.exit("Error: no ion to move!")
    if lattice[2][2][0] != atom_types[1][1]:
        sys.exit("Error: nearest neighbour not empty!")
    site = np.array([0, lattice[0][0][0], lattice[1][0][0], lattice[2][0][0], lattice[3][1][0], lattice[3][2][0], lattice[3][3][0]], lattice[0][1][0], lattice[1][2][0],
                    lattice[2][3][0])  # defines the sites 0-9, as indicated in notebook

    lattice = np.array(
        input_lattice)  # must have same config as sites lattice

    eci = [-0.486960, -0.002638, -0.004162, -0.003692, 0.002122, 0.010200, 0.039383, 0.039386, 0.038639, 0.038494, 0.038629,
           0.038665, 0.038475, 0.001066, 0.001081, 0.000977, 0.001076, 0.001066, 0.001066, 0.001077, 0.000975, 0.000978, 0.000440,
           0.000464, 0.000000, 0.000467, 0.000440, 0.000000, 0.000000, 0.000000, -
           0.000082, -0.000004, -0.000002, -0.000087,
           -0.000004, -0.000038, -0.000006, -0.000085, -0.000082, -
           0.000082, -0.000088, -0.000037, -0.000039, -0.000012, -0.000013,
           -0.000013, -0.000012, -0.000014, -0.000011, -0.000003, -
           0.000007, -0.000005, 0.000000, -0.000011, -0.000012, -0.000010,
           0.000006, -0.000006, 0.000004]
    multiplicities = np.array([1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2,
                               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2])
    energy_no_li = -76398.702259132575970
    Li_unit_energy = -12.856886553  # energy no Li-energy all Li dividede by 16

    # product is the the expectation values of the product of every site in a cluster
    product = np.zeros(len(eci))
    product[0] = eci[0]  # by definition
    product[1] = eci[1] * (site[6] + site[9])
    product[2] = eci[2] * (site[1] + site[3])
    product[3] = eci[3] * (site[4] + site[7])
    product[4] = eci[4] * (site[2])
    product[5] = eci[5] * (site[5] + site[8])
    product[6] = eci[6] * (site[3] * site[6] + site[1] * site[9])
    product[7] = eci[7] * (site[4] * site[7])
    product[8] = eci[8] * (site[5] * site[4] + site[8] * site[7])
    product[9] = eci[9] * (site[4] * site[1] + site[7] * site[3])
    product[10] = eci[10] * (site[5] * site[6] + site[8] * site[9])
    product[11] = eci[11] * (site[2] * site[3] + site[2] * site[1])
    product[12] = eci[12] * (site[1] * site[3])
    product[13] = eci[13] * (site[6] * site[7] + site[9] * site[4])
    product[14] = eci[14] * (site[1] * site[8] + site[3] * site[5])
    product[15] = eci[15] * (site[1] * site[5] + site[8] * site[3])
    product[16] = eci[16] * (site[5] * site[7] + site[8] * site[1])
    product[17] = eci[17] * (site[4] * site[3] + site[7] * site[1])
    product[18] = eci[18] * (site[9] * site[3] + site[6] * site[1])
    product[19] = eci[19] * (site[9] * site[1] + site[6] * site[3])
    product[20] = eci[20] * (site[5] * site[2] + site[2] * site[8])
    product[21] = eci[21] * (site[5] * site[9] + site[8] * site[6])
    product[22] = eci[22] * (site[2] * site[7] + site[2] * site[4])

    # contribution from having n li, ev
    energy_from_li = np.count_nonzero(
        site == atom_types[1][0]) * Li_unit_energy
    # tiny, energy per cite (order meVs), contribution from configuration, in ev
    energy_from_CE = sum(product)
    # print "CE energy: " +str(energy_from_CE)
    lattice_energy = energy_no_li + energy_from_li + energy_from_CE
    return lattice_energy  # total energy of 4x4 lattice (order keV) ~=90kev


def get_excited_energy(lattice):
    """
    actually returns barrier energy now
    lattice must be in the 16 character string form, hopper in 5, endpoint in 9, nn in 10
    lattice values MUST BE FLOATS!
    lattice must be a 4x4x1,rotated so the hop is going from 1,1 to 2,1 endpoint,
    with the 2,2 nearset neighbour endpoint empty
    uses the local cluster expansion, returns the excited state energy
    """

    # -25.5985 #magic number, yay!, from difference between 10 and 16 site cluster expansions, in ev.  take lowest concentration
    # states of each Ce and Lce, divide difference by 5 (4 constat atoms+hopper)
    Li_offsite_unit_energy = -913.3365
    atom_key_types = {'Li': '1', 'Vac': '0'}
    if lattice[9] != atom_key_types['Vac']:
        sys.exit("Error: endpoint not empty!")
    if lattice[5] != atom_key_types['Li']:
        sys.exit("Error: no ion to move!")
    if lattice[10] != atom_key_types['Vac']:
        sys.exit("Error: nearest neighbour not empty!")
    eci_old = [-89292.2076392, 12.0246248871, 0.5321464472, 0.6013549835, 0.6062609396, 0.5901284127, 0.5864268761,
               0.1392609219, 0.0912246106, 0.0125126163, 0.040332075]  # from LCE, energy contribution of each cluster
    eci = [0.5039840998, 0.0231779629, -0.0124155613, 0.0190676809, -0.0240978034, 0.0359364321, -0.0322688382, 0.0189131728, 0.0059693908, -0.0107495262, 0.0044709962, 0.0076906411, -0.0135407679, 0.0014011203, -0.0008762663, -0.0015711384, -0.0074600309, -0.0049897914, 0.0098191199, 0.0087838401, 0.0032917204, 0.0075716889, 0.0020836078, -
           0.0043424798, -0.0107164342, -0.0079384991, -0.0047555242, 0.0061737865, 0.0100261837, 0.0016684245, 0.0040444273, 0.007575093, -0.0146307688, -0.0101701531, 0.0011528037, -0.0028328131, -0.0020787791, 0.0038948159, 0.0131163879, 0.0048079312, 0.0016672859, 0.0047770905, 0.0114784225, 0.0028142975, -0.002387814, -0.0109707522, -0.0032730688]
    # hard coded, depends on which clusters are chosen in the LCE,
    multiplicities = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    lattice = lattice.replace('0', '3')
    sites = np.array([0, lattice[0], lattice[4], lattice[8], lattice[1], lattice[6], lattice[11], lattice[13],
                      lattice[14], lattice[15]]).astype(float) - 2  # defines the sites 0-9, as indicated in notebook
    number_of_sites = len(sites) - 1
    number_of_clusters = len(eci)
    # the product bit of the cluster expansion, eci*expted occupancy
    product = np.zeros(number_of_clusters)
    # Li sites, not included in the cluster expansion
    li_sites_not_in_lce = list(
        [lattice[3], lattice[2], lattice[7], lattice[12]])
    # corrects for the 4 Li sites that are always full in the LCE, in ev, counts which ones are empty, subtracts the offset energy
    correction_for_sites_not_in_lce = (li_sites_not_in_lce.count(
        atom_types[1][1])) * (-Li_offsite_unit_energy)
    # clusters:
    # energy contribution from clusters, dot product with eci is done on the way:
    product[0] = eci[0]
    product[1] = eci[1] * sum(sites[1:])
    product[2] = eci[2] * (sites[1])
    product[3] = eci[3] * (sites[3])
    product[4] = eci[4] * (sites[4])
    product[5] = eci[5] * (sites[7])
    product[6] = eci[6] * (sites[5])
    product[7] = eci[7] * (sites[8])
    product[8] = eci[8] * (sites[6])
    product[9] = eci[9] * (sites[9])
    product[10] = eci[10] * (sites[2])
    product[11] = eci[11] * (sites[6] * sites[9])
    product[12] = eci[12] * (sites[4] * sites[1])
    product[13] = eci[13] * (sites[3] * sites[7])
    product[14] = eci[14] * (sites[4] * sites[5] *
                             sites[6])
    product[15] = eci[15] * (sites[7] * sites[8] * sites[9])
    product[16] = eci[16] * (sites[1] * sites[2] * sites[3])
    product[17] = eci[17] * (sites[5] * sites[2] * sites[8])
    product[18] = eci[18] * (sites[5] * sites[9])
    product[19] = eci[19] * (sites[6] * sites[8])
    product[20] = eci[20] * (sites[5] * sites[1])
    product[21] = eci[21] * (sites[3] * sites[8])
    product[22] = eci[22] * (sites[2] * sites[4])
    product[23] = eci[23] * (sites[2] * sites[7])
    product[24] = eci[24] * (sites[5] * sites[6])
    product[25] = eci[25] * (sites[8] * sites[9])
    product[26] = eci[26] * (sites[5] * sites[4])
    product[27] = eci[27] * (sites[7] * sites[8])
    product[28] = eci[28] * (sites[1] * sites[2])
    product[29] = eci[29] * (sites[2] * sites[3])
    product[30] = eci[30] * (sites[2] * sites[1] *
                             sites[4])
    product[31] = eci[31] * (sites[2] * sites[3] * sites[7])
    product[32] = eci[32] * (sites[5] * sites[4] *
                             sites[1])
    product[33] = eci[33] * (sites[8] * sites[7] * sites[3])
    product[34] = eci[34] * (sites[5] * sites[6] *
                             sites[9])
    product[35] = eci[35] * (sites[8] * sites[9] * sites[6])
    product[36] = eci[36] * (sites[1] * sites[3])
    product[37] = eci[37] * (sites[4] * sites[6])
    product[38] = eci[38] * (sites[7] * sites[9])
    product[39] = eci[39] * (sites[1] * sites[4] * sites[5] * sites[6])
    product[40] = eci[40] * (sites[3] * sites[7] * sites[8] * sites[9])
    product[41] = eci[41] * (sites[5] * sites[1] *
                             sites[2] * sites[3] * sites[6])
    product[42] = eci[42] * (sites[2] * sites[8])
    product[43] = eci[43] * (sites[5] * sites[8])
    product[44] = eci[44] * (sites[5] * sites[4] * sites[1] * sites[2])
    product[45] = eci[45] * (sites[5] * sites[6] * sites[9] * sites[8])
    product[46] = eci[46] * (sites[2] * sites[3] * sites[7] * sites[8])
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

    eci = [-1.908877, 0.001238, 0.038481, 0.001010, 0.000222, 0.000000, -0.000421, -0.000054, -0.000001, 0.000012, -0.000007, 0.000000, -
           0.000001, 0.000000, 0.000000, -0.000000, 0.000000, 0.000000, 0.000029, 0.000001, 0.000004, 0.000001, 0.000001, 0.000003, -0.000000]
    multiplicities = np.array([1, 16, 48, 48, 48, 96, 32, 96, 32, 48,
                               192, 32, 192, 192, 96, 192, 96, 64, 48, 32, 192, 96, 48, 96, 96])
    energy_no_li = -76332.869735545391364
    Li_unit_energy = -12.934983385  # energy no Li-energy all Li dividede by 16

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

    # contribution from having n li, ev
    energy_from_li = np.count_nonzero(
        lattice == atom_types[1][0]) * Li_unit_energy
    # tiny, energy per cite (order meVs), contribution from configuration, in ev
    energy_from_CE = (np.dot(multiplicities * product, eci))
    # print "CE energy: " +str(energy_from_CE)
    lattice_energy = energy_no_li + energy_from_li + energy_from_CE
    return lattice_energy  # total energy of 4x4 lattice (order keV) ~=90kev


def lattice_convert(lattice):
    """function that takes a lattice and converts it into a 16 digit int """

    lattice = np.reshape(lattice, 16).astype(int).astype(str)
    s = "".join(lattice)
    return int(s)


def bad_LCE_energy(input_lattice):
    """
    input_lattice values MUST BE FLOATS!
    input_lattice must be a 4x4x1, function takes input_lattice, uses the global cluster expansion to return the configurational energy of the state
    """
    lattice = np.array(input_lattice)

    if lattice[2][1][0] != atom_types[1][1]:
        sys.exit("Error: endpoint not empty!")
    if lattice[1][1][0] != atom_types[1][0]:
        sys.exit("Error: no ion to move!")
    if lattice[2][2][0] != atom_types[1][1]:
        sys.exit("Error: nearest neighbour not empty!")
    site = np.array([0, lattice[0][0][0], lattice[1][0][0], lattice[2][0][0], lattice[0][1][0], lattice[1][2][0],
                     lattice[2][3][0], lattice[3][1][0], lattice[3][2][0], lattice[3][3][0]])  # defines the sites 0-9, as indicated in notebook

    lattice = np.array(
        input_lattice)  # must have same config as sites lattice

    eci = [-0.486960, -0.002638, -0.004162, -0.003692, 0.002122, 0.010200, 0.039383, 0.039386, 0.038639, 0.038494, 0.038629,
           0.038665, 0.038475, 0.001066, 0.001081, 0.000977, 0.001076, 0.001066, 0.001066, 0.001077, 0.000975, 0.000978, 0.000440,
           0.000464, 0.000000, 0.000467, 0.000440, 0.000000, 0.000000, 0.000000, -
           0.000082, -0.000004, -0.000002, -0.000087,
           -0.000004, -0.000038, -0.000006, -0.000085, -0.000082, -
           0.000082, -0.000088, -0.000037, -0.000039, -0.000012, -0.000013,
           -0.000013, -0.000012, -0.000014, -0.000011, -0.000003, -
           0.000007, -0.000005, 0.000000, -0.000011, -0.000012, -0.000010,
           0.000006, -0.000006, 0.000004]
    multiplicities = np.array([1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2,
                               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2])
    energy_no_li = -76398.702259132575970
    Li_unit_energy = -12.856886553  # energy no Li-energy all Li dividede by 16

    # product is the the expectation values of the product of every site in a cluster
    product = np.zeros(len(eci))
    product[0] = eci[0]  # by definition
    product[1] = eci[1] * (site[6] + site[9])
    product[2] = eci[2] * (site[1] + site[3])
    product[3] = eci[3] * (site[4] + site[7])
    product[4] = eci[4] * (site[2])
    product[5] = eci[5] * (site[5] + site[8])
    product[6] = eci[6] * (site[3] * site[6] + site[1] * site[9])
    product[7] = eci[7] * (site[4] * site[7])
    product[8] = eci[8] * (site[5] * site[4] + site[8] * site[7])
    product[9] = eci[9] * (site[4] * site[1] + site[7] * site[3])
    product[10] = eci[10] * (site[5] * site[6] + site[8] * site[9])
    product[11] = eci[11] * (site[2] * site[3] + site[2] * site[1])
    product[12] = eci[12] * (site[1] * site[3])
    product[13] = eci[13] * (site[6] * site[7] + site[9] * site[4])
    product[14] = eci[14] * (site[1] * site[8] + site[3] * site[5])
    product[15] = eci[15] * (site[1] * site[5] + site[8] * site[3])
    product[16] = eci[16] * (site[5] * site[7] + site[8] * site[1])
    product[17] = eci[17] * (site[4] * site[3] + site[7] * site[1])
    product[18] = eci[18] * (site[9] * site[3] + site[6] * site[1])
    product[19] = eci[19] * (site[9] * site[1] + site[6] * site[3])
    product[20] = eci[20] * (site[5] * site[2] + site[2] * site[8])
    product[21] = eci[21] * (site[5] * site[9] + site[8] * site[6])
    product[22] = eci[22] * (site[2] * site[7] + site[2] * site[4])

    # contribution from having n li, ev
    energy_from_li = np.count_nonzero(
        site == atom_types[1][0]) * Li_unit_energy
    # tiny, energy per cite (order meVs), contribution from configuration, in ev
    energy_from_CE = sum(product)
    # print "CE energy: " +str(energy_from_CE)
    lattice_energy = energy_no_li + energy_from_li + energy_from_CE
    return lattice_energy  # total energy of 4x4 lattice (order keV) ~=90kev


def dictionary_builder():  # creates a dictionary that contains the hop probailities for all of the lattice combinations, must be rerun for each temperature
    master_dictionary = {}
    start_point_value = 1
    constant_sites = 1
    empty_site = 0

    for configuration_number in xrange(2**(16), 2**(16) + 2**(16)):
        configuration_key = bin(configuration_number)[3:]
        configuration_array = [[[configuration_key[0]], [configuration_key[1]], [constant_sites], [constant_sites]], [[configuration_key[4]], [start_point_value], [configuration_key[6]], [
            constant_sites]], [[configuration_key[8]], [empty_site], [empty_site], [configuration_key[11]]], [[constant_sites], [configuration_key[13]], [configuration_key[14]], [configuration_key[15]]]]

        dict_key = "".join(np.reshape((configuration_array), 16))
        configuration_array = np.array(configuration_array).astype(float)
        configuration_array[configuration_array == 1] = -1
        configuration_array[configuration_array == 0] = 1
        print dict_key
        master_dictionary[dict_key] = get_hop_probability(dict_key,
                                                          configuration_array, prefactor, kb_t)  # write the probailities to the dicitonary

    pickle.dump(master_dictionary, open("normcons_dictionary.p", "wb+"),
                pickle.HIGHEST_PROTOCOL)  # saves dictionary to pickle
    print "foo"


def get_hop_probability(dict_key, local_lattice, prefactor, kb_t):
    """
    given an input lattice with a hoping ion on from the 1,1 site, to the 2,1 site,
    this function returns the probability of the hop
        """
    use_dictionary_mode = False  # toggles dicitonary option
    if use_dictionary_mode == False:
        drift_potential = 0  # in V, to apply an external field
        # dot products the potential
        drift_directions = np.array(
            [1, 0.5, -0.5, -1, -0.5, 0.5]) * drift_potential
        # calculates excited and groundstate energies to get barrier

        final_lattice = np.array(local_lattice)
        final_lattice[1][1][0] = atom_types[1][1]
        final_lattice[2][1][0] = atom_types[1][0]

        hop_energy = get_excited_energy(
            dict_key)  # - ((get_config_energy(local_lattice) + get_config_energy(final_lattice)) / 2)

        """
        Background fitting correction.
       

        hop_energy = hop_energy + \
            (0.05 + (np.count_nonzero(local_lattice ==
                                      atom_types[1][1]) - 2) * 0.02)
        if np.count_nonzero(local_lattice == atom_types[1][1]) == 11:
            hop_energy = hop_energy - 0.1

        if hop_energy < 0.3:
            hop_energy = hop_energy = 0.3
        """
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


def get_hex_probability(lattice_point):
    """
    function that gets hop probabilities for all endpoints, given an atom's location in the mc_lattice
    """
    i, j, k = lattice_point[0], lattice_point[1], lattice_point[2]
    hex_dict_key = np.array([mc_lattice[i][(j + 2) % mc_lattice_size[1]][k], mc_lattice[i - 1][(j + 2) % mc_lattice_size[1]][k], mc_lattice[i - 2][(j + 2) % mc_lattice_size[1]][k], mc_lattice[(i + 1) % mc_lattice_size[0]][(j + 1) % mc_lattice_size[1]][k], mc_lattice[i][(j + 1) % mc_lattice_size[1]][k], mc_lattice[i - 1][(j + 1) % mc_lattice_size[1]][k], mc_lattice[i - 2][(j + 1) % mc_lattice_size[1]][k], mc_lattice[(i + 2) %
                                                                                                                                                                                                                                                                                                                                                                                                                                   mc_lattice_size[0]][j][k], mc_lattice[(i + 1) % mc_lattice_size[0]][j][k], mc_lattice[i - 1][j][k], mc_lattice[i - 2][j][k], mc_lattice[(i + 2) % mc_lattice_size[0]][j - 1][k], mc_lattice[(i + 1) % mc_lattice_size[0]][j - 1][k], mc_lattice[i][j - 1][k], mc_lattice[i - 1][j - 1][k], mc_lattice[(i + 2) % mc_lattice_size[0]][j - 2][k], mc_lattice[(i + 1) % mc_lattice_size[0]][j - 2][k], mc_lattice[i][j - 2][k]]) + 2

    hex_dict_key = ''.join(
        map(str, (list(hex_dict_key.astype(int))))).replace("3", "0")
    return hex_dict[hex_dict_key]
    # return [1,1,1,1,1,1]


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
        max_i = max([startpoint[0], endpoint[0]]) + 2
        max_j = max([startpoint[1], endpoint[1]]) + 2
        min_i = min([startpoint[0], endpoint[0]]) - 2
        min_j = min([startpoint[1], endpoint[1]]) - 2
        max_k = 1
        min_k = 0  # 2d case

    for i, j, k in itertools.product(range(min_i, max_i), range(min_j, max_j), range(min_k, max_k)):

        i = i % mc_lattice_size[0]  # periodic boundary conditions
        j = j % mc_lattice_size[1]

        hop_start_point = int(mc_lattice[i][j][k])

        # is there an ion on the site at the start of the cycle?
        if hop_start_point == int(atom_types[1][0]):
            hop_probability_matrix[i][j][k] = get_hex_probability([i, j, k])
        # if there is no atom to move,hop probabilities are zero
        elif hop_start_point == int(atom_types[1][1]):
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
        if hop_master_index != 0:
            chosen_barrier = - \
                math.log((normalized_hop_probabilities[hop_master_index] -
                          normalized_hop_probabilities[hop_master_index - 1]) / 10**(13)) * 0.025
        # mc_lattice indices of hop start point, both use size[1] because thats the only one getting counted accross
        hopping_ion_j = (hop_master_index // 6) % mc_lattice_size[1]
        hopping_ion_i = (hop_master_index // 6) // mc_lattice_size[1]

        hop_endpoints = np.array([[hopping_ion_i + 1, hopping_ion_j], [hopping_ion_i, hopping_ion_j + 1], [hopping_ion_i - 1, hopping_ion_j + 1], [hopping_ion_i - 1,
                                                                                                                                                   hopping_ion_j], [hopping_ion_i, hopping_ion_j - 1], [hopping_ion_i + 1, hopping_ion_j - 1]])  # all of the possible endpoint indices for the hop
        hop_endpoint_i = hop_endpoints[endpoint_index][0]
        hop_endpoint_j = hop_endpoints[endpoint_index][1]
        # moving vaccancy to initial ion location on the mc_lattice
        """
        print hop_endpoint_i
        print hop_endpoint_j
        print hopping_ion_i
        print hopping_ion_j
        print endpoint_index

        print
        """

        mc_lattice[hopping_ion_i][hopping_ion_j][k] = atom_types[1][1]
        mc_lattice[(hop_endpoint_i % mc_lattice_size[0])][hop_endpoint_j % mc_lattice_size[1]
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
                                            (hop_endpoint_i), (hop_endpoint_j), 0], use_initialize_mode)  # update the probability lattice after the hop
        hop_histogram[endpoint_index] += 1
    if use_presimulation_mode == False:

        # print hop_histogram
        return (mc_lattice, distance_lattice, time_step_per_kmcstep, hop_probability_matrix, hop_histogram, chosen_barrier)

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


dictionary_builder()
sys.exit()
"""
empty_lattice = np.transpose(np.array(
    [[[1, 1, 1, -1], [1, -1, 1, 1], [-1, 1, 1, 1], [-1, -1, 1, 1]]]) * 1.)
empty_lattice_final= np.transpose(np.array(
    [[[-1, -1, -1, -1], [1, 1, 1, 1], [-1, 1, -1, 1], [-1, -1, 1, 1]]]) * 1.)
full_lattice_up = np.transpose(np.array(
    [[[-1., -1., -1., -1.], [-1., -1., 1., -1.], [-1., -1., 1., -1.], [-1., -1., -1., -1.]]]) * 1)
play_lattice = np.transpose(np.array(
    [[[1., -1., 1., -1.], [1., -1., 1., -1.], [-1., -1., 1., 1.], [-1., -1., 1., -1.]]]) * 1)
x = get_config_energy(full_lattice_up)
y = get_excited_energy(full_lattice_up)
a = get_config_energy(empty_lattice)
z = get_excited_energy(empty_lattice)
b = get_config_energy(play_lattice)
c = get_excited_energy(play_lattice)

print'play'
print b
print c
print c-b

print 'empty'
print a
print z
print z - a

sys.exit()

hex_dict = pickle.load(open("hex_dict.p", "rb"))

play_hex_dict = "111" + "1001" + "1001" + "1001" + "111"
print hex_dict[play_hex_dict]
sys.exit()  # stops the code so you can test

# loads the dictionaries
# master_dictionary = pickle.load(open("master_dictionary.p", "rb"))
hex_dict = pickle.load(open("hex_dict.p", "rb"))

"""
# dictionary_builder()


diffusion_coefficient_vs_concentration = []
diffusion_coefficient_j_vs_concentration = []
# for averaging_iterations in [1]:#range(1, 20,2):
averaging_iterations = 0   # loop over this
mc_lattice_size = [0, 0, 0]
dimensions = [10]  # loop over this
simulation_iterations = 750  # how many steps
presimulation_iterations = 250  # how many pre simulation step
# np.array(range(10,90,2))/100. # looping over these too
concentrations = [0.2]
#[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
average_path_counts = []

total_time = 0

output_file = open('LiCoO2_kmc.out', 'w')  # the results file
output_file.write('Size:' + str(mc_lattice_size) + '\n kbT:' + str(kb_t) + '\n Simulation parameters: \n Presimulation cycles: ' +
                  str(presimulation_iterations) + '\n Simulation Cycles: ' + str(simulation_iterations) + '\n')
output_file.write("\n RESULTS \n")

# data file to be opened and ploted with pandas
output_data_file = open('LiCoO2_data.out', 'w')
output_data_file.write(
    'Concentration Dimension D_star D_j error_star error_j time\n')


offset_score = np.array([0])

# range(5,65,8): #to check size dependence of parameters
for dimension in dimensions:
    # initializing lattice sizes, other size dependent parameters
    [mc_lattice_size[0], mc_lattice_size[1],
        mc_lattice_size[2]] = [dimension, dimension, 1]

    n_sites = mc_lattice_size[0] * mc_lattice_size[1] * mc_lattice_size[2]
    total_barrier = []
    for Li_concentration in concentrations:
        print ("concentration is: " + str(Li_concentration))
        averageing_iteration = 0
        # where the diffusion_coefficients that will be averaged go, one list per concentration with as many values as averaging cycles
        diffusion_coefficient_averaging = []
        diffusion_coefficient_j_averaging = []  # same thing but for other coefficient

        while averageing_iteration < averaging_iterations:
            print "concentration, averaging step is: " + str(Li_concentration) + ", " + str(averageing_iteration)
            hop_probability_matrix = np.tile(
                np.zeros(6), (mc_lattice_size[0], mc_lattice_size[1], mc_lattice_size[2], 1))
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
                if averaging_step % 500 == 0:
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

                (mc_lattice, distance_lattice, time_step, hop_probability_matrix, hop_histogram, chosen_barrier) = kmc_step(
                    mc_lattice, distance_lattice, hop_probability_matrix, use_presimulation_mode)  # perform 1 KMC step
                if simulation % 450 == 0:  # show how far we've gone in the simulation
                    print "simulation step: " + str(simulation)
                total_barrier = np.append(total_barrier, chosen_barrier)
                # colorsquare(np.transpose(mc_lattice)[0], "lattice_pictures/" + str(simulation + 10) + ".png")
                total_time = total_time + time_step  # the total time taken
                # diffusion_cycle,distances_travelled_list,diffusion_j=get_diffusion_coefficient(distance_lattice,mc_lattice_size,total_time,lattice_constant,Li_concentration) #getting the diffusion Coefficient, and the distances travelled by each atom
                # r_squared_vs_time=np.append(r_squared_vs_time,np.average(distances_travelled_list))
                # x_axis_times=np.append(x_axis_times,total_time)
                simulation += 1

            offset_score = np.append(offset_score, [((hop_histogram[1] + hop_histogram[4]) / 2) / (
                (hop_histogram[0] + hop_histogram[2] + hop_histogram[3] + hop_histogram[5]) / 4)])
            print offset_score
            (diffusion_coefficient_cycle, tmp, diffusion_j_cycle) = get_diffusion_coefficient(distance_lattice, mc_lattice_size,
                                                                                              total_time, lattice_constant, Li_concentration)  # diffusion_coefficient_cycle is obtained for every averaging step
            diffusion_coefficient_averaging = np.append(
                diffusion_coefficient_averaging, diffusion_coefficient_cycle)
            diffusion_coefficient_j_averaging = np.append(
                diffusion_coefficient_j_averaging, diffusion_j_cycle)
            # plt.figure()
            # averaging_value=25
            # average_distances=np.mean(r_squared_vs_time.reshape(-1, averaging_value), axis=1)
            # distances_errors=np.std(r_squared_vs_time.reshape(-1, averaging_value), axis=1)
            # plt.errorbar(x_axis_times[::averaging_value],average_distances,distances_errors)
            # plt.plot(range(simulation_iterations)[::averaging_value],average_distances)
            # plt.show()
            # plt.pause(10000)
            averageing_iteration += 1
        """
        output_file.write("Li concentration: " + str(Li_concentration) +
                          ". Diffusion_coefficients at this concentration: " + str(diffusion_coefficient_averaging) + "\n")
        output_file.write("Li concentration: " + str(Li_concentration) +
                          ". Diffusion_coefficient J at this concentration: " + str(diffusion_coefficient_j_averaging) + "\n")
        final_diffusion_coefficient = [np.average(
            diffusion_coefficient_averaging), np.std(diffusion_coefficient_averaging)]
        final_diffusion_coefficient_j = [np.average(
            diffusion_coefficient_j_averaging), np.std(diffusion_coefficient_j_averaging)]

        output_data_file.write(str(Li_concentration) + ' ' + str(dimension) + ' ' + str(final_diffusion_coefficient[0]) + ' ' + str(
            final_diffusion_coefficient_j[0]) + ' ' + str(final_diffusion_coefficient[1]) + ' ' + str(final_diffusion_coefficient_j[1]) + ' ' + str(total_time) + '\n')

        diffusion_coefficient_vs_concentration = np.append(diffusion_coefficient_vs_concentration, [
                                                           final_diffusion_coefficient[0], Li_concentration, final_diffusion_coefficient[1]])  # need to put in masterloop variable here
        diffusion_coefficient_j_vs_concentration = np.append(diffusion_coefficient_j_vs_concentration, [
                                                             final_diffusion_coefficient_j[0], Li_concentration, final_diffusion_coefficient_j[1]])  # need to put in masterloop variable here
        print "average barrier = " + str(np.average(total_barrier))
        print "max, min barrir = " + str(max(total_barrier)) + ", " + str(min(total_barrier))
        """
print offset_score

"""Stop timer"""
stop_timer = time.time()

output_file.write("\n\n Total Time: " +
                  str(stop_timer - start_timer) + "\n \n")
output_file.write("\n\n\n END\n")

output_file.write(" d_star  variable error_d_star d_j error_d_j \n")

"""
output_matrix = np.append(np.reshape(diffusion_coefficient_vs_concentration, (-1, 3)), zip(
    diffusion_coefficient_j_vs_concentration[::3], diffusion_coefficient_j_vs_concentration[2::3]), -1)
print output_matrix
output_file.write(re.sub('[\[\]]', '', str(
    output_matrix).strip().replace("\n   ", "")))
"""

output_file.close()
output_data_file.close()


print "diffusion_coefficient_vs_concentration is: " + str(diffusion_coefficient_vs_concentration)
print "diffusion_coefficient_j_vs_concentration is: " + str(diffusion_coefficient_j_vs_concentration)

"""
# plt.figure()
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
"""
