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


atom_types = ([["Li", "Vac"], [-1, 1]]
              )  # what values are assigned to which atoms in the matrix


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


def get_excited_energy(lattice):
    """
    lattice values MUST BE FLOATS!
    lattice must be a 4x4x1,rotated so the hop is going from 1,1 to 2,1 endpoint,
    with the 2,2 nearset neighbour endpoint empty
    uses the local cluster expansion, returns the excited state energy
    """

    # -25.5985 #magic number, yay!, from difference between 10 and 16 site cluster expansions, in ev.  take lowest concentration
    # states of each Ce and Lce, divide difference by 5 (4 constat atoms+hopper)
    Li_offsite_unit_energy = -913.3365

    if lattice[2][1][0] != atom_types[1][1]:
        sys.exit("Error: endpoint not empty!")
    if lattice[1][1][0] != atom_types[1][0]:
        sys.exit("Error: no ion to move!")
    if lattice[2][2][0] != atom_types[1][1]:
        sys.exit("Error: nearest neighbour not empty!")
    eci_old = [-89292.2076392, 12.0246248871, 0.5321464472, 0.6013549835, 0.6062609396, 0.5901284127, 0.5864268761,
               0.1392609219, 0.0912246106, 0.0125126163, 0.040332075]  # from LCE, energy contribution of each cluster
    eci = [-76457.0274111, 6.4320750473, 0.0166744804, 0.0043730637, -0.0167306476, -0.0039148771, 0.003869138, -0.0147444973,
           -0.0256469175, -0.0097084785, 0.0310352457, 0.0387651223, 0.0477511846, 0.0464696592, -
           0.0102525071, 0.0062501442,
           -0.0023335628, 0.0104847218, 0.0002541559, 0.0039699977, -
           0.0255239485, -0.0065800581, 0.0104507847, -0.0006385793,
           0.0290048499, 0.0426818779, 0.0381105622, 0.0346312853, 0.0576990811, 0.0567661583, -
           0.0024826459, -0.0017543717,
           -0.0033382232, 0.0068727074, -0.0057059822, -
           0.0055276197, 0.0103420799, 0.006103058, -0.0042513118, 0.0154418504,
           0.0067255146, 0.0063637922, -0.0093762346, 0.0111120693, -0.0024042498, 0.0080821917, 0.0057507372]
    # hard coded, depends on which clusters are chosen in the LCE,
    multiplicities = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    sites = np.array([0, lattice[0][0][0], lattice[1][0][0], lattice[2][0][0], lattice[0][1][0], lattice[1][2][0],
                      lattice[2][3][0], lattice[3][1][0], lattice[3][2][0], lattice[3][3][0]])  # defines the sites 0-9, as indicated in notebook
    number_of_sites = len(sites) - 1
    number_of_clusters = len(eci)
    # the product bit of the cluster expansion, eci*expted occupancy
    product = np.zeros(number_of_clusters)
    # Li sites, not included in the cluster expansion
    li_sites_not_in_lce = list(
        [lattice[3][0][0], lattice[1][3][0], lattice[0][3][0], lattice[0][2][0]])
    # corrects for the 4 Li sites that are always full in the LCE, in ev, counts which ones are empty, subtracts the offset energy
    correction_for_sites_not_in_lce = (li_sites_not_in_lce.count(
        atom_types[1][1])) * (-Li_offsite_unit_energy)
    # clusters:
    # energy contribution from clusters, dot product with eci is done on the way:
    product[0] = eci[0]
    product[1] = eci[1] * sum(sites)
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


def get_LCE_energy(input_lattice):
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


empty_lattice = np.transpose(np.array(
    [[[1, 1, 1, -1], [1, -1, 1, 1], [-1, 1, 1, 1], [-1, -1, 1, 1]]]) * 1.)
empty_lattice_final = np.transpose(np.array(
    [[[-1, -1, -1, -1], [1, 1, 1, 1], [-1, 1, -1, 1], [-1, -1, 1, 1]]]) * 1.)
full_lattice_up = np.transpose(np.array(
    [[[-1., -1., -1., -1.], [-1., -1., 1., -1.], [-1., -1., 1., -1.], [-1., -1., -1., -1.]]]) * 1)
play_lattice = np.transpose(np.array(
    [[[-1., 1., 1., -1.], [1., -1., 1., -1.], [-1., -1., 1., -1.], [-1., -1., 1., -1.]]]) * 1)

b = get_config_energy(play_lattice)
c = get_LCE_energy(play_lattice)
d = get_excited_energy(play_lattice)

print'play'
print b
print c
print d
print c - b
print d - b
