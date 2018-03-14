
"""
code that creates the hex dict, a dictionary that given a lattice hexagon of radius 3 (in binary notation), returns all of the hop probabilities for the central atom.

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

# loads the dictionary
master_dictionary = pickle.load(open("normcons_dictionary.p", "rb"))
master_dictionary['3'] = 0
# loops through all combinations:
# 1==ocupied with li, 0==vac, need 3==vac, 1==Li
atom_types = 2  # li and vac
atoms_in_configuration = 18  # number of sites in your groovy double hexagon
kb_t = 0.025  # kb=~8.6*10^-5, t~300
prefactor = 10**(13)  # entropic factor from VderV paper
# in ev, constant, hop energy when the two sites on each side are occupied
dumbell_hop_probability = prefactor * \
    math.exp(-(0.900) / kb_t)  # the stat mech bit#from NEB
constant_sites = '1'
start_point_value = '1'
endpoint_value = '0'
empty_site = '0'
hex_dict = {}
total_index_sum = 0
# atom_types**atoms_in_configuration): #counts to 2^n.
for configuration_number in xrange(2**(atoms_in_configuration), 2**(atoms_in_configuration) + atom_types**(atoms_in_configuration)):
    configuration_key = bin(configuration_number)[3:]
    configuration_probabilities = np.zeros(6)
    endpoint_labels = [8, 4, 5, 9, 13, 12]

    # endpoint 1, spot 8
    # gets the appropriately oriented local lattice to hand to normcons_dict
    for endpoint in range(len(endpoint_labels)):
        master_dictionary_key1 = '3'
        master_dictionary_key2 = '3'
        # if endpoint is empty
        if configuration_key[endpoint_labels[endpoint]] == '0':

            # if both NN's are full, then its a dumbell hop
            if configuration_key[endpoint_labels[endpoint - 1]] == '1' and configuration_key[endpoint_labels[(endpoint + 1) % len(endpoint_labels)]] == '1':
                configuration_probabilities[endpoint] = dumbell_hop_probability
            else:

                if endpoint_labels[endpoint] == 8:
                    if configuration_key[endpoint_labels[endpoint - 1]] == '0':
                        master_dictionary_key1 = configuration_key[5] + configuration_key[9] + constant_sites + constant_sites + configuration_key[4] + start_point_value + configuration_key[13] + constant_sites + \
                            configuration_key[3] + endpoint_value + empty_site + configuration_key[16] + \
                            constant_sites + \
                            configuration_key[7] + \
                            configuration_key[11] + configuration_key[15]

                    if configuration_key[endpoint_labels[(endpoint + 1) % len(endpoint_labels)]] == '0':
                        master_dictionary_key2 = configuration_key[13] + configuration_key[9] + constant_sites + constant_sites + configuration_key[12] + start_point_value + configuration_key[5] + \
                            constant_sites + configuration_key[11] + endpoint_value + empty_site + configuration_key[1] + \
                            constant_sites + \
                            configuration_key[7] + \
                            configuration_key[3] + configuration_key[0]
                elif endpoint_labels[endpoint] == 4:
                    if configuration_key[endpoint_labels[endpoint - 1]] == '0':
                        master_dictionary_key1 = configuration_key[9] + configuration_key[13] + constant_sites + constant_sites + configuration_key[5] + start_point_value + configuration_key[12] + \
                            constant_sites + configuration_key[1] + endpoint_value + empty_site + configuration_key[11] + \
                            constant_sites + \
                            configuration_key[0] + \
                            configuration_key[3] + configuration_key[7]
                    if configuration_key[endpoint_labels[(endpoint + 1) % len(endpoint_labels)]] == '0':
                        master_dictionary_key2 = configuration_key[12] + configuration_key[13] + constant_sites + constant_sites + configuration_key[8] + start_point_value + configuration_key[9] + \
                            constant_sites + configuration_key[3] + endpoint_value + empty_site + configuration_key[6] + \
                            constant_sites + \
                            configuration_key[0] + \
                            configuration_key[1] + configuration_key[2]
                elif endpoint_labels[endpoint] == 5:
                    if configuration_key[endpoint_labels[endpoint - 1]] == '0':
                        master_dictionary_key1 = configuration_key[13] + configuration_key[12] + constant_sites + constant_sites + configuration_key[9] + start_point_value + configuration_key[8] + \
                            constant_sites + configuration_key[6] + endpoint_value + empty_site + configuration_key[3] + \
                            constant_sites + \
                            configuration_key[2] + \
                            configuration_key[1] + configuration_key[0]
                    if configuration_key[endpoint_labels[(endpoint + 1) % len(endpoint_labels)]] == '0':
                        master_dictionary_key2 = configuration_key[8] + configuration_key[12] + constant_sites + constant_sites + configuration_key[4] + start_point_value + configuration_key[13] + \
                            constant_sites + configuration_key[1] + endpoint_value + empty_site + configuration_key[14] + \
                            constant_sites + \
                            configuration_key[2] + \
                            configuration_key[6] + configuration_key[10]
                elif endpoint_labels[endpoint] == 9:
                    if configuration_key[endpoint_labels[endpoint - 1]] == '0':
                        master_dictionary_key1 = configuration_key[12] + configuration_key[8] + constant_sites + constant_sites + configuration_key[13] + start_point_value + configuration_key[4] + \
                            constant_sites + configuration_key[14] + endpoint_value + empty_site + configuration_key[1] + \
                            constant_sites + \
                            configuration_key[10] + \
                            configuration_key[6] + configuration_key[2]
                    if configuration_key[endpoint_labels[(endpoint + 1) % len(endpoint_labels)]] == '0':
                        master_dictionary_key2 = configuration_key[4] + configuration_key[8] + constant_sites + constant_sites + configuration_key[5] + start_point_value + configuration_key[12] + constant_sites + \
                            configuration_key[6] + endpoint_value + empty_site + configuration_key[16] + \
                            constant_sites + \
                            configuration_key[10] + \
                            configuration_key[14] + configuration_key[17]
                elif endpoint_labels[endpoint] == 13:
                    if configuration_key[endpoint_labels[endpoint - 1]] == '0':
                        master_dictionary_key1 = configuration_key[8] + configuration_key[4] + constant_sites + constant_sites + configuration_key[12] + start_point_value + configuration_key[5] + constant_sites + \
                            configuration_key[16] + endpoint_value + empty_site + configuration_key[6] + \
                            constant_sites + \
                            configuration_key[17] + \
                            configuration_key[14] + configuration_key[10]
                    if configuration_key[endpoint_labels[(endpoint + 1) % len(endpoint_labels)]] == '0':
                        master_dictionary_key2 = configuration_key[5] + configuration_key[4] + constant_sites + constant_sites + configuration_key[9] + start_point_value + configuration_key[8] + constant_sites + \
                            configuration_key[14] + endpoint_value + empty_site + configuration_key[11] + \
                            constant_sites + \
                            configuration_key[17] + \
                            configuration_key[16] + configuration_key[15]
                elif endpoint_labels[endpoint] == 12:
                    if configuration_key[endpoint_labels[endpoint - 1]] == '0':
                        master_dictionary_key1 = configuration_key[4] + configuration_key[5] + constant_sites + constant_sites + configuration_key[8] + start_point_value + configuration_key[9] + constant_sites + \
                            configuration_key[11] + endpoint_value + empty_site + configuration_key[14] + \
                            constant_sites + \
                            configuration_key[15] + \
                            configuration_key[16] + configuration_key[17]
                    if configuration_key[endpoint_labels[(endpoint + 1) % len(endpoint_labels)]] == '0':
                        master_dictionary_key2 = configuration_key[9] + configuration_key[5] + constant_sites + constant_sites + configuration_key[13] + start_point_value + configuration_key[4] + constant_sites + \
                            configuration_key[16] + endpoint_value + empty_site + configuration_key[3] + \
                            constant_sites + \
                            configuration_key[15] + \
                            configuration_key[11] + configuration_key[7]

                configuration_probabilities[endpoint] = master_dictionary[
                    master_dictionary_key1] + master_dictionary[master_dictionary_key2]
        else:
            configuration_probabilities[endpoint] = 0

    # assigns probabilities to the hex dictionary under the binary key
    hex_dict[configuration_key] = configuration_probabilities

total_sum = sum(hex_dict.values())
print np.log(total_sum)

pickle.dump(hex_dict, open("hex_dict.p", "wb+"),
            pickle.HIGHEST_PROTOCOL)  # saves dictionary to pickle
