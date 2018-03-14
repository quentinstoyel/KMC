"""
needs to be run with do-all in the master directory of the cluster expansion, then it will return a list of occupancies, based off of the dictionary, of each case
These lists are printed to std out, to be copy pasted into gedit, where
"""


import numpy as np
import os
import re
cwd = os.getcwd()  # gets the current working directory
filename = cwd + '/str.out'
energy_filename = cwd + '/energy'
energy_file = open(energy_filename, 'r')
energy = energy_file.read()

# the atom_types from the KMC file/cluster expansion
atom_types = ([["Li", "Vac"], [-1, 1]])
# thing that is getting pooped out at the end of this, filled with vacancies
output_list = np.zeros(12) + atom_types[1][1]
output_list[-2] = energy
# records the number of the directory for bookkeeping
output_list[-1] = int(re.sub('[^0-9]', '', cwd[-3:]))
data = open(filename, 'r')
data = data.readlines()
mydict = {"['0.333333', '0.916667', '0.166667', 'Li']": 1, "['0.583333', '0.916667', '0.166667', 'Li']": 2, "['0.833333', '0.916667', '0.166667', 'Li']": 3, "['0.083333', '0.666667', '0.166667', 'Li']": 4,
          "['0.083333', '0.416667', '0.166667', 'Li']": 5, "['0.083333', '0.166667', '0.166667', 'Li']": 6, "['0.833333', '0.666667', '0.166667', 'Li']": 7, "['0.583333', '0.416667', '0.166667', 'Li']": 8, "['0.333333', '0.166667', '0.166667', 'Li']": 9}
for lin in np.transpose(data):
    structure_list = []
    for value in lin.split():
        try:
            # 6 is number of sig figs, adds all the numbers to entry
            structure_list.append(round(float(value), 6))
        except ValueError:
            try:
                structure_list.append(str(value))  # adds atom names to entry
            except ValueError:
                pass
    for index in range(len(structure_list) - 1):
        # deals with the fact that maps like using negative lattice coordinates
        if structure_list[index] < 0:
            structure_list[index] = (
                format((structure_list[index] + 1.), '.6f'))  # rounds to 6 sigfigs
        elif structure_list[index] >= 0:
            structure_list[index] = (format(structure_list[index], '.6f'))

    if str(structure_list) in mydict:
        output_list[mydict[str(structure_list)]] = atom_types[1][0]
print output_list
