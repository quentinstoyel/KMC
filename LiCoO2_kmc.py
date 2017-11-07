# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:36:34 2017

@author: qstoyel
First stab at KMC code:
"""


import numpy as np
import matplotlib.pyplot as plt
import random
import time
import sys

"""Initiallizing useful variables"""
atom_types=([["Li","Vac"],[-1,1]]) #what values are assigned to which atoms in the matrix
kb_t=0.025 #kb=~8.6*10^-5
prefactor=10**(13)#entropic factor from VderV paper
dumbell_hop_energy=1.48 #in ev, constant, hop energy when the two sites on each side are occupied
lattice_constant = 2.8334 #in angstrom

"""Initiallizing LCE variables"""

hop_histogram=np.zeros(6)



def get_excited_energy(lattice):
    """
    lattice values MUST BE FLOATS!
    lattice must be a 4x4x1,rotated so the hop is going from 1,1 to 2,1 endpoint,
    uses the local cluster expansion, returns the excited state energy
    """

    Li_offsite_unit_energy=-25.832 #-25.5985 #magic number, yay!, from difference between 10 and 16 site cluster expansions
    Li_unit_energy=-25.05 #magic number, contribution of each of the "on stie" Lithiums

    if lattice[2][1][0]!=atom_types[1][1]:
        sys.exit("Error: endpoint not empty!")
    if lattice[1][1][0]!=atom_types[1][0]:
        sys.exit("Error: no ion to move!")
    eci=[-89292.2076392, 12.0246248871, 0.5321464472, 0.6013549835, 0.6062609396, 0.5901284127, 0.5864268761, 0.1392609219, 0.0912246106, 0.0125126163, 0.040332075 ] #from LCE
    multiplicities=np.array([1,10,2,2,2,2,1,1,2,2,1]) #hard coded, depends on which clusters are chosen in the LCE
    sites=np.array([lattice[2][2][0],lattice[0][0][0],lattice[1][0][0],lattice[2][0][0],lattice[0][1][0],lattice[1][2][0],lattice[2][3][0],lattice[3][1][0],lattice[3][2][0],lattice[3][3][0]]) #defines the sites 0-9, as indicated in notebook
    number_of_sites=len(sites)
    number_of_clusters=len(eci)
    product=np.zeros(number_of_clusters)
    fixed_li_sites=list([lattice[3][0][0],lattice[1][3][0],lattice[0][3][0],lattice[0][2][0]])
    Li_correction=(fixed_li_sites.count(atom_types[1][1]))*(-Li_offsite_unit_energy) #corrects for the 4 Li sites that are always full in the LCE.
    #clusters:
    #energy contribution from clusters, dot product is already done:
    product[0]=eci[0]
    product[1]=eci[1]*sum(sites)
    product[2]=eci[2]*(sites[1]+sites[3])
    product[3]=eci[3]*(sites[4]+sites[7])
    product[4]=eci[4]*(sites[5]+sites[8])
    product[5]=eci[5]*(sites[6]+sites[9])
    product[6]=eci[6]*(sites[2])
    product[7]=eci[7]*(sites[6]*sites[9])
    product[8]=eci[8]*(sites[4]*sites[1]+sites[3]*sites[7])
    product[9]=eci[9]*(sites[4]*sites[5]*sites[6]+sites[7]*sites[8]*sites[9])
    product[10]=eci[10]*(sites[1]*sites[2]*sites[3])
    excited_energy=sum(product)+Li_correction
    return excited_energy #returns total excited state energy, on order 90keV

def get_config_energy(lattice): #most computationally heavy bit of code that is run often
    """
    lattice values MUST BE FLOATS!
    lattice must be a 4x4x1 function takes input lattice, uses the global cluster expansion to obtain the configurational energy of the state
    """
    lattice=np.array(lattice)
    size=(4,4,1) #it had better be, otherwise the CE won't work...
    eci=[-2.333258,0.001058,0.045567,0.002186,0.000803,0.000000,-0.000329,-0.000050,-0.000001,0.000009,-0.000008,0.000001,-0.000003,0.000000,0.000000,-0.000000,0.000000,0.000000,0.000027,0.000001,0.000004,0.000001,0.000002,0.000003,0.000001]#eci, from maps
    multiplicities=np.array([1,16,48,48,48,96,32,96,32,48,192,32,192,192,96,192,96,64,48,32,192,96,48,96,96]) #multiplicities from maps
    #reference_energy=[-5590.22,-5564.88] #also from maps, ref.out
    Li_unit_energy=-25.337714498
    product=np.zeros(len(eci)) #the expectation values of every site in a cluster
    product[0]=1.0 #by definition
    product[1]=np.average(lattice) #also by definition
    k=0
    for i in range(size[0]):#defines the product with all of the expectation values of the occupancies
        for j in range(size[1]):
            #for k in range(size[2]):#always zero, no need to loop
            #pairs
            cluster_location=lattice[i][j][k] # for speedup, by minimizing memory reads into lattice
            nearest_neighbour1=lattice[(i+1)%size[0]][j][k]
            nearest_neighbour2=lattice[i][(j+1)%size[1]][k]
            nearest_neighbour3=lattice[(i+1)%size[0]][(j+1)%size[1]][k]
            neighbour1=lattice[(i+2)%size[0]][(j+1)%size[1]][k]
            neighbour2=lattice[(i+1)%size[0]][(j+2)%size[1]][k]
            product[2]=product[2]+(cluster_location*nearest_neighbour1+cluster_location*nearest_neighbour2+cluster_location*nearest_neighbour3)/multiplicities[2]
            product[3]=product[3]+(cluster_location*neighbour1+cluster_location*neighbour2+cluster_location*lattice[(i+1)%size[0]][(j+3)%size[1]][k])/multiplicities[3]
            product[4]=product[4]+(cluster_location*lattice[(i+2)%size[0]][j][k]+cluster_location*lattice[i][(j+2)%size[1]][k]+cluster_location*lattice[(i+2)%size[0]][(j+2)%size[1]][k])/multiplicities[4]
            product[5]=0 #eci=0, atoms are too far away
            #triplets:
            product[6]=product[6]+(cluster_location*nearest_neighbour1*nearest_neighbour3+cluster_location*nearest_neighbour2*nearest_neighbour3)/multiplicities[6]
            product[7]=product[7]+(cluster_location*nearest_neighbour1*neighbour1+cluster_location*nearest_neighbour3*neighbour1+cluster_location*nearest_neighbour3*neighbour2+cluster_location*nearest_neighbour2*neighbour2+cluster_location*nearest_neighbour2*lattice[(i+3)%size[0]][(j+1)%size[1]][k]+cluster_location*lattice[(i+3)%size[0]][j][k]*lattice[(i+3)%size[0]][(j+1)%size[1]][k])/multiplicities[7]
            product[8]=0 #product[8]+(cluster_location*neighbour1*neighbour2+cluster_location*lattice[(i+3)%size[0]][(j+1)%size[1]][k]*neighbour2)/multiplicities[8]
            product[9]=product[9]+(cluster_location*nearest_neighbour1*lattice[(i+2)%size[0]][j][k]+cluster_location*nearest_neighbour2*lattice[i][(j+2)%size[1]][k]+cluster_location*nearest_neighbour3*lattice[(i+2)%size[0]][(j+2)%size[1]][k])/multiplicities[9]
            #quartics
            product[18]=product[18]+(cluster_location*nearest_neighbour1*nearest_neighbour3*neighbour1+cluster_location*nearest_neighbour3*nearest_neighbour2*neighbour2+cluster_location*nearest_neighbour2*lattice[(i+3)%size[0]][j][k]*lattice[(i+3)%size[0]][(j+1)%size[1]][k])/multiplicities[18]
    energy_no_li=-89038.093349139890911 #the groundstate energy
    energy_from_li=np.count_nonzero(lattice==atom_types[1][0])*Li_unit_energy #contribution from li
    energy_from_CE=np.dot(multiplicities*product,eci) #tiny, energy per cite (order meVs), contribution from configuration
    #print "CE energy: " +str(energy_from_CE)
    cell_energy=energy_no_li+energy_from_li+energy_from_CE
    return cell_energy #total energy of 4x4 cell (order keV) ~=90kev

def get_hop_probability(local_lattice,prefactor,kb_t,endpoint_index):
    """input lattice has hopping ion on from 1,1, to 1,2, as defined by local_lattice
    returns the probability of the hop
        """
    drift_potential=0 #in V, to apply an external field
    drift_directions=np.array([1,0.5,-0.5,-1,-0.5,0.5])*drift_potential

    hop_energy=get_jump_energy(local_lattice)

    probability=prefactor*np.exp(-(hop_energy)/kb_t)

    return probability

def get_jump_energy(initial_lattice):
    """function that will calculate the hop energy, using Ehop=Eas-Ei.
    lattices must be in 4x4x1 format, with the hop going from 1,1 to 1,2, ie as defined by local_lattice above.
    """
    initial_energy=get_config_energy(initial_lattice)
    excited_energy=get_excited_energy(initial_lattice)
    hop_energy=(excited_energy-initial_energy)
    #print  "hop energy: " +str(hop_energy)
    return hop_energy

def get_distance(input_lattice,size,lattice_constant):
    """function that takes the input/distance_lattice of coordinates at the end of the monte carlo and returns an array with values of how far all of the ions have travelled
    """
    distances_travelled=np.zeros([size[0],size[1],size[2]])
    distances_out=[]
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size[2]):
                x=input_lattice[i][j][k]
                if x[0]>=0:
                    i_index=i+size[0]*round(float(x[2])/100) #adding to deal with atoms that have looped around
                    j_index=j+size[1]*(x[2]-round(float(x[2])/100)*100)
                    #distances_travelled[i][j][k]=((np.abs(x[0]-i_index)+np.abs(x[0]+x[1]-i_index-j_index)+np.abs(x[1]-j_index))/2)
                    distances_travelled[i][j][k]=np.sqrt(np.abs((x[0]-i_index)**2+(x[1]-j_index)**2+(x[0]-i_index)*(x[1]-j_index))) #hexagonal distance between two points, matrix form
                    distances_out=np.append(distances_out,distances_travelled[i][j][k]) #distance travelled per atom, in list form, no order to it though
    return distances_travelled*lattice_constant,distances_out*lattice_constant

def get_diffusion_coefficient(distance_lattice,size,time,lattice_constant,Li_concentration):
    number_of_atoms=size[0]*size[1]*size[2]*Li_concentration
    if number_of_atoms==0:
        sys.exit("concentration is zero, no atoms to move!")
    distances_travelled,distances_out=get_distance(distance_lattice,size,lattice_constant)

    diffusion_star=1/(2*2*time/number_of_atoms)*np.average(np.square(distances_out)) #need to divide time by number of atoms, 2*2 is for the fact that this thing is 2D
    return diffusion_star, distances_out

def colorsquare(s,filename):#to plot the lattices as you go
    """Figure animation stuff"""
    plt.ion()
    plt.imshow(s, cmap='gray', interpolation='none')
    plt.show()
    plt.savefig(filename)
    plt.pause(0.001)
    return

def kmc_evolve(mc_lattice,hop_probability_by_ion,startpoint,endpoint):
    """function that updates the hop_probability_by_ion after a hop has occured, updating only the effected probabilities, startpoint,endpoint are passed as lists: [1,j,k]"""
    max_i=max([startpoint[0],endpoint[0]])+3#which atoms are affected by a hop
    max_j=max([startpoint[1],endpoint[1]])+3
    min_i=min([startpoint[0],endpoint[0]])-2
    min_j=min([startpoint[1],endpoint[1]])-2
    k=0
    dumbell_hop_probability=prefactor*np.exp(-(dumbell_hop_energy)/kb_t)

    for i in range(min_i,max_i):
        for j in range(min_j,max_j):
            i=i%size[0]
            j=j%size[1]
            hop_start_point=float(mc_lattice[i][j][k])
            if int(mc_lattice[i][j][k])==int(atom_types[1][0]):#is there an ion on the site at the start of the cycle?
                endpoints=np.array([[i+1,j],[i,j+1],[i-1,j+1],[i-1,j],[i,j-1],[i+1,j-1]]) #directions ion can hop
                endpoint_occupancy=np.array([mc_lattice[endpoints[0][0]%size[0]][endpoints[0][1]%size[1]][0],mc_lattice[endpoints[1][0]%size[0]][endpoints[1][1]%size[1]][0],mc_lattice[endpoints[2][0]%size[0]][endpoints[2][1]%size[1]][0],mc_lattice[endpoints[3][0]%size[0]][endpoints[3][1]%size[1]][0],mc_lattice[endpoints[4][0]%size[0]][endpoints[4][1]%size[1]][0],mc_lattice[endpoints[5][0]%size[0]][endpoints[5][1]%size[1]][0]])
                hop_probability_by_site=np.zeros(6)
                endpoint_index=0
                for (endpoint_index, endpoint) in list(enumerate(endpoints)): #defines local lattice, based on endpoint (direction indepenance), creates hop_probability_by_site a vector with probability/per site of hops
                    probability_multiplier=1
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
                                    local_lattice=np.array([[[mc_lattice[i][j-1][k]],[mc_lattice[i-1][j][k]],[mc_lattice[i-2][(j+1)%size[1]][k]],[mc_lattice[i-3][(j+2)%size[1]][k]]],[[mc_lattice[(i+1)%size[0]][j-1][k]],[hop_start_point],[hop_end_point],[mc_lattice[i-2][(j+2)%size[1]][k]]],[[mc_lattice[(i+1)%size[0]][j][k]],[mc_lattice[i][(j+1)%size[1]][k]],[mc_lattice[i-1][(j+2)%size[1]][k]],[mc_lattice[(i+3)%size[0]][j-1][k]]],[[mc_lattice[(i+2)%size[0]][j][k]],[mc_lattice[(i+1)%size[0]][(j+1)%size[1]][k]],[mc_lattice[i][(j+2)%size[1]][k]],[mc_lattice[(i+2)%size[0]][(j+2)%size[1]][k]]]])#good
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
                        if hop_probability_by_site[endpoint_index]!=dumbell_hop_probability: #if its not a dumbell hop,
                            hop_probability_by_site[endpoint_index]=probability_multiplier*get_hop_probability(np.transpose(local_lattice,(1,0,2)),prefactor,kb_t,endpoint_index)#gets the themal probability of the hop in question
                            #tetrahedral_path_count+=1*probability_multiplier
                    endpoint_index=endpoint_index+1
                hop_probability_by_ion[i][j][k]=hop_probability_by_site[:]
            elif int(mc_lattice[i][j][k])==int(atom_types[1][1]):
                hop_probability_by_ion[i][j][k]=np.zeros(6)
    return hop_probability_by_ion

def kmc_step(mc_lattice,input_distance_lattice):
    """main monte carlo loop
    going over each lattice site and getting local lattices
    """
    distance_lattice=np.array(input_distance_lattice)
    #index_lattice=np.array(mc_lattice) #index lattice is the one to loop over does not evolve during a step, so as not to multiple hop the forward hopping ions
    time_step_per_hop=0
    time_step_per_kmcstep=0
    cycle_diffusion_coefficient=[]
    k=0
    dumbell_hop_probability=prefactor*np.exp(-(dumbell_hop_energy)/kb_t)
    hop_probability_by_ion=np.tile(np.zeros(6),(size[0],size[1],size[2],1))
    for i in range(size[0]):#looping over elements in the array to calculate all of the probabilities. write them into hop_probability_by_ion
        for j in range(size[1]):
            for k in range(size[2]):
                hop_start_point=float(mc_lattice[i][j][k])
                hop_probability_by_site=np.zeros(6)
                if int(mc_lattice[i][j][k])==int(atom_types[1][0]):#is there an ion on the site at the start of the cycle?
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
                                        local_lattice=np.array([[[mc_lattice[i][j-1][k]],[mc_lattice[i-1][j][k]],[mc_lattice[i-2][(j+1)%size[1]][k]],[mc_lattice[i-3][(j+2)%size[1]][k]]],[[mc_lattice[(i+1)%size[0]][j-1][k]],[hop_start_point],[hop_end_point],[mc_lattice[i-2][(j+2)%size[1]][k]]],[[mc_lattice[(i+1)%size[0]][j][k]],[mc_lattice[i][(j+1)%size[1]][k]],[mc_lattice[i-1][(j+2)%size[1]][k]],[mc_lattice[(i+3)%size[0]][j-1][k]]],[[mc_lattice[(i+2)%size[0]][j][k]],[mc_lattice[(i+1)%size[0]][(j+1)%size[1]][k]],[mc_lattice[i][(j+2)%size[1]][k]],[mc_lattice[(i+2)%size[0]][(j+2)%size[1]][k]]]])#good
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
                            if hop_probability_by_site[endpoint_index]!=dumbell_hop_probability: #if its not a dumbell hop,
                                hop_probability_by_site[endpoint_index]=probability_multiplier*get_hop_probability(np.transpose(local_lattice,(1,0,2)),prefactor,kb_t,endpoint_index)#gets the themal probability of the hop in question
                                #tetrahedral_path_count+=1*probability_multiplier

                        endpoint_index=endpoint_index+1
                    #path_counter=np.append(path_counter,(dumbell_path_count,tetrahedral_path_count,vaccancy_count))
                    hop_probability_by_ion[i][j][k]=hop_probability_by_site[:]
    gamma=np.sum(hop_probability_by_ion)#gamma as defined in VDVen paper
    normalized_hop_probabilities=np.cumsum(hop_probability_by_ion)/gamma
    rho=np.random.uniform(0,1) #the random number
    lattice_index=np.searchsorted(normalized_hop_probabilities, rho) #get master index of hop
    endpoint_index=lattice_index%6 #get which endpoint index of where the ion hopped to
    hopping_ion_i= (lattice_index//6)//size[0] #the i location of the ion about to hop
    hopping_ion_j= (lattice_index//6)%size[0] # the j location of the ion about to hop
    hop_endpoints=np.array([[hopping_ion_i+1,hopping_ion_j],[hopping_ion_i,hopping_ion_j+1],[hopping_ion_i-1,hopping_ion_j+1],[hopping_ion_i-1,hopping_ion_j],[hopping_ion_i,hopping_ion_j-1],[hopping_ion_i+1,hopping_ion_j-1]])
    hop_endpoint_i=hop_endpoints[endpoint_index][0]
    hop_endpoint_j=hop_endpoints[endpoint_index][1]


    mc_lattice[hopping_ion_i][hopping_ion_j][k]=atom_types[1][1] #moving vaccancy to initial ion location
    mc_lattice[hop_endpoint_i%size[0]][hop_endpoint_j%size[1]][0]=atom_types[1][0] #moving ion to appropriate endpoint
    distance_lattice[hop_endpoint_i%size[0]][hop_endpoint_j%size[1]][k]=distance_lattice[hopping_ion_i][hopping_ion_j][k] #moving the ion's coordinate to the appropriate site
    distance_lattice[hop_endpoint_i%size[0]][hop_endpoint_j%size[1]][k][2]=distance_lattice[hop_endpoint_i%size[0]][hop_endpoint_j%size[1]][k][2]+100*(hop_endpoint_i//size[0])+hop_endpoint_j//size[1] #adds index to distance lattice for looping purposes
    distance_lattice[hopping_ion_i][hopping_ion_j][k]=[-1,0,0]#clears the coordiates from the old site ->sticks in a vaccancy
    time_step_per_hop=(-1./gamma)*np.log(np.random.uniform(0,1))
    time_step_per_kmcstep=time_step_per_kmcstep+time_step_per_hop

    for hop in range(int(size[0]*size[1]*size[2]*Li_concentration)):
        hop_probability_by_ion=kmc_evolve(mc_lattice,hop_probability_by_ion, [hopping_ion_i,hopping_ion_j,0],[hop_endpoint_i,hop_endpoint_j,0])
        gamma=np.sum(hop_probability_by_ion)#gamma as defined in VDVen paper
        normalized_hop_probabilities=np.cumsum(hop_probability_by_ion)/gamma
        rho=np.random.uniform(0,1) #the random number
        lattice_index=np.searchsorted(normalized_hop_probabilities, rho) #get master index of hop
        #lattice_index = np.argmin(np.abs(normalized_hop_probabilities-rho))#get master index of hop
        endpoint_index=lattice_index%6 #get which endpoint the ion hopped to
        hopping_ion_i= (lattice_index//6)//size[0]
        hopping_ion_j= (lattice_index//6)%size[0]
        hop_endpoints=np.array([[hopping_ion_i+1,hopping_ion_j],[hopping_ion_i,hopping_ion_j+1],[hopping_ion_i-1,hopping_ion_j+1],[hopping_ion_i-1,hopping_ion_j],[hopping_ion_i,hopping_ion_j-1],[hopping_ion_i+1,hopping_ion_j-1]])
        hop_endpoint_i=hop_endpoints[endpoint_index][0]
        hop_endpoint_j=hop_endpoints[endpoint_index][1]
        mc_lattice[hopping_ion_i][hopping_ion_j][k]=atom_types[1][1] #moving vaccancy to initial ion location
        mc_lattice[hop_endpoint_i%size[0]][hop_endpoint_j%size[1]][0]=atom_types[1][0] #moving ion to appropriate endpoint
        distance_lattice[hop_endpoint_i%size[0]][hop_endpoint_j%size[1]][k]=distance_lattice[hopping_ion_i][hopping_ion_j][k] #moving the ion's coordinate to the appropriate site
        distance_lattice[hop_endpoint_i%size[0]][hop_endpoint_j%size[1]][k][2]=distance_lattice[hop_endpoint_i%size[0]][hop_endpoint_j%size[1]][k][2]+100*(hop_endpoint_i//size[0])+hop_endpoint_j//size[1] #adds index to distance lattice for looping purposes
        distance_lattice[hopping_ion_i][hopping_ion_j][k]=[-1,0,0]#clears the coordiates from the old site ->sticks in a vaccancy
        time_step_per_hop=(-1./gamma)*np.log(np.random.uniform(0,1))
        time_step_per_kmcstep=time_step_per_kmcstep+time_step_per_hop
        hop_histogram[endpoint_index]+=1

    #print "timestep is: " +str(time_step_per_kmcstep)


                    # if np.any(hop_probability_by_site!=0):
                    #     gamma=np.sum(hop_probability_by_site)
                    #     normalized_hop_probabilities=np.append(0.,np.cumsum(hop_probability_by_site)/gamma)
                    #     for endpoint_index in range(len(normalized_hop_probabilities))[:-1]:
                    #         if rho<=normalized_hop_probabilities[endpoint_index+1] and rho>normalized_hop_probabilities[endpoint_index]: #condition for hop set in VDVen paper
                    #             mc_lattice[i][j][k]=atom_types[1][1] #moving vacancy to initial lattice location
                    #             mc_lattice[endpoints[endpoint_index][0]%size[0]][endpoints[endpoint_index][1]%size[1]][k]=atom_types[1][0] #moving the ion to the appropriate vaccancy
                    #             distance_lattice[endpoints[endpoint_index][0]%size[0]][endpoints[endpoint_index][1]%size[1]][k]=distance_lattice[i][j][k] #moving the ion's coordinate to the appropriate site
                    #             distance_lattice[endpoints[endpoint_index][0]%size[0]][endpoints[endpoint_index][1]%size[1]][k][2]=distance_lattice[endpoints[endpoint_index][0]%size[0]][endpoints[endpoint_index][1]%size[1]][k][2]+100*(endpoints[endpoint_index][0]//size[0])+endpoints[endpoint_index][1]//size[1]
                    #             distance_lattice[i][j][k]=[-1,0,0]#clears the coordiates from the old site ->sticks in a vaccancy
                    #             hop_histogram[endpoint_index]+=1

    #print hop_histogram
    return (mc_lattice, distance_lattice, time_step_per_kmcstep, hop_probability_by_ion)

def presim_step(mc_lattice):#lite version of kmc_step, because you don't need to track the distances, just make the atoms move
    index_lattice=np.array(mc_lattice) #index lattice is the one to loop over does not evolve during a step, so as not to multiple hop the forward hopping ions
    time_step_per_hop=0
    time_step_per_kmcstep=0
    total_hop_probability=np.zeros(6)
    hop_probability_by_ion=np.tile(np.zeros(6),(size[0],size[1],size[2],1))
    dumbell_hop_probability=prefactor*np.exp(-(dumbell_hop_energy)/kb_t)

    """initialize the first lattice of hop_probabilitys"""
    for i in np.random.randint(low=0, high=size[0], size=(size[0],)):#looping randomly over every element in array, one KMC step
        for j in np.random.randint(low=0, high=size[1], size=(size[1],)):
            for k in range(size[2]):
                hop_start_point=float(mc_lattice[i][j][k])
                if int(mc_lattice[i][j][k])==int(atom_types[1][0]):#is there an ion on the site at the start of the cycle?
                    endpoints=np.array([[i+1,j],[i,j+1],[i-1,j+1],[i-1,j],[i,j-1],[i+1,j-1]]) #directions ion can hop
                    endpoint_occupancy=np.array([mc_lattice[endpoints[0][0]%size[0]][endpoints[0][1]%size[1]][0],mc_lattice[endpoints[1][0]%size[0]][endpoints[1][1]%size[1]][0],mc_lattice[endpoints[2][0]%size[0]][endpoints[2][1]%size[1]][0],mc_lattice[endpoints[3][0]%size[0]][endpoints[3][1]%size[1]][0],mc_lattice[endpoints[4][0]%size[0]][endpoints[4][1]%size[1]][0],mc_lattice[endpoints[5][0]%size[0]][endpoints[5][1]%size[1]][0]])
                    hop_probability_by_site=np.zeros(6)
                    endpoint_index=0
                    for (endpoint_index, endpoint) in list(enumerate(endpoints)): #defines local lattice, based on endpoint (direction indepenance), creates hop_probability_by_site a vector with probability/per site of hops
                        probability_multiplier=1
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
                                        local_lattice=np.array([[[mc_lattice[i][j-1][k]],[mc_lattice[i-1][j][k]],[mc_lattice[i-2][(j+1)%size[1]][k]],[mc_lattice[i-3][(j+2)%size[1]][k]]],[[mc_lattice[(i+1)%size[0]][j-1][k]],[hop_start_point],[hop_end_point],[mc_lattice[i-2][(j+2)%size[1]][k]]],[[mc_lattice[(i+1)%size[0]][j][k]],[mc_lattice[i][(j+1)%size[1]][k]],[mc_lattice[i-1][(j+2)%size[1]][k]],[mc_lattice[(i+3)%size[0]][j-1][k]]],[[mc_lattice[(i+2)%size[0]][j][k]],[mc_lattice[(i+1)%size[0]][(j+1)%size[1]][k]],[mc_lattice[i][(j+2)%size[1]][k]],[mc_lattice[(i+2)%size[0]][(j+2)%size[1]][k]]]])#good
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
                            if hop_probability_by_site[endpoint_index]!=dumbell_hop_probability: #if its not a dumbell hop,
                                hop_probability_by_site[endpoint_index]=probability_multiplier*get_hop_probability(np.transpose(local_lattice,(1,0,2)),prefactor,kb_t,endpoint_index)#gets the themal probability of the hop in question
                                #tetrahedral_path_count+=1*probability_multiplier
                        endpoint_index=endpoint_index+1
                    hop_probability_by_ion[i][j][k]=hop_probability_by_site[:]

    gamma=np.sum(hop_probability_by_ion)#gamma as defined in VDVen paper
    normalized_hop_probabilities=np.cumsum(hop_probability_by_ion)/gamma
    rho=np.random.uniform(0,1) #the random number
    lattice_index=np.searchsorted(normalized_hop_probabilities, rho) #get master index of hop
    endpoint_index=lattice_index%6 #get which endpoint the ion hopped to
    hopping_ion_i= (lattice_index//6)//size[0]
    hopping_ion_j= (lattice_index//6)%size[0]
    hop_endpoints=np.array([[hopping_ion_i+1,hopping_ion_j],[hopping_ion_i,hopping_ion_j+1],[hopping_ion_i-1,hopping_ion_j+1],[hopping_ion_i-1,hopping_ion_j],[hopping_ion_i,hopping_ion_j-1],[hopping_ion_i+1,hopping_ion_j-1]])
    hop_endpoint_i=hop_endpoints[endpoint_index][0]
    hop_endpoint_j=hop_endpoints[endpoint_index][1]
    mc_lattice[hopping_ion_i][hopping_ion_j][k]=atom_types[1][1] #moving vaccancy to initial ion location
    mc_lattice[hop_endpoint_i%size[0]][hop_endpoint_j%size[1]][0]=atom_types[1][0] #moving ion to appropriate endpoint

    for hop in range(int(size[0]*size[1]*size[2]*Li_concentration)):
        hop_probability_by_ion=kmc_evolve(mc_lattice,hop_probability_by_ion, [hopping_ion_i,hopping_ion_j,0],[hop_endpoint_i,hop_endpoint_j,0])
        gamma=np.sum(hop_probability_by_ion)#gamma as defined in VDVen paper
        normalized_hop_probabilities=np.cumsum(hop_probability_by_ion)/gamma
        rho=np.random.uniform(0,1) #the random number
        lattice_index=np.searchsorted(normalized_hop_probabilities, rho) #get master index of hop
        endpoint_index=lattice_index%6 #get which endpoint the ion hopped to
        hopping_ion_i= (lattice_index//6)//size[0]
        hopping_ion_j= (lattice_index//6)%size[0]
        hop_endpoints=np.array([[hopping_ion_i+1,hopping_ion_j],[hopping_ion_i,hopping_ion_j+1],[hopping_ion_i-1,hopping_ion_j+1],[hopping_ion_i-1,hopping_ion_j],[hopping_ion_i,hopping_ion_j-1],[hopping_ion_i+1,hopping_ion_j-1]])
        hop_endpoint_i=hop_endpoints[endpoint_index][0]
        hop_endpoint_j=hop_endpoints[endpoint_index][1]
        mc_lattice[hopping_ion_i][hopping_ion_j][k]=atom_types[1][1] #moving vaccancy to initial ion location
        mc_lattice[hop_endpoint_i%size[0]][hop_endpoint_j%size[1]][0]=atom_types[1][0] #moving ion to appropriate endpoint


    return (mc_lattice)




#play_lattice=np.transpose(np.array([[[-1,-1,-1,1],[-1,1,-1,1],[1,-1,1,1],[1,1,-1,-1]]])*-1.)
#play_lattice=np.transpose(np.array([[[1.,1.,1.,1.],[1.,-1.,1.,1.],[1.,1.,1.,1.],[1.,1.,1.,1.]]])*1)
#x= get_config_energy(play_lattice)
#y= get_excited_energy(play_lattice)
#print x
#print y
#print y-x






diffusion_coefficient_vs_concentration=[]
#for averaging_iterations in [1]:#range(1, 20,2):
averaging_iterations=0
size=[0,0,0]
dimensions=[25] #loop over this
simulation_iterations=100  #how many steps
presimulation_iterations=0 #how many pre simulation steps
concentrations= [0.1]#[0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]#np.array(range(1,100,6))/100. # looping over these too
average_path_counts=[]
for dimension in dimensions:# range(5,65,8): #to check size dependence of parameters
    size[0]=dimension
    size[1]=dimension
    size[2]=1
    t0=time.time()
    n_sites=size[0]*size[1]*size[2]

    for Li_concentration in concentrations:
        print ("concentration is: "+str(Li_concentration))
        averaging_index=0
        diffusion_coefficient=[]

        while averaging_index<=averaging_iterations:

            distance_lattice=np.tile([-1,0,0],(size[0],size[1],size[2],1))
            averaging_step=0
            simulation=0
            total_time=0
            """populate the lattice with the right concentration"""
            mc_lattice=np.zeros([size[0],size[1],size[2]])
            mc_lattice=mc_lattice+np.float32(atom_types[1][1])
            Li_atoms=0

            while Li_atoms <= Li_concentration*n_sites-1:
                i=np.random.randint(0,size[0])
                j=np.random.randint(0,size[1])
                k=np.random.randint(0,size[2])
                if mc_lattice[i][j][k] == atom_types[1][1]:
                    mc_lattice[i][j][k] = np.float32(atom_types[1][0])
                    Li_atoms+=1



            while averaging_step < presimulation_iterations: #initializes the mc_lattice for the simulation to avoid starting artifacts
                mc_lattice=presim_step(mc_lattice)
                if averaging_step%10==0:
                    print averaging_step
                #colorsquare(np.transpose(mc_lattice)[0],"lattice_pictures/"+str(averaging_step)+".png") $to see the lattice

                averaging_step+=1


            for i in range(size[0]):  #need to initialize the distance_lattice after presimulation
                for j in range(size[1]):
                    for k in range(size[2]):
                        if mc_lattice[i][j][k]==atom_types[1][0]:
                            distance_lattice[i][j][k]=list([i,j,k]) #put in a starting coordinate tag for every ion
                        else:
                            distance_lattice[i][j][k]=[-1,0,0] #puts a void location at every vaccancy
            r_squared_vs_time=[]
            x_axis_times=[]

            """THE KMC LOOP"""
            #path_counter=[]
            while simulation < simulation_iterations: #how many kmc steps to take
                (mc_lattice,distance_lattice,time_step,hop_probability_by_ion)=kmc_step(mc_lattice,distance_lattice)
                if simulation%10==0:
                    print "simulation step: " +str( simulation)
                #colorsquare(np.transpose(mc_lattice)[0],"lattice_pictures/"+str(simulation+10)+".png")

                total_time=total_time+time_step #the total time taken
                diffusion_cycle,distances_out=get_diffusion_coefficient(distance_lattice,size,total_time,lattice_constant,Li_concentration) #getting the diffusion Coefficient, and the distances travelled by each atom
                r_squared_vs_time=np.append(r_squared_vs_time,np.average(distances_out))
                x_axis_times=np.append(x_axis_times,total_time)

                simulation+=1
            print "total time is: " +str(total_time)
            #average_path_counts=np.append(average_path_counts,(np.average(path_counter[::3]), np.average(path_counter[1::3]), np.average(path_counter[2::3])))
            (diffusion_coefficient_cycle,tmp)=get_diffusion_coefficient(distance_lattice,size,total_time,lattice_constant,Li_concentration)
            diffusion_coefficient=np.append(diffusion_coefficient,diffusion_coefficient_cycle)
            #plt.figure()
            #averaging_value=25
            #average_distances=np.mean(r_squared_vs_time.reshape(-1, averaging_value), axis=1)
            #distances_errors=np.std(r_squared_vs_time.reshape(-1, averaging_value), axis=1)
            #plt.errorbar(x_axis_times[::averaging_value],average_distances,distances_errors)
            #plt.plot(range(simulation_iterations)[::averaging_value],average_distances)
            #plt.show()
            #plt.pause(10000)
            averaging_index+=1
        final_diffusion_coefficient=[np.average(diffusion_coefficient),np.std(diffusion_coefficient)]

        diffusion_coefficient_vs_concentration=np.append(diffusion_coefficient_vs_concentration,[final_diffusion_coefficient[0],Li_concentration,final_diffusion_coefficient[1]])#need to put in masterloop variable here
    t1=time.time()
    print t1-t0


#plt.figure()
#plt.plot(concentrations,average_path_counts[::3])
#plt.plot(concentrations,average_path_counts[1::3])
#plt.plot(concentrations,average_path_counts[2::3])
#plt.show()
#plt.pause(1000)
#plt.plot(range(0,averaging_iterations+1),diffusion_coefficient)
#plt.errorbar(diffusion_coefficient_vs_concentration[1::3],np.log(diffusion_coefficient_vs_concentration[::3]),diffusion_coefficient_vs_concentration[2::3])
#
#

output_file=open(str(Li_concentration)+'LiCoO2_kmc.out','w')
output_file.write('Size:'+ str(size)+'\n Concentration:' +str(Li_concentration)+'\n kbT:' +str(kb_t)+'\n Simulation parameters: \n Presimulation cycles: ' +str(presimulation_iterations)+'\n Simulation Cycles: ' +str(simulation_iterations)+'\n')
output_file.write("\n RESULTS \n")
output_file.write('Averaged Diffusion Coefficients:'+str(diffusion_coefficient_vs_concentration[::3])+'\n Variables: ' +str(diffusion_coefficient_vs_concentration[1::3])+ '\n Errors on D_star: ' +str(diffusion_coefficient_vs_concentration[2::3]))
output_file.close()


print "diffusion_coefficient_vs_concentration is: "+str( diffusion_coefficient_vs_concentration)
