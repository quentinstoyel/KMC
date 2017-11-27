# KMC

Kinetic Monte Carlo Code for Lithium Diffussion
By no means a general or transferable piece of code.  For all academic purposes, the reader is refered to:
A. Van der Ven, G. Ceder, M. Asta, and P. D. Tepesch. Phys. Rev. B 64, 184307 – Published 25 October 2001

General Structure of the code:

Creats a 2d Hexagonal lattice (defined by "size/dimension") with a random distribution of Li atoms on it.
Performs a number of KMC steps on the lattice, each step consists of:
1. determining which Li atoms can hop
2. for each of those, hand the local lattice to a dictionary that returns the hop probability. 
3. randomly (weighted by the probabilities) making one atom hop
4. repeating steps 1-3 as many times as there are atoms

once enough steps are done, the code calculates the diffusion coefficient(get_diffusion_coeficient) of the material

What is behind each step:
1. checks all the endpoints of a hop to see which ones are empty
2. reads from dictionary.  The probabilities in the dictionary are determined fro the energy of the local lattice in the original state (get_config_energy) and at the highest energy point on the hop (get_excited_energy) to get the energy barrier, and then puts that into  exp(-E_barrier/kbT) (get_hop_probability).  the probabilities are written to a matrix the same size as the lattice.
3. cumsum all of the probabilities, normalize these values so they are between 0-1, then pick a random number between 0-1, choose the corresponding hop
4. for loops all the way down

That gives you a single KMC cycle, these are done multiple times at each concentration to get a statistically significant data set.  
