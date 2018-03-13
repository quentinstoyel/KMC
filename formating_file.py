
# the list of polynomials multiplying each eci
average_occupancy = [np.zeros(len(multiplicities))]
for config in dft_configurations:
  x = [1, sum(config), (config[1] + config[3]), (config[4] + config[7]), (config[5] + config[8]), (config[6] + config[9]), config[2], config[6] * config[9],
       (config[4] * config[1] + config[3] * config[7]), (config[4] * config[5] * config[6] + config[7] * config[8] * config[9]), config[1] * config[2] * config[3],config[5]*config[2]*config[8],(config[5]*config[9]+config[8]*config[6]),(config[5]*config[1]+config[8]*config[3]),(config[2]*config[4]+config[2]*config[7]),(config[5]*config[6]+config[8]*config[9]),(config[5]*config[4]+config[8]*config[7]),(config[2]*config[1]+config[2]*config[1]),(config[2]*config[1]*config[4]+config[2]*config[3]*config[7]),(config[5]*config[4]*config[1]+config[8]*config[7]*config[3]),(config[5]*config[6]*config[9]+config[8]*config[9]*config[6]),(config[1]*config[3]),(config[4]*config[6]+config[7]*config[9])]
  average_occupancy = np.append(average_occupancy, [x], axis=0)


def minimizer(eci):  # this is what needs to be minimized -it is like your chi square
  return sum(np.square(dft_activation_energies - np.dot(average_occupancy[1:], eci)))


# does least square fitting of the minimizer function with respect to the eci input
fit = minimize(minimizer, local_eci, method='Powell')
# print dft_activation_energies, np.dot(average_occupancy[1:],fit.x)

print fit


for index in range(len(fit.x)):
  print round(fit.x[index], 10)

cross_validation_score = (fit.fun / len(dft_configurations))**(0.5)
print "CRV is: "+str(cross_validation_score)
