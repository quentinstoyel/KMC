import matplotlib.pyplot as plt
import numpy as np
import pickle
import math

#normcons_dictionary = pickle.load(open("master_dictionary.p", "rb"))

normcons_dictionary = pickle.load(open("hex_dict.p", "rb"))


for data_dict in normcons_dictionary:
    x = str(data_dict).count("1")
    y = normcons_dictionary[data_dict]
    if -math.log(y / 10**(13)) * 0.025 < 0.25:
        print data_dict
    plt.scatter(x, -math.log(y / 10**(13)) * 0.025)
    # plt.scatter(x,y)

plt.show()
