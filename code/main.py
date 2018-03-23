#!/usr/bin/python3

import load
import numpy as np
import matplotlib.pyplot as plt
import utils


if __name__=="__main__":
    s = load.load()
    sampling_n = np.geomspace(12, utils.max_len(s), num=5)
    sampling_n = [int(i) for i in sampling_n]
    #sampling_n = [12,24]
    stress_samples = 1000
    stress = []
    for n in sampling_n:
        stress.append(utils.eval_stress(s,nb_points=n,num_rows_2_sample=stress_samples))
    print (sampling_n)
    print(stress)
    plt.plot(sampling_n, stress)
    plt.show()