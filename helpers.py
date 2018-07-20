import pickle
import os
import numpy as np
from constants import *


def get_results_map(var, level):
    # TODO: This function should create a dictionary for a specific realization/var (delta or Conc)
    # TODO: It should also assign metrics based on one of the data considered (i.e. level = catch, composite or comp + detailed)
    try:
        results = pickle.load(open("resultsmap.p", "rb"))
    except IOError:
        results = dict()

    for pc in computers:  # Multiple computers, with specific names running models
        for model in models:  # Two types: fix and var
            RUNS = True  # Keeps track of how many sets have been tested on each machine
            version = 1  # Counter for 'experiments' made, eg. var1, var2, etc...
            while RUNS:  # Will turn to False, when no more experiments of this model version
                path = './LHS_' + pc + model + str(version)
                if os.path.exists(path):
                    matrix = np.loadtxt(path + "/lhs_vectors.txt")  # Sets for this experiment
                    sets = matrix.shape[0]  # No. of rows/sets in experiment/matrix
                    for row in range(sets):
                        vector = matrix[row]
                        for name in names:  # Parameter name tested
                            results[name] = vector[names.index(name)]  # Assign parameter value to dict()
                            for measure in measures:  # Assign the respective likelihood/metric to each parameter
                                results[name][measure] = get_likelihood(measure, var, level)
                    version += 1
                else:
                    RUNS = False

    # if favorite_color:
    #     pickle.dump(favorite_color, open("resultsmap.p", "wb"))


def get_likelihood(measure, var, level):
    # Get obs (for specific level)
    # get SIM (for specific level)
    # Merge
    # Compute metric
    raise NotImplementedError


get_results_map()
