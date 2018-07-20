# from SALib.sample import latin
from SALib.analyze import delta
import numpy as np
# from scipy.stats.distributions import norm
#
# from pcraster._pcraster import *
# from pcraster.framework import *
# from model_v2 import *

# Option to use pyDOE (same sampling results)
# https://pythonhosted.org/pyDOE/randomized.html#randomized
# from pyDOE import *


def scale(bounds):
    scaled = []
    for e in bounds:
        se = []
        up = max(e[0], e[1])
        se.append(e[0] / up)
        se.append(e[1] / up)
        scaled.append(se)

    return scaled


def get_upper(bounds):
    upper = []
    for e in bounds:
        upper.append(e[1])
    return upper


def saveLHSmatrix(param_values):
    np.savetxt("lhs_vectors.txt", param_values)


def get_runs(params):
    runs = int(params.shape[0])
    print("Runs: ", int(runs))
    return runs




"""
Attention!!
- The 'problem' here has fewer variables than the Morris problem!
- Bounds have changed based on Morris results for:
    - Koc
"""


def get_problem(Mini_TEST = False):
    if Mini_TEST:
        bounds = [[0.75, 0.99],
                  [0.01, 1.0], [0.01, 1.0],  # c_lf
                  [0.001, 1.0]]
        problem = {
            'num_vars': len(bounds),
            'names': ['z3_factor',
                      'cZ0Z1', 'cZ',
                      'c_adr'],
            'bounds': scale(bounds)
        }
    else:
        # Will be scaled with scale_morris()
        bounds = [[0.75, 0.99],  # z3_factor
                  [0.01, 1.0], [0.01, 1.0],  # c_lf
                  [0.001, 1.0],  # cadr
                  [1100.0, 3650.0],  # k_g
                  [0.0001, 1.0], [0, 1.0],  # gamma
                  [0.0001, 1.0],  # f_transp
                  [0.01, 0.05], [7.0, 16180],  # f_oc, k_oc
                  [0.01, 1.0],  # beta_runoff
                  [140.0, 7000.0],  # age_rate
                  [130.0, 230.0],  # dt_50_ab
                  [26.0, 37.0],  # dt_50_ref
                  [1.0, 3.0],  # epsilon (in absolute, convert to negative!!)
                  [0.01, 1.0]]  # beta_moisture

        names = ['z3_factor',
                 'cZ0Z1', 'cZ',
                 'c_adr',
                 'k_g',
                 'gamma01', 'gammaZ',
                 'f_transp',
                 'f_oc', 'k_oc',
                 'beta_runoff',
                 'dt_50_aged',
                 'dt_50_ab',
                 'dt_50_ref',
                 'epsilon_iso',
                 'beta_moisture'
                 ]

        problem = {
            'num_vars': len(bounds),
            'names': names,
            'bounds': scale(bounds),
            'upper': get_upper(bounds)
        }
    return problem


def get_vector_test():
    names = ['z3_factor',
             'cZ0Z1', 'cZ',
             'c_adr',
             'k_g',
             'gamma01', 'gammaZ',
             'f_transp',
             'f_oc', 'k_oc',
             'beta_runoff',
             'dt_50_aged',
             'dt_50_ab',
             'dt_50_ref',
             'epsilon_iso',
             'beta_moisture'
             ]
    # print("names length : " + str(len(names)))
    import csv
    ini_path = 'initial.csv'
    test_param = dict()  # Dictionary to store the values
    with open(ini_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            test_param[row[0].strip()] = float(row[1])
    name_values = []
    for i in range(len(names)):
        name_values.append(test_param[names[i]])
    # print("vales length : " + str(len(name_values)))
    return name_values