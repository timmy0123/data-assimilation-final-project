"""
General settings
"""
import numpy as np

# parameters
N = 40              # number of grid point
F = 8.              # forcing term

Tmax = 10.          # time length of the experiment
dT = 0.05           # forecast-analysis cycle length
nT = int(Tmax / dT) # number of cycles
