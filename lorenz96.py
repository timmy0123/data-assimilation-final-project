"""
Definition of the Lorenz 96 (40-variable) model
"""
import numpy as np

def f(t, y, F):
    return (np.roll(y, -1) - np.roll(y, 2)) * np.roll(y, 1) - y + F
