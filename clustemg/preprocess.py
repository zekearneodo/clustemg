import os
import logging

import h5py
import json
import pandas as pd
import numpy as np

from scipy import stats

logger = logging.getLogger("clustemg.preprocess")


class LeakyIntegrator:
    # a very inneficient implementation of 
    # dx/dt = - x/tau + y
    # where y is the incoming perturbation

    def __init__(self, tau):
        self.alpha = 1./tau # Leaking factor
        self.state = 0  # Initial state

    def update(self, x, dt):
        self.state = (1 - self.alpha * dt) * self.state + x * dt
        return self.state
    
def leaky_integrate(x, tau, s_f=1000):
    t = np.arange(x.size)/s_f
    dt = 1/s_f
    integrator = LeakyIntegrator(tau=tau)
    x_smooth = np.array([integrator.update(z, dt) for z in x])
    return x_smooth


def pad_spectrogram(spectrogram: np.array, pad_length: int) -> np.array:
    """ Pads a spectrogram to being a certain length
    """
    excess_needed = pad_length - np.shape(spectrogram)[1]
    pad_left = np.floor(float(excess_needed) / 2).astype("int")
    pad_right = np.ceil(float(excess_needed) / 2).astype("int")
    return np.pad(
        spectrogram, [(0, 0), (pad_left, pad_right)], "constant", constant_values=0
    )