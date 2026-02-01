import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import ks_2samp

def centroid_drift(a, b):
    return cosine(a.mean(axis=0), b.mean(axis=0))

def ks_drift(a, b):
    stat, p = ks_2samp(a, b)
    return stat, p
