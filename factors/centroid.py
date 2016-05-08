import numpy as np
from math import sqrt


def centroid(x: np.matrix):
    t = x.sum(axis=0)
    s = x.sum()
    return t / sqrt(s)


def reflect(x: np.matrix):
    r = x.__copy__()
    signs = np.zeros(shape=x.shape[0])
    # if np.sum(x) < 0:
    #     r *= -1
    #     signs.fill(1)
    # else:
    for i in range(0, r.shape[0]):
        if (r[:, i] < 0).sum() >= r.shape[0] / 2:
            r[i, :] *= -1
            r[:, i] *= -1
            signs[i] += 1
    return r, signs
