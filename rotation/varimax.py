import numpy as np
from math import pi


def varimax_angle(f: np.matrix):
    u = np.square(f[:, 0]) - np.square(f[:, 1])
    v = 2 * np.multiply(f[:, 0], f[:, 1])
    a = np.sum(u)
    b = np.sum(v)
    c = np.sum(np.square(u) - np.square(v))
    d = 2 * np.sum(np.multiply(u, v))

    numerator = d - 2 * a * b / f.shape[0]
    denominator = c - (a * a - b * b) / f.shape[0]

    tan4phi = numerator / denominator

    print("numerator = " + str(numerator))
    print("denominator = " + str(denominator))
    print("tan = " + str(tan4phi))

    phi = np.arctan(numerator / denominator)
    if numerator > 0 > denominator:
        phi += pi
    else:
        if numerator < 0 and denominator < 0:
            phi -= pi
    print("4phi = " + str(phi))
    phi /= 4
    print("phi = " + str(phi))
    return phi


def normalize_saturation(f: np.matrix):
    saturation = np.sqrt(np.square(f[:, 0]) + np.square(f[:, 1]))
    print("saturation: " + str(saturation))
    f1 = f[:, 0] / saturation
    f1 = np.vstack((f1, f[:, 1] / saturation))
    return f1, saturation


def denormalize_saturation(f: np.matrix, saturation: np.matrix):
    f1 = np.multiply(f[:, 0], saturation[:, None])
    f1 = np.hstack((f1, np.multiply(f[:, 1], saturation[:, None])))
    return f1
