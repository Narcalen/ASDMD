import numpy as np
from numpy.linalg import matrix_rank, norm

epsilon = 1e-8


def nipals1(x: np.matrix):

    max_it = 3
    # max_it = matrix_rank(x)
    # print("rank(x) = " + str(max_it))
    p = np.zeros(shape=(x.shape[1], max_it))
    t = np.zeros(shape=(x.shape[0], max_it))

    for j in range(0, max_it):
        # a = norm(x, axis=0)
        # idx = a.argmax()
        t_j = x[:, j]

        # # verify if the column is a zero vector
        # zero_col = np.zeros(shape=t_j.shape)
        # if norm(t_j - zero_col) <= epsilon:
        #     j += 1
        #     max_it += 1
        #     continue

        t_j1 = t_j - t_j
        while norm(t_j1 - t_j) > epsilon:
            p_j = x.transpose().dot(t_j)
            p_j /= norm(p_j)
            t_j1 = t_j
            t_j = x.dot(p_j)

        p[:, j] = p_j
        t[:, j] = t_j
        j += 1
        x = x - np.dot(t_j[:, None], p_j[None, :])

    return t, p
