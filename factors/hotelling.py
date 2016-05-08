import numpy as np

epsilon = 0.01


def hotelling(x: np.matrix):
    s = x.sum(axis=1)
    alpha = s / np.amax(s)
    # print("alpha = " + str(alpha))

    alpha_old = alpha - alpha
    while (np.absolute(alpha - alpha_old)).sum() >= epsilon:
        # print("============================")
        beta = x.dot(alpha)
        # print("beta = " + str(beta))
        alpha_old = alpha
        alpha = beta / np.amax(beta)
        # print("alpha = " + str(alpha))
        # print("difference = " + str(np.absolute(alpha - alpha_old)))
        # print("diff_sum = " + str((np.absolute(alpha - alpha_old)).sum()))
    return alpha, np.amax(beta)
