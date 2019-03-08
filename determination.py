import numpy as np


def TRIAD(vn, vb):
    """

    :param vn: vectors in inertial frame
    :param vb: vectors in body frame
    :return:
    """
    # normalize vectors
    vn[0] = vn[0] / np.linalg.norm(vn[0])
    vn[1] = vn[1] / np.linalg.norm(vn[1])
    vb[0] = vb[0] / np.linalg.norm(vb[0])
    vb[1] = vb[1] / np.linalg.norm(vb[1])

    # Body frame triad vectors
    cross_prod_b = np.cross(vb[0], vb[1])
    tb1 = vb[0]
    tb2 = cross_prod_b / np.linalg.norm(cross_prod_b)
    tb = np.array([tb1, tb2, np.cross(tb1, tb2)])

    # Inertial frame triad vectors
    cross_prod_n = np.cross(vn[0], vn[1])
    tn1 = vn[0]
    tn2 = cross_prod_n / np.linalg.norm(cross_prod_n)
    tn = np.array([tn1, tn2, np.cross(tn1, tn2)])

    BT = tb.T
    NT = tn.T

    BN = BT @ (NT.T)

    return BN


if __name__ == '__main__':
    vb = [[0.8190, -0.5282, 0.2242], [-0.3138, -0.1584, 0.9362]]
    vn = [[1, 0, 0], [0, 0, 1]]
    print(TRIAD(vn, vb))
