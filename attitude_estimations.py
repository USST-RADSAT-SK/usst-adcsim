import util as ut
import numpy as np
import transformations as tr


def triad(v1n, v2n, v1b, v2b):
    # note: first vector should be the one that is known more accurately
    t1n = v1n
    cross = ut.cross_product_operator(v1n) @ v2n
    t2n = cross/np.linalg.norm(cross)
    t3n = ut.cross_product_operator(t1n) @ t2n

    matn = np.array([t1n, t2n, t3n])

    t1b = v1b
    cross = ut.cross_product_operator(v1b) @ v2b
    t2b = cross/np.linalg.norm(cross)
    t3b = ut.cross_product_operator(t1b) @ t2b

    matb = np.array([t1b, t2b, t3b]).T

    return matb @ matn


def mrp_triad_with_noise(sigma, v1n, v2n, arbitrary_error1, arbitrary_error2):
    # TODO: make the error not arbitrary (e.g. This function could produce a random unit vector in a conical section defined by some angle input)

    dcm_bn = tr.mrp_to_dcm(sigma)
    v1b = dcm_bn @ v1n + np.random.random_sample(3) * arbitrary_error1
    v2b = dcm_bn @ v2n + np.random.random_sample(3) * arbitrary_error2
    v1b = v1b/np.linalg.norm(v1b)
    v2b = v2b/np.linalg.norm(v2b)

    dcm_bn_triad = triad(v1n, v2n, v1b, v2b)
    return tr.dcm_to_mrp(dcm_bn_triad)
