import numpy as np
import transformations as tr
import util as ut


# Solution Q1
sun_vec = np.random.random(3)
sun_vec = sun_vec/np.linalg.norm(sun_vec)

mag_vec = np.random.random(3)
mag_vec = mag_vec/np.linalg.norm(mag_vec)


# Solution Q2
dcm_bn = tr.mrp_to_dcm(sigmas[i])
mag_vec_b_true = dcm_bn @ mag_vec
sun_vec_b_true = dcm_bn @ sun_vec


# Solution Q3 (not optimized)
def triad(v1n, v2n, v1b, v2b):
    t1n = v1n
    cross = ut.cross_product_operator(v1n) @ v2n
    t2n = cross/np.linalg.norm(cross)
    t3n = ut.cross_product_operator(t1n) @ t2n

    matn = np.array([t1n, t2n, t3n]).T

    t1b = v1b
    cross = ut.cross_product_operator(v1b) @ v2b
    t2b = cross/np.linalg.norm(cross)
    t3b = ut.cross_product_operator(t1b) @ t2b

    matb = np.array([t1b, t2b, t3b]).T

    return matb @ matn.T


# Solution Q4
dcm_bn_triad = triad(sun_vec, mag_vec, sun_vec_b_true, mag_vec_b_true)
sigmas_triad = tr.dcm_to_mrp(dcm_bn_triad)
controls[i] = control_torque(sigmas_triad, omegas[i])


# Solution Q5 (quick and easy)
mag_vec_b_true = dcm_bn @ mag_vec + np.array([0.05, 0.05, 0.05])
sun_vec_b_true = dcm_bn @ sun_vec + np.array([0.01, 0.01, 0.01])
mag_vec_b_true = mag_vec_b_true / np.linalg.norm(mag_vec_b_true)
sun_vec_b_true = sun_vec_b_true / np.linalg.norm(sun_vec_b_true)


# Solution Q6
solution = None
