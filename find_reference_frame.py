import transformations as tr
import numpy as np


def get_mrp_br(dcm_rn, sigma):
    dcm_br = tr.mrp_to_dcm(sigma) @ dcm_rn.T
    return tr.dcm_to_mrp(dcm_br)


def get_dcm_rn(t):
    # create reference frame
    v = np.array([0.5, 0.5, 0.1])  # create a vector that represents euler angle rotation
    dcm_rn = tr.euler_angles_to_dcm(v, type='3-2-1')  # find dcm corresponding to euler angle rotation

    # slowly rotate the reference frame about it's own z-axis
    omega = -np.pi / 180.
    s, c = np.sin(omega * t), np.cos(omega * t)
    dcm_rn[:, 0], dcm_rn[:, 1] = c * dcm_rn[:, 0] + s * dcm_rn[:, 1], -s * dcm_rn[:, 0] + c * dcm_rn[:, 1]
    return dcm_rn
    # dcm_rn = np.eye(3)
    # omega = -np.pi / 50.
    # s, c = np.sin(omega * t), np.cos(omega * t)
    # dcm_rn[:, 0], dcm_rn[:, 1] = c * dcm_rn[:, 0] + s * dcm_rn[:, 1], -s * dcm_rn[:, 0] + c * dcm_rn[:, 1]
    # return dcm_rn


def get_omega_r(t):
    return np.array([0., 0., -np.pi / 180.])


def get_omega_r_dot(t):
    return np.array([0., 0., 0.])

