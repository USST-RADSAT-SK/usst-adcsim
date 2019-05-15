import numpy as np
from adcsim import util as ut


# control torque equations taken from Part3/3_Control.Mod.3.Slides.pdf page 27.
# note: terms in the equation are missing here for this implementation
def control_torque(sigma, omega, inertia, K, P, i, prev_torque, max_torque=None):

    # restrict control torque update to be only once per 100 iterations
    if i % 100 != 0:
        return prev_torque

    # control torque law equation
    val = -K * sigma - P * omega + ut.cross_product_operator(omega) @ inertia @ omega

    # if a max_torque is specified limit the torque along each dimension to the max torque
    if max_torque is not None:
        val[abs(val) > max_torque] = max_torque * np.sign(val)[abs(val) > max_torque]

    return val


def control_torque_ref_frame(sigma, omega, omega_r, omega_dot_r, inertia, K, P, i, prev_torque, max_torque=None):

    # restrict control torque update to be only once per 100 iterations
    if i % 100 != 0:
        return prev_torque

    # control torque law equation
    val = -K * sigma - P * (omega - omega_r) + ut.cross_product_operator(omega) @ inertia @ omega
    val += inertia @ (omega_dot_r - ut.cross_product_operator(omega) @ omega_r)

    # if a max_torque is specified limit the torque along each dimension to the max torque
    if max_torque is not None:
        val[abs(val) > max_torque] = max_torque * np.sign(val)[abs(val) > max_torque]

    return val


def reaction_wheel_control(sigma, omega, inertia_rw, K, P, i, prev_torque, hs, max_torque=None):
    # restrict control torque update to be only once per 100 iterations
    if i % 100 != 0:
        return prev_torque

    # control torque law equation
    val = -K * sigma - P * omega + ut.cross_product_operator(omega) @ (inertia_rw @ omega + hs)

    # if a max_torque is specified limit the torque along each dimension to the max torque
    if max_torque is not None:
        val[abs(val) > max_torque] = max_torque * np.sign(val)[abs(val) > max_torque]

    return val


def b_dot(B, Bprev, K, i, prev_m, time_diff_measurements):

    if i % 100 != 0:
        return prev_m

    Bdot = (B - Bprev)/time_diff_measurements

    m = -K * Bdot

    max_m = 0.3
    m[abs(m) > max_m] = max_m * np.sign(m)[abs(m) > max_m]

    return m  # this is not a control torque, its a magnetic moment
