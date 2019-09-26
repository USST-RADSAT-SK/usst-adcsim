"""
Functions in this file output the result of the differential equations that are used to propagate attitude given the
state passed to them (state includes attitude information and angular velocity information).

Includes the 'differential attitude equation" and the 'differential angular velocity equation' (Euler's rotational
equation of motion).

The angular velocity equation is always the same, but the attitude equation can use any attitude parametrization.
"""
import numpy as np
from adcsim import util as ut
from adcsim.CubeSat_model import CubeSat
from adcsim.disturbance_torques import DisturbanceTorques
from adcsim.simulations.sim import AttitudeData, OrbitData


# differential equations for an MRP attitude parametrization
def state_dot_mrp(time: float, state: np.ndarray, attitude: AttitudeData, orbit: OrbitData, cubesat: CubeSat, disturbance_torques: DisturbanceTorques):
    """
    Differential attitude equation for the Modified Rodriguez Parameters (MRP) attitude parametrization, with eulers
    rotational equation of motion.
    :param state: np.ndarray. The current attitude state (first index is attitude, second index is angular velocity)
    :param
    :return: np.ndarray. First index is value of attitude differential equation, second is value of angular velocity
    differential equation
    """
    # sigmas
    a = (1 - state[0] @ state[0]) * np.identity(3) + 2 * ut.cross_product_operator(state[0]) + 2 * \
        np.outer(state[0], state[0])
    sigma_propagation = (1/4) * a  @ state[1]

    control = disturbance_torques.torque(time, state, attitude, orbit, cubesat)

    # omegas
    omega_propagation = cubesat.inertia_inv @ ((-ut.cross_product_operator(state[1]) @ cubesat.inertia @ state[1]) + control)

    return np.array([sigma_propagation, omega_propagation])


# old (could be in 'old' folder)
# equations change with reference frame
def state_dot_ref_frame(state, control, omega_r, inertia, inertia_inv):
    # sigmas
    a = (1 - state[0] @ state[0]) * np.identity(3) + 2 * ut.cross_product_operator(state[0]) + 2 * \
        np.outer(state[0], state[0])
    sigma_propagation = (1/4) * a  @ (state[1] - omega_r)

    # omegas
    omega_propagation = inertia_inv @ ((-ut.cross_product_operator(state[1]) @ inertia @ state[1]) + control)

    return np.array([sigma_propagation, omega_propagation])


# old (could be in 'old' folder)
# equations change with reaction wheels
def state_dot_reaction_wheels(state, control, inertia_rw, inertia_inv_rw, hs):
    # sigmas
    a = (1 - state[0] @ state[0]) * np.identity(3) + 2 * ut.cross_product_operator(state[0]) + 2 * \
        np.outer(state[0], state[0])
    sigma_propagation = (1/4) * a  @ state[1]

    # omegas
    cross = ut.cross_product_operator(state[1])
    omega_propagation = inertia_inv_rw @ ((-cross @ inertia_rw @ state[1]) - cross @ hs + control)

    return np.array([sigma_propagation, omega_propagation])
