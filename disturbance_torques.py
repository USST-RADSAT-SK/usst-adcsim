import util as ut
import numpy as np
from CubeSat_model import CubeSat

a_solar_constant = 1366 / (3 * (10 ** 8))
a_gravity_gradient_constant = 3 * 3.986004418 * (10**14)

# Note: it appears that ut.cross_product_operator is a significant amount faster than using np.cross()


def gravity_gradient(ue, r0, cubesat: CubeSat):
    return (a_gravity_gradient_constant/(r0**3)) * (ut.cross_product_operator(ue) @ cubesat.inertia @ ue)


def aerodynamic_torque(v, rho, cubesat: CubeSat):
    """
    v: velocity of spacecraft relative to air in body frame in m/s
    rho: atmospheric density in kg/m^3
    cubesat: model of the faces of the cubesat
    """
    # This function uses the equations for the free molecular flow dynamics
    # (i.e. neglecting the affect of reemitted particles on the incident stream)

    vm = np.linalg.norm(v)
    ev = v * (1.0 / vm)

    # calculate net torque
    net_torque = np.zeros(3)
    for face in cubesat.faces:
        mu = np.dot(ev, face.normal)  # cosine of angle between velocity and surface normal
        if mu >= 0:
            # units N, see eq'n 2-2 in NASA SP-8058 Spacecraft Aerodynamic Torques
            force = -rho * vm**2 * (0.4 * mu**2 * face.normal + 0.8 * mu * ev) * \
                    face.area  # units N, see eq'n 2-2 in NASA SP-8058 Spacecraft Aerodynamic Torques
            net_torque += ut.cross_product_operator(face.centroid - cubesat.center_of_mass) @ force  # units N.m

    return net_torque


def solar_pressure(sun_vec, cubesat: CubeSat):
    # TODO: absorption, and diffuse reflection (and redo this one)
    net_torque = np.zeros(3)
    for face in cubesat.faces:
        h = face.normal @ sun_vec

        if h > 0:
            net_torque += a_solar_constant * face.area * (1 + face.reflection_coeff) * h * \
                          (ut.cross_product_operator(face.centroid - cubesat.center_of_mass) @
                           -face.normal)  # force is opposite to area

    return net_torque
