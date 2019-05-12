import util as ut
import numpy as np
from CubeSat_model import CubeSat

a_solar_constant = (3.823 * 10**26) / (3 * (10 ** 8)) / 4 / np.pi
a_solar_constant_2 = (3.823 * 10**26) / 4 / np.pi
a_gravity_gradient_constant = 3 * 3.986004418 * (10**14)
a_earth_rotational_constant = 0.000072921158553
u0 = 4 * np.pi * 10**-7

# Note: it appears that ut.cross_product_operator is a significant amount faster than using np.cross()


def gravity_gradient(ue, r0, cubesat: CubeSat):
    return (a_gravity_gradient_constant/(r0**3)) * (ut.cross_product_operator(ue) @ cubesat.inertia @ ue)


def get_air_velocity(vel_inertial, pos_inertial):
    # this would be equivalent to dcm @ (vel_inertial - ut.cross_product_operator(a_earth_rotational_constant*np.array([0, 0, 1])) @ pos_inertial)
    # equation 3.160 and 3.161 in Landis Markley's Fundamentals of Spacecraft Attitude Determination and Control
    # textbook
    return np.array([vel_inertial[0] + a_earth_rotational_constant*pos_inertial[1],
                     vel_inertial[1] - a_earth_rotational_constant*pos_inertial[0],
                     vel_inertial[2]])


def aerodynamic_torque(v, rho, cubesat: CubeSat):
    """
    v: velocity of spacecraft relative to air in body frame in m/s
    rho: atmospheric density in kg/m^3
    cubesat: model of the cubesat, that includes all the faces
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
            force = -rho * vm**2 * (2 * (1-face.accommodation_coeff) * mu**2 * face.normal +
                                    face.accommodation_coeff * mu * ev) * face.area  # units N, see eq'n 2-2 in NASA SP-8058 Spacecraft Aerodynamic Torques
            # This addition is the diffuse term in Chris Robson's thesis. Just use 5 % of vm for now for the diffuse vel
            # note: I think Chris is missing a second cosine in the specular reflection equation
            force += face.accommodation_coeff * rho * vm * (0.05*vm) * face.area * mu * face.normal
            net_torque += ut.cross_product_operator(face.centroid - cubesat.center_of_mass) @ force  # units N.m

    return net_torque


def solar_pressure(sun_vec, sun_vec_inertial, satellite_vec_inertial, cubesat: CubeSat):
    """
    Calculate the solar pressure torque on the cubesat.
    :param sun_vec: sun unit vector in body frame
    :param sun_vec_inertial: sun vector in inertial frame
    :param satellite_vec_inertial: satellite position vector in inertial frame
    :param cubesat: model of the cubesat, that includes all the faces
    """

    # calculate the solar irradiance using equation 3-53 in Chris Robson's thesis
    const = a_solar_constant / (np.linalg.norm(sun_vec_inertial - satellite_vec_inertial)) ** 2

    net_torque = np.zeros(3)
    for face in cubesat.faces:
        cos_theta = face.normal @ sun_vec

        if cos_theta > 0:
            # calculate the solar radiation pressure torque using the equation in SMAD
            # net_torque += const * face.area * (1 + face.reflection_coeff) * cos_theta * \
            #               (ut.cross_product_operator(face.centroid - cubesat.center_of_mass) @
            #                -face.normal)  # force is opposite to area

            # Calculate the solar radiation pressure torque using equation 3.167 in Landis Markley's Fundamentals of
            # Spacecraft Attitude Determination and Control textbook. This equation is equivalent to Chris Robson's
            # equations 3-55 to 3-58. Note: I think Chris is missing a second cosine in the specular reflection equation
            force = -const * face.area * cos_theta * \
                    (2*(face.diff_ref_coeff/3 + face.spec_ref_coeff*cos_theta)*face.normal
                     + (1 - face.spec_ref_coeff)*sun_vec)
            net_torque += ut.cross_product_operator(face.centroid - cubesat.center_of_mass) @ force

    return net_torque


# This function is here because it could go together with the solar pressure disturbance torque function.
def solar_panel_power(sun_vec, sun_vec_inertial, satellite_vec_inertial, cubesat: CubeSat):
    watt_per_meter = a_solar_constant_2 / (np.linalg.norm(sun_vec_inertial - satellite_vec_inertial)) ** 2
    power = 0

    for face in cubesat.solar_panel_faces:
        cos_theta = face.normal @ sun_vec

        if cos_theta > 0:
            power += watt_per_meter * face.area * cos_theta * face.solar_power_efficiency

    return power


def total_magnetic(b, cubesat: CubeSat):
    return ut.cross_product_operator(cubesat.total_magnetic_moment) @ b  # both vectors must be in the body frame


def hysteresis_rod_torque(b, i, cubesat: CubeSat):
    h = b/u0
    torque = 0

    # for each hysteresis rod
    for rod in cubesat.hyst_rods:

        # propagate the magnetic field of the rod
        h_proj = h @ rod.axes_alignment  # calculate component of h along axis of hysteresis rod
        rod.propagate_and_save_magnetization(h_proj, i)

        # calculate m from b of the rod
        m = rod.axes_alignment * rod.b[i] * rod.scale_factor * rod.volume / u0

        # calculate m x B torque
        torque += ut.cross_product_operator(m) @ b

    return torque
