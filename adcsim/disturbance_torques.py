"""
This file contains functions that are used to calculate torques on the satellite.

The disturbance torques from:
    - Solar Pressure
    - Aerodynamic pressure
    - Gravity gradient

Torques from:
    - total Magnetization of the CubeSat
    - hysteresis rods

There is also a function in here that calculates to solar power in Watts for given conditions. This is here because of
how similar the calculation is to the Solar Pressure torque calculation.

"""


from adcsim import util as ut
import numpy as np
from adcsim.CubeSat_model import CubeSat
from adcsim.containers import AttitudeData, OrbitData
from adcsim.transformations import mrp_to_dcm
from adcsim.util import inertial_to_orbit_frame


class DisturbanceTorques(object):
    _a_solar_constant = (3.823 * 10 ** 26) / (3 * (10 ** 8)) / 4 / np.pi
    _a_solar_constant_2 = (3.823 * 10**26) / 4 / np.pi
    _a_gravity_gradient_constant = 3 * 3.986004418 * (10**14)
    _a_earth_rotational_constant = 0.000072921158553
    _u0 = 4 * np.pi * 10**-7
    # Note: it appears that ut.cross_product_operator is a significant amount faster than using np.cross()

    def __init__(self, gravity=False, aerodynamic=False, solar=False, magnetic=False, hysteresis=False):
        self._include_torques = dict(gravity=gravity, aerodynamic=aerodynamic, solar=solar, magnetic=magnetic, hysteresis=hysteresis)
        self._previous_torques = {key: np.zeros(3) for key in self._include_torques}
        self.propagate_hysteresis = False
        self.save_hysteresis = False
        self.save_torques = False

    def torque(self, time: float, state: np.ndarray, attitude: AttitudeData, orbit: OrbitData, cubesat: CubeSat):
        # interpolate all the pre-calculated data
        attitude.interp_orbit_data(orbit, time, save=self.save_torques)

        if self.save_torques:
            c = attitude.save
            self.save_torques = False
        else:
            c = attitude.temp


        c.controls = np.zeros(3)

        c.dcm_bn = mrp_to_dcm(state[0])
        c.dcm_on = inertial_to_orbit_frame(c.positions, c.velocities)
        c.dcm_bo = c.dcm_bn @ c.dcm_on.T

        R0 = np.linalg.norm(c.positions)
        c.nadir = -c.positions / R0

        c.mag_field_body = (c.dcm_bn @ c.mag_field) * 1e-9  # body frame, units T

        # the request for propagation should come at the beginning of each step
        if self.propagate_hysteresis:
            h = c.mag_field_body / self._u0
            for rod in cubesat.hyst_rods:
                h_proj = h @ rod.axes_alignment
                # the request to save should come in every X steps
                if self.save_hysteresis:
                    rod.propagate_and_save_magnetization(h_proj)
                else:
                    rod.propagate_magnetization(h_proj)
            self.save_hysteresis = False
            self.propagate_hysteresis = False

        if self._include_torques['gravity']:
            ue = c.dcm_bn @ c.nadir
            c.gravityd = self.gravity_gradient(ue, R0, cubesat)
            c.controls += c.gravityd
        if self._include_torques['aerodynamic']:
            vel_body = c.dcm_bn @ self.get_air_velocity(c.velocities, c.positions)
            c.aerod = self.aerodynamic_torque(vel_body, c.density, cubesat)
            c.controls += c.aerod
        if self._include_torques['solar']:
            sun_vec_norm = c.sun_vec / np.linalg.norm(c.sun_vec)
            c.sun_vec_body = c.dcm_bn @ sun_vec_norm
            theta = np.arcsin(6.378e6 / (6.378e6 + c.alts))
            angle_btw = np.arccos(c.nadir @ sun_vec_norm)
            if angle_btw < theta:
                c.is_eclipse = 1
            else:
                c.solard = self.solar_pressure(c.sun_vec_body, c.sun_vec, c.positions, cubesat)
                c.controls += c.solard
        if self._include_torques['magnetic']:
            c.magneticd = self.total_magnetic(c.mag_field_body, cubesat)
            c.controls += c.magneticd
        if self._include_torques['hysteresis']:
            c.hyst_rod = self.hysteresis_rod_torque_peek(c.mag_field_body, cubesat)
            c.controls += c.hyst_rod

        return c.controls






    def gravity_gradient(self, ue, r0, cubesat: CubeSat):
        """
        Calculates the gravity gradient disturbance torque
        :param ue: Unit vector towards nadir in the cubesat body frame
        :param r0: distance to the center of the earth
        :param cubesat: CubeSat model
        :return: np.ndarray. gravity gradient disturbance torque in the cubesat body frame
        """
        return (self._a_gravity_gradient_constant/(r0**3)) * (ut.cross_product_operator(ue) @ cubesat.inertia @ ue)


    def get_air_velocity(self, vel_inertial, pos_inertial):
        """
        Calculates the velocity the spacecraft is moving relative to air

        this would be equivalent to:
        dcm @ (vel_inertial - ut.cross_product_operator(a_earth_rotational_constant*np.array([0, 0, 1])) @ pos_inertial)
        equation 3.160 and 3.161 in Landis Markley's Fundamentals of Spacecraft Attitude Determination and Control
        textbook

        :param vel_inertial: inertial velocity vector
        :param pos_inertial: inertial position vector
        :return: np.ndarray. velocity of spacecraft relative to air
        """

        return np.array([vel_inertial[0] + self._a_earth_rotational_constant*pos_inertial[1],
                         vel_inertial[1] - self._a_earth_rotational_constant*pos_inertial[0],
                         vel_inertial[2]])


    def aerodynamic_torque(self, v, rho, cubesat: CubeSat):
        """
        Calculates the aerodynamic disturbance torque
        :param v: velocity of spacecraft relative to air in body frame in m/s
        :param rho: atmospheric density in kg/m^3
        :param cubesat: CubeSat model
        :return: np.ndarray. aerodynamic disturbance torque in the cubesat body frame
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


    def solar_pressure(self, sun_vec, sun_vec_inertial, satellite_vec_inertial, cubesat: CubeSat):
        """
        Calculate the solar pressure torque on the cubesat.
        :param sun_vec: sun unit vector in body frame
        :param sun_vec_inertial: sun vector in inertial frame
        :param satellite_vec_inertial: satellite position vector in inertial frame
        :param cubesat: CubeSat model
        """

        # calculate the solar irradiance using equation 3-53 in Chris Robson's thesis
        const = self._a_solar_constant / (np.linalg.norm(sun_vec_inertial - satellite_vec_inertial)) ** 2

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
    def solar_panel_power(self, sun_vec, sun_vec_inertial, satellite_vec_inertial, cubesat: CubeSat):
        """
        This function calculates the current solar panel power output
        :param sun_vec: sun unit vector in body frame
        :param sun_vec_inertial: sun vector in inertial frame
        :param satellite_vec_inertial: satellite position vector in inertial frame
        :param cubesat: CubeSat model
        :return: float. Current solar panel power output
        """
        watt_per_meter = self._a_solar_constant_2 / (np.linalg.norm(sun_vec_inertial - satellite_vec_inertial)) ** 2
        power = 0

        for face in cubesat.solar_panel_faces:
            cos_theta = face.normal @ sun_vec

            if cos_theta > 0:
                power += watt_per_meter * face.area * cos_theta * face.solar_power_efficiency

        return power


    def total_magnetic(self, b, cubesat: CubeSat):
        """
        This function calculates the total magnetic torque exhibited on the satellite from magnetic moments defined in
        the CubeSat model
        :param b: external magnetic field in the body frame (Units: Telsa)
        :param cubesat: CubeSat model
        :return: np.ndarray. magnetic torque in the cubesat body frame
        """
        return ut.cross_product_operator(cubesat.total_magnetic_moment) @ b  # both vectors must be in the body frame

    def _hysteresis_rod_torque(self, b, cubesat: CubeSat, rod_function: str):
        h = b / self._u0
        torque = 0

        # for each hysteresis rod
        for rod in cubesat.hyst_rods:

            # propagate the magnetic field of the rod
            h_proj = h @ rod.axes_alignment  # calculate component of h along axis of hysteresis rod
            if rod_function == 'propagate_magnetization':
                b_current = rod.propagate_magnetization(h_proj)
            elif rod_function == 'propagate_and_save_magnetization':
                b_current = rod.propagate_and_save_magnetization(h_proj)
            elif rod_function == 'peek_magnetization':
                b_current = rod.peek_magnetization(h_proj)
            else:
                raise ValueError(f'Unknown rod function {rod_function}')

            # calculate m from b of the rod
            m = rod.axes_alignment * b_current * rod.volume / self._u0

            # calculate m x B torque
            torque += ut.cross_product_operator(m) @ b

        return torque

    def hysteresis_rod_torque(self, b, cubesat: CubeSat):
        """
        This function calculates the torque exerted on the spacecraft from any hysteresis rods in the cubesat model
        :param b: external magnetic field in the body frame (Units: Telsa)
        :param cubesat: CubeSat model
        :return: np.ndarray. magnetic torque from hysteresis rods in the cubesat body frame
        """
        return self._hysteresis_rod_torque(b, cubesat, 'propagate_magnetization')


    # if you want to save the hysteresis data
    def hysteresis_rod_torque_save(self, b, cubesat: CubeSat):
        """
        This function is the same as above but should be used if you want to save the hystersis rods data.
        :param b: external magnetic field in the body frame (Units: Telsa)
        :param i: current index of integration
        :param cubesat: CubeSat model
        :return:
        """
        return self._hysteresis_rod_torque(b, cubesat, 'propagate_and_save_magnetization')

    def hysteresis_rod_torque_peek(self, b, cubesat: CubeSat):
        """
        This function is the same as above but should be used if don't want to propagate the magnetization.
        :param b: external magnetic field in the body frame (Units: Telsa)
        :param i: current index of integration
        :param cubesat: CubeSat model
        :return:
        """
        return self._hysteresis_rod_torque(b, cubesat, 'peek_magnetization')


