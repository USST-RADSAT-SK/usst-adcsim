import util as ut
import numpy as np

a_solar_constant = 1366 / (3 * (10 ** 8))


def gravity_gradient(ue, r0, inertia):
    u = 3.986004418 * (10**14)
    return (3*u/(r0**3)) * (ut.cross_product_operator(ue) @ inertia @ ue)


def aerodynamic_torque(v, rho):
    """
    v: velocity of spacecraft relative to air in body frame in m/s
    rho: atmospheric density in kg/m^3
    """
    # This function uses the equations for the free molecular flow dynamics
    # (i.e. neglecting the affect of reemitted particles on the incident stream)

    vm = np.linalg.norm(v)
    ev = v * (1.0 / vm)

    # crude model of cubesat faces including antennas, doesn't account for shadowing
    M = np.zeros((6, 30))
    # indices 0-2: vector from center of cubesat to center of the face
    # indices 3-5: outward normal vector with magnitude = area

    # main faces
    M[:, 0] = np.array([50., 0., 0., 20000., 0., 0.])
    M[:, 1] = np.array([0., 50., 0., 0., 20000., 0.])
    M[:, 2] = np.array([0., 0., 100., 0., 0., 10000.])
    M[:, 3] = np.array([-50., 0., 0., -20000., 0., 0.])
    M[:, 4] = np.array([0., -50., 0., 0., -20000., 0.])
    M[:, 5] = np.array([0., 0., -100., 0., 0., -10000.])

    # +x antenna
    M[:, 6] = np.array([50. + 17.85, 22., 97. + 2., 0., 0., 214.2])
    M[:, 7] = np.array([50. + 17.85, 22., 97. - 2., 0., 0., -214.2])
    M[:, 8] = np.array([50. + 17.85, 22. + 3., 97., 0., 142.8, 0.])
    M[:, 9] = np.array([50. + 17.85, 22. - 3., 97., 0., -142.8, 0.])
    M[:, 10] = np.array([50. + 103.85, 22. + 1., 97., 0., 408.9, 0. ])
    M[:, 11] = np.array([50. + 103.85, 22. + 1., 97., 0., -408.9, 0. ])

    # +y antenna
    M[:, 12] = np.array([22., 50. + 17.85, 97. + 2., 0., 0., 214.2])
    M[:, 13] = np.array([22., 50. + 17.85, 97. - 2., 0., 0., -214.2])
    M[:, 14] = np.array([22. + 3, 50. + 17.85, 97., 142.8, 0., 0.])
    M[:, 15] = np.array([22. - 3, 50. + 17.85, 97., -142.8, 0., 0.])
    M[:, 16] = np.array([22. + 1., 50. + 274.85, 97., 1434.9, 0., 0. ])
    M[:, 17] = np.array([22. + 1., 50. + 274.85, 97., -1434.9, 0., 0. ])

    # -x antenna
    M[:, 18] = -np.array([50. + 17.85, 22., -97. - 2., 0., 0., 214.2])
    M[:, 19] = -np.array([50. + 17.85, 22., -97. + 2., 0., 0., -214.2])
    M[:, 20] = -np.array([50. + 17.85, 22. + 3., -97., 0., 142.8, 0.])
    M[:, 21] = -np.array([50. + 17.85, 22. - 3., -97., 0., -142.8, 0.])
    M[:, 22] = -np.array([50. + 103.85, 22. + 1., -97., 0., 408.9, 0. ])
    M[:, 23] = -np.array([50. + 103.85, 22. + 1., -97., 0., -408.9, 0. ])

    # -y antenna
    M[:, 24] = -np.array([22., 50. + 17.85, -97. - 2., 0., 0., 214.2])
    M[:, 25] = -np.array([22., 50. + 17.85, -97. + 2., 0., 0., -214.2])
    M[:, 26] = -np.array([22. + 3, 50. + 17.85, -97., 142.8, 0., 0.])
    M[:, 27] = -np.array([22. - 3, 50. + 17.85, -97., -142.8, 0., 0.])
    M[:, 28] = -np.array([22. + 1., 50. + 274.85, -97., 1434.9, 0., 0. ])
    M[:, 29] = -np.array([22. + 1., 50. + 274.85, -97., -1434.9, 0., 0. ])

    # calculate net torque
    net_torque = np.zeros(3)
    for i in range(30):
        A = np.linalg.norm(M[3:, i])  # mm^2
        en = M[3:, i] / A  # outward normal unit vector
        mu = np.dot(ev, en)  # cosine of angle between velocity and surface normal
        if mu >= 0:
            A *= 1e-6  # mm^2 -> m^2
            force = -rho * vm**2 * (0.4 * mu**2 * en + 0.8 * mu * ev) * A  # units N, see eq'n 2-2 in NASA SP-8058 Spacecraft Aerodynamic Torques

            r = M[:3, i] * 1e-3  # mm -> m
            torque = np.cross(r, force)  # units N.m

            # we could calculate the net_force here if needed for orbit propagation
            net_torque += torque

    return net_torque


def solar_pressure(sun_vec, faces):
    # TODO: absorption, and diffuse reflection (and redo this one)
    net_torque = np.zeros(3)
    for face in faces:
        h = face.area_unit_vec @ sun_vec

        if h > 0:
            net_torque += a_solar_constant * face.area * (1 + face.face.reflection_coeff) * h * \
                          (ut.cross_product_operator(face.centroid) @ -face.area_unit_vec)  # force is opposite to area

    return net_torque
