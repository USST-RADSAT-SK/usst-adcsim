import numpy as np
import transformations as tr


def random_dcm():
    """
    This function generates a random DCM.
    method: generate a random vector and angle of rotation (PRV attitude coordinates) then calculate the corresponding
    DCM
    :return: random DCM
    """
    e = 2*np.random.random(3) - 1
    e = e/np.linalg.norm(e)  # random unit vector
    r = np.pi*np.random.random()  # random angle between 0 and 180 (-180 to 180 would also be fine?)
    return tr.prv_to_dcm(r, e)


def cross_product_operator(vec):
    """
    Takes in a vector and outputs its 'cross product operator'.

    See Part1/3_Directional-Cosine-Matrix-_DCM_.pdf page 14

    :param vec: any 3D vector
    :return: 3x3 cross product operator
    """
    return np.array([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])


def align_z_to_nadir(pos_vec):
    # this function takes a position vector and outputs one of the infinite amount of DCM's that represent the
    # translation from the inertial frame to the body frame, such that the body frames z-axis is perfectly aligned
    # with the position vector input.
    p = pos_vec
    p = p / np.linalg.norm(p)

    r = np.random.random(3)
    r = r / np.linalg.norm(r)

    t1 = p
    t2 = cross_product_operator(p) @ r
    t2 = t2 / np.linalg.norm(t2)
    t3 = cross_product_operator(t1) @ t2
    return np.array([t2, t3, t1])


def initial_align_gravity_stabilization(pos_vec, vel_vec):
    # this function takes a position and velocity vector and outputs the DCM that represents the translation from the
    # inertial frame to the body frame, such that the body frames z-axis aligns with the position vector and the
    # body frames y-axis aligns with the cross track (i.e. the direction perpendicular to nadir and the velocity track)
    p = pos_vec
    p = p / np.linalg.norm(p)

    v = vel_vec
    v = v / np.linalg.norm(v)

    t1 = cross_product_operator(p) @ v

    t1 = t1/np.linalg.norm(t1)

    v_corrected = cross_product_operator(t1) @ p

    dcm = np.array([v_corrected, t1, p])

    return dcm


def gstime(jdut1):
    deg2rad = np.pi/180

    tut1 = (jdut1 - 2451545.0) / 36525.0

    temp = - 6.2e-6 * tut1 * tut1 * tut1 + 0.093104 * tut1 * tut1 + (876600.0 * 3600.0 + 8640184.812866) * tut1 + 67310.54841

    temp = np.fmod(temp * deg2rad / 240.0, 2*np.pi)

    if temp < 0.0:
        temp = temp + 2*np.pi

    gst = temp
    return gst


def polarm(xp, yp):
    cosxp = np.cos(xp)
    sinxp = np.sin(xp)
    cosyp = np.cos(yp)
    sinyp = np.sin(yp)

    pm = np.array([[cosxp, 0, -sinxp], [sinxp*sinyp, cosyp, cosxp*sinyp], [sinxp*cosyp, -sinyp, cosxp*cosyp]])
    return pm


def teme_to_ecef(rteme, vteme, ateme, ttt, jdut1, lod=0, xp=0, yp=0, eqeterms=1):
    deg2rad = np.pi / 180.0

    # find gmst
    gmst = gstime(jdut1)

    # find omega fro nutation theory
    omega = 125.04452222 + (-6962890.5390 * ttt + 7.455 * ttt * ttt + 0.008 * ttt * ttt * ttt) / 3600.0
    omega = np.fmod(omega, 360.0) * deg2rad

    # find mean ast
    # teme does not include the geometric terms here after 1997, kinematic terms apply
    if (jdut1 > 2450449.5) and (eqeterms > 0):
        gmstg = gmst + 0.00264 * np.pi / (3600 * 180) * np.sin(omega) + 0.000063 * np.pi / (3600 * 180) * np.sin(2.0 * omega)
    else:
        gmstg = gmst

    gmstg = np.fmod(gmstg, 2 * np.pi)

    st = np.array([[np.cos(gmstg), -np.sin(gmstg), 0], [np.sin(gmstg), np.cos(gmstg), 0], [0, 0, 1]])

    pm = polarm(xp, yp)

    rpef = st.T @ rteme
    recef = pm.T @ rpef

    thetasa = 7.29211514670698e-05 * (1.0 - lod/86400.0)

    omegaearth = np.array([0, 0, thetasa])  # This line could potentially cause issues in the conversion from MatLab

    temp = cross_product_operator(omegaearth) @ rpef

    vpef = st.T @ vteme - temp
    vecef = pm.T @ vpef

    aecef = pm.T @ (st.T @ ateme - cross_product_operator(omegaearth) @ temp - 2*cross_product_operator(omegaearth) @ vpef)

    return recef, vecef, aecef

