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
