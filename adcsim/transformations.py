"""
This set of functions does transformations between the following attitude parameterizations:

DCM = Directional Cosine Matrix         (The mother of all attitude parameterizations)
PRV = Principal Rotation Vector         (The building block of many advanced attitude parameterizations)
Quaternions                             (Voted most popular attitude coordinates in the non-singular category)
CRP = Classical Rodrigues Parameters    (Popular coordinates for large rotations and robotics)
MRP = Modified Rodriques Parameters     (The "cool" new attitude coordinates)

"""
import numpy as np
from adcsim import util as ut


def prv_to_dcm(angle, unit_vector):
    """
    This function generates the DCM corresponding to the input PRV coordinates

    Taken from Part1/5_Principal_rotation_vector_PRV.pdf page 58

    :param angle: PRV angle
    :param unit_vector: PRV unit vector
    :return: DCM
    """
    r = angle
    e = unit_vector
    s = 1 - np.cos(r)
    dcm = [[s*e[0]**2 + np.cos(r), e[0]*e[1]*s + e[2]*np.sin(r), e[0]*e[2]*s - e[1]*np.sin(r)],
           [e[1]*e[0]*s - e[2]*np.sin(r), s*e[1]**2 + np.cos(r), e[1]*e[2]*s + e[0]*np.sin(r)],
           [e[2]*e[0]*s + e[1]*np.sin(r), e[2]*e[1]*s - e[0]*np.sin(r), s*e[2]**2 + np.cos(r)]]
    return np.array(dcm)


def dcm_to_prv(dcm):
    """
    This function generates the PRV coordinates corresponding to the DCM input

    Taken from Part1/5_Principal_rotation_vector_PRV.pdf page 58

    note: this function has a SINGULARITY
    :param dcm: DCM
    :return: PRV angle, PRV unit vector
    """
    angle = np.arccos((1/2)*(np.trace(dcm) - 1))  # this gives the 'short angle' not the long one
    e = (1/(2*np.sin(angle)))*np.array([dcm[1, 2] - dcm[2, 1], dcm[2, 0] - dcm[0, 2], dcm[0, 1] - dcm[1, 0]])
    return angle, e


def quaternions_to_dcm(b):
    """
    This function generates the DCM corresponding to the input quaternion coordinates

    Taken from Part1/6_Euler-Parameters-_Quaternions_.pdf page 67

    :param b: quaternions
    :return: DCM
    """
    dcm = [[b[0]**2 + b[1]**2 - b[2]**2 - b[3]**2, 2*(b[1]*b[2] + b[0]*b[3]), 2*(b[1]*b[3] - b[0]*b[2])],
           [2*(b[1]*b[2] - b[0]*b[3]), b[0]**2 - b[1]**2 + b[2]**2 - b[3]**2, 2*(b[2]*b[3] + b[0]*b[1])],
           [2*(b[1]*b[3] + b[0]*b[2]), 2*(b[2]*b[3] - b[0]*b[1]), b[0]**2 - b[1]**2 - b[2]**2 + b[3]**2]]
    return np.array(dcm)


def dcm_to_quaternions_bad(dcm):
    """
    This function generates the quaternion coordinates corresponding to the DCM input

    Taken from Part1/6_Euler-Parameters-_Quaternions_.pdf page 67

    note: this function has a SINGULARITY
    note: the positive value of b[0] represents the 'short angle rotation'
    :param dcm: DCM
    :return: quaternion coordinate vector
    """
    b0 = (1/2)*np.sqrt(np.trace(dcm) + 1)
    return np.array([b0, (dcm[1, 2] - dcm[2, 1])/(4*b0), (dcm[2, 0] - dcm[0, 2])/(4*b0),
                     (dcm[0, 1] - dcm[1, 0]) / (4*b0)])


def dcm_to_quaternions(dcm):
    """
    This function generates the quaternion coordinates corresponding to the DCm input.

    This is called sheppard's method, which does not contain a singularity like the 'common' way to do this.

    See Part1/6_Euler-Parameters-_Quaternions_.pdf page 68

    :param dcm: DCM
    :return: quaternion coordinate vector
    """
    trace = np.trace(dcm)
    b_2 = (1/4)*np.array([(1+trace), (1 + 2*dcm[0, 0] - trace), (1 + 2*dcm[1, 1] - trace), (1 + 2*dcm[2, 2] - trace)])
    argmax = np.argmax(b_2)
    b = np.zeros(4)
    b[argmax] = np.sqrt(b_2[argmax])

    # There is probably a cleaner way to do these checks
    if argmax == 0:
        b[1] = (dcm[1, 2] - dcm[2, 1])/(4*b[0])
        b[2] = (dcm[2, 0] - dcm[0, 2])/(4*b[0])
        b[3] = (dcm[0, 1] - dcm[1, 0])/(4*b[0])
    elif argmax == 1:
        b[0] = (dcm[1, 2] - dcm[2, 1])/(4*b[1])
        b[2] = (dcm[0, 1] + dcm[1, 0])/(4*b[1])
        b[3] = (dcm[2, 0] + dcm[0, 2])/(4*b[1])
    elif argmax == 2:
        b[0] = (dcm[2, 0] - dcm[0, 2])/(4*b[2])
        b[1] = (dcm[0, 1] + dcm[1, 0])/(4*b[2])
        b[3] = (dcm[1, 2] + dcm[2, 1])/(4*b[2])
    elif argmax == 3:
        b[0] = (dcm[0, 1] - dcm[1, 0])/(4*b[3])
        b[1] = (dcm[2, 0] + dcm[0, 2])/(4*b[3])
        b[2] = (dcm[1, 2] + dcm[2, 1])/(4*b[3])

    # last step to make sure we have the 'short rotation'
    if b[0] < 0:
        b = -b

    return b


def crp_to_dcm(q):
    """
    This function generates the DCM coordinates corresponding to the DCM input

    Taken from Part1/7_Classical-Rodrigues-Parameters-_CRP_.pdf page 82

    note: this function has a SINGULARITY
    :param q: CRP coordinate vector
    :return: DCM
    """
    s = q @ q
    return (1/(1 + s))*((1 - s)*np.identity(3) + 2*np.outer(q, q) - 2*ut.cross_product_operator(q))


def dcm_to_crp(dcm):
    """
    This function generates the CRP coordinates corresponding to the DCM input

    Taken from Part1/7_Classical-Rodrigues-Parameters-_CRP_.pdf page 78

    note: this function has a SINGULARITY
    :param dcm: DCM
    :return: CRP coordinate vector
    """
    c = np.trace(dcm) + 1
    return (1/c)*np.array([dcm[1, 2] - dcm[2, 1], dcm[2, 0] - dcm[0, 2], dcm[0, 1] - dcm[1, 0]])


def mrp_to_dcm(sigma):
    """
    This function generates the DCM coordinates corresponding to the MRP input

    Taken from Part1/8_Modified-Rodrigues-Parameters-_MRP_.pdf page 97

    :param sigma: MRP coordinate vector
    :return: DCM
    """
    sigma_cross = ut.cross_product_operator(sigma)
    s = sigma @ sigma
    return np.identity(3) + (8*(sigma_cross @ sigma_cross) - 4*(1 - s)*sigma_cross)/(1 + s)**2


def dcm_to_mrp(dcm):
    """
    This function generates the MRP coordinates corresponding to the DCM input

    Taken from Part1/8_Modified-Rodrigues-Parameters-_MRP_.pdf page 92

    :param dcm: DCM
    :return: MRP coordinate vector
    """
    c = np.sqrt(np.trace(dcm) + 1)
    return (1/(c*(c + 2))) * np.array([dcm[1, 2] - dcm[2, 1], dcm[2, 0] - dcm[0, 2], dcm[0, 1] - dcm[1, 0]])


def euler_angles_to_dcm(vec, type='3-2-1'):
    """
    This function generates the DCM coordinates corresponding to the euler angles of the specified type.


    :param vec: euler angles
    :param type: type of euler angle rotation (e.g. '3-2-1', '3-1-3', etc.). 12 total types
    :return: DCM
    """
    def m(angle, num):
        if num == 3:
            return np.array([[np.cos(angle), np.sin(angle), 0], [-np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        if num == 2:
            return np.array([[np.cos(angle), 0, -np.sin(angle)], [0, 1, 0], [np.sin(angle), 0, np.cos(angle)]])
        if num == 1:
            return np.array([[1, 0, 0], [0, np.cos(angle), np.sin(angle)], [0, -np.sin(angle), np.cos(angle)]])

    parse = np.array([int(i) for i in type.split('-')])

    matrices = np.zeros((3, 3, 3))
    for i, da in enumerate(parse):
        matrices[i] = m(vec[i], da)

    return matrices[2] @ matrices[1] @ matrices[0]

