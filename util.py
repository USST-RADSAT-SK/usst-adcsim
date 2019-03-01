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
