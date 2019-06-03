"""
For all the different types of attitude parametrizations (e.g. MRP's, Euler angles, DCM, PRV's) there are 'things' you
need to consider when you integrate their corresponding differential equations.

For MRP's (the one implemented here) you have to switch to "shadow MRP's" if the magnitude of the MRP vector becomes
greater than 1.

For Quaternions I believe what you have to do is re-normalize the vector each time you update it in the integration.

For DCM I believe what you have to do is make the determinant equal to 1 again. Or in other words, 'convert to the
nearest orthonormal representation'.
"""
import numpy as np


def mrp_switching(state):
    """
    Integral consideration for the Modified Rodriguez Parameter (MRP) attitude parametrization. Checks to see if the
    magnitude of the MRP vector is greater than one, if it is we 'switch' to the 'Shadow MRP set'.
    :param state: The current attitude state
    :return: The attitude state after the switching is done
    """
    if np.linalg.norm(state[0]) > 1:
        state[0] = -(state[0]) / (state[0] @ state[0])
    return state
