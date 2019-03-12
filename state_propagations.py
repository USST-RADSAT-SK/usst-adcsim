import numpy as np
import util as ut


def state_dot(state, control, inertia, inertia_inv):
    a = (1 - state[0] @ state[0]) * np.identity(3) + 2 * ut.cross_product_operator(state[0]) + 2 * np.outer(state[0], state[0])
    return np.array([(1/4) * a  @ state[1], inertia_inv @ ((-ut.cross_product_operator(state[1]) @ inertia @ state[1]) + control)])
