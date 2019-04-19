import numpy as np


def mrp_switching(state):
    if np.linalg.norm(state[0]) > 1:
        print(state[0])
        state[0] = -(state[0]) / (state[0] @ state[0])
    return state
