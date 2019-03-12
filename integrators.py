def rk4(fn, time_step, state, *args):
    k1 = time_step*fn(state, *args)
    k2 = time_step*fn(state + k1/2, *args)
    k3 = time_step*fn(state + k2/2, *args)
    k4 = time_step*fn(state + k3, *args)
    return state + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
