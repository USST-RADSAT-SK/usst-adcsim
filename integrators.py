# see https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods for rk4 implementation


def rk4(fn, time_step, state, *args):
    # *args is everything else that you would want to pass to 'fn'
    k1 = time_step*fn(state, *args)
    k2 = time_step*fn(state + k1/2, *args)
    k3 = time_step*fn(state + k2/2, *args)
    k4 = time_step*fn(state + k3, *args)
    return state + (1/6)*(k1 + 2*k2 + 2*k3 + k4)


def rk4_general(fn, time_step, t, y, *args):
    # *args is everything else that you would want to pass to 'fn'
    k1 = time_step*fn(t, y, *args)
    k2 = time_step*fn(t + time_step/2, y + k1/2, *args)
    k3 = time_step*fn(t + time_step/2, y + k2/2, *args)
    k4 = time_step*fn(t + time_step, y + k3, *args)
    return y + (1/6)*(k1 + 2*k2 + 2*k3 + k4)

