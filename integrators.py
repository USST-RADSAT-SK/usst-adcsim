def rk4(fn, time_step, state, control, inertia, inertia_inv):
    k1 = time_step*fn(state, control, inertia, inertia_inv)
    k2 = time_step*fn(state + k1/2, control, inertia, inertia_inv)
    k3 = time_step*fn(state + k2/2, control, inertia, inertia_inv)
    k4 = time_step*fn(state + k3, control, inertia, inertia_inv)
    return state + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
