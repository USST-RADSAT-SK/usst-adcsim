"""
Implementations of the rk4 numeral integration algorithm. These can be used to propagate attitude states.

see https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods for rk4 implementation
"""


def rk4(fn, time, state, time_step, *args):
    """
    Performs rk4 numerical integration given the differential equation defined in 'fn' variable.

    The use this function is set up for is to pass the functions in state_propagation.py as the fn argument
    :param fn: function that evaluates differential equation.
    :param time_step: time step of integration
    :param state: current state of the variables being integrated
    :param args: everything (other than the state, which is taken care of automatically) that you would want to pass to
    'fn'
    :return: The new state
    """
    k1 = time_step*fn(time, state, *args)
    k2 = time_step*fn(time + 0.5 * time_step, state + 0.5 * k1, *args)
    k3 = time_step*fn(time + 0.5 * time_step, state + 0.5 * k2, *args)
    k4 = time_step*fn(time + time_step, state + k3, *args)
    return state + (1/6)*(k1 + 2*k2 + 2*k3 + k4)


def rk4_general(fn, time_step, t, y, *args):
    """
    Similar to rk4 function above, but can be used if 'fn' depends on the independent variable (labeled time here)
    as well as the dependant variable. Right now this is used for propagation of the hysteresis rod magnetization state.
    :param fn: function that evaluates differential equation.
    :param time_step: time step of integration
    :param t: current value of dependant variable (usually time)
    :param y: current value of variable being integrated
    :param args: everything (other than t and y, which is taken care of automatically) that you would want to pass to
    'fn'
    :return: The new state
    """
    k1 = time_step*fn(t, y, *args)
    k2 = time_step*fn(t + time_step/2, y + k1/2, *args)
    k3 = time_step*fn(t + time_step/2, y + k2/2, *args)
    k4 = time_step*fn(t + time_step, y + k3, *args)
    return y + (1/6)*(k1 + 2*k2 + 2*k3 + k4)

