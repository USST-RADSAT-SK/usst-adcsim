import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# declare things used for testing
time_step = 0.01
end_time = 300
time = np.arange(0, end_time, time_step)


# Solution Q1
def x2(x):
    return 2*x


def cos(x):
    return np.cos(x)


# Solution Q2
def eulers_int(fn, x):
    y_calc = np.zeros(len(x))
    y_calc[0] = 1
    for i in range(len(x) - 1):
        y_calc[i + 1] = y_calc[i] + time_step*fn(x[i])
    return y_calc


# Solution Q3 (note: y_calc should be t^2 + 1 and sin(t) + 1)
y_calc = eulers_int(x2, time)
plt.figure()
plt.plot(time, y_calc)
y_calc = eulers_int(cos, time)
plt.figure()
plt.plot(time, y_calc)


# Solution Q4
def rk4_int(fn, x):
    y_calc = np.zeros(len(x))
    y_calc[0] = 1
    for i in range(len(x) - 1):
        k1 = time_step*fn(x[i])
        k2 = time_step*fn(x[i] + time_step/2)
        k3 = k2
        k4 = time_step*fn(x[i] + time_step)
        y_calc[i + 1] = y_calc[i] + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    return y_calc


# test
y_calc = rk4_int(x2, time)
plt.figure()
plt.plot(time, y_calc)
y_calc = rk4_int(cos, time)
plt.figure()
plt.plot(time, y_calc)


# Solution Q5
def scipy_int(fn, x):
    inte = integrate.ode(fn).set_integrator("dopri5")
    inte.set_initial_value(1)
    y_calc = np.zeros(len(x))
    for i in range(len(x)):
        y_calc[i] = inte.integrate(x[i])
    return y_calc


# test
y_calc = scipy_int(x2, time)
plt.figure()
plt.plot(time, y_calc)
y_calc = scipy_int(cos, time)
plt.figure()
plt.plot(time, y_calc)


# Solution Q6
def double(x):
    return np.array([np.cos(x), 2*x])


def eulers_int(fn, x):
    y_calc = np.zeros((len(x), 2))
    y_calc[0] = [1, 1]
    for i in range(len(x) - 1):
        y_calc[i + 1] = y_calc[i] + time_step*fn(x[i])
    return y_calc


def rk4_int(fn, x):
    y_calc = np.zeros((len(x), 2))
    y_calc[0] = [1, 1]
    for i in range(len(x) - 1):
        k1 = time_step*fn(x[i])
        k2 = time_step*fn(x[i] + time_step/2)
        k3 = k2
        k4 = time_step*fn(x[i] + time_step)
        y_calc[i + 1] = y_calc[i] + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    return y_calc


def scipy_int(fn, x):
    inte = integrate.ode(fn).set_integrator("dopri5")
    inte.set_initial_value([1, 1])
    y_calc = np.zeros((len(x), 2))
    for i in range(len(x)):
        y_calc[i] = inte.integrate(x[i])
    return y_calc


# Solution Q7
y_calc = scipy_int(double, time)
plt.figure()
plt.plot(time, y_calc[:, 0])
plt.figure()
plt.plot(time, y_calc[:, 1])

y_calc = rk4_int(double, time)
plt.figure()
plt.plot(time, y_calc[:, 0])
plt.figure()
plt.plot(time, y_calc[:, 1])

y_calc = eulers_int(double, time)
plt.figure()
plt.plot(time, y_calc[:, 0])
plt.figure()
plt.plot(time, y_calc[:, 1])


# Solution Q8
import numpy as np
import matplotlib.pyplot as plt
import util as ut
import transformations as tr


# declare the bodies inertia, initial attitude, initial angular velocity, and control torque constants
inertia = np.diag([140, 100, 80])
inertia_inv = np.linalg.inv(inertia)
omega0 = np.array([0.7, 0.2, -0.15])
sigma0 = np.array([0.60, -0.4, 0.2])
K = 7.11
P = np.array([18.67, 2.67, 10.67])
# add ability to specific max torque in each direction
max_torque = 1


def state_dot(state, control):
    a = (1 - state[0] @ state[0]) * np.identity(3) + 2 * ut.cross_product_operator(state[0]) + 2 * np.outer(state[0], state[0])
    return np.array([(1/4) * a  @ state[1],
                     inertia_inv @ ((-ut.cross_product_operator(state[1]) @ inertia @ state[1]) + control)])


# control torque equations taken from Part3/3_Control.Mod.3.Slides.pdf page 27.
# note: terms in the equation are missing here for this implementation
def control_torque(state):
    return -K * state[0] - P * state[1] + ut.cross_product_operator(state[1]) @ inertia @ state[1]


# The integration (with Euler's method)
time_step = 0.01
end_time = 300
time = np.arange(0, end_time, time_step)
states = np.zeros((len(time), 2, 3))
controls = np.zeros((len(time), 3))
states[0] = [sigma0, omega0]
for i in range(len(time) - 1):
    # note: comment out the next line to run the simulation in the absence of control torques
    controls[i] = control_torque(states[i])

    # if the torque is greater than the max torque, then truncate it to the max torque (along each axis)
    controls[i][abs(controls[i]) > max_torque] = max_torque*np.sign(controls[i])[abs(controls[i]) > max_torque]

    states[i+1] = states[i] + time_step*state_dot(states[i], controls[i])
    # switch mrp's if needed
    if np.linalg.norm(states[i+1][0]) > 1:
        states[i+1][0] = -states[i+1][0]


# function to dry up plots
def _title(title, ylabel):
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel(ylabel)

omegas = states[:, 1]
sigmas = states[:, 0]
plt.figure()
plt.plot(time, omegas)
_title('angular velocity components', 'angular velocity (rad/s)')
plt.figure()
plt.plot(time, sigmas)
_title('mrp components', 'mrp component values')

# get prv's
angle = np.zeros(len(time))
e = np.zeros((len(time), 3))
for i in range(len(time)):
    angle[i], e[i] = tr.dcm_to_prv(tr.mrp_to_dcm(sigmas[i]))

# The prv's are obtained and plotted here because they are an intuitive attitude coordinate system
# and the prv angle as a function of time is the best way to visualize your attitude error.
plt.figure()
plt.plot(time, angle)
_title('prv angle', 'prv angle (rad)')
plt.figure()
plt.plot(time, e)
_title('prv unit vector', 'prv unit vector components')

# plot the control torque
plt.figure()
plt.plot(time, controls)
_title('control torque components', 'Torque (Nm)')

# plot the mrp magnitude
plt.figure()
plt.plot(time, np.linalg.norm(sigmas, axis=1))
_title('mrp magnitude', '')
plt.show()


# Solution Q9
"""
Within the for loop:
    k1 = time_step*state_dot(states[i], controls[i])
    k2 = time_step*state_dot(states[i] + k1/2, controls[i])
    k3 = time_step*state_dot(states[i] + k2/2, controls[i])
    k4 = time_step*state_dot(states[i] + k3, controls[i])

    states[i+1] = states[i] + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
"""
