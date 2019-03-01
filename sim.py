import numpy as np
import matplotlib.pyplot as plt
import util as ut
import transformations as tr


# declare the bodies inertia, initial attitude, initial angular velocity, and control torque constants
inertia = np.diag([140, 100, 80])
omega0 = np.array([0.1, 0.01, -0.05])
sigma0 = np.array([0.60, -0.4, 0.2])
K = 7.11
P = np.array([16.7, 6.7, 15.67])


# equation name: Euler's rotational equation of motion
# taken from Part2/1_Kinetics.Mod.1.Slides.pdf page 15.
def omega_dot(omega, control):
    return np.linalg.inv(inertia) @ ((-ut.cross_product_operator(omega) @ inertia @ omega) + control)


# "differential equation" for the MRP attitude coordinates
# taken from Part1/8_Modified-Rodrigues-Parameters-_MRP_.pdf page 99.
def sigma_dot(sigma, omega):
    a = (1-sigma @ sigma) * np.identity(3) + 2 * ut.cross_product_operator(sigma) + 2 * np.outer(sigma, sigma)
    return (1/4) * a  @ omega


# control torque equations taken from Part3/3_Control.Mod.3.Slides.pdf page 27.
# note: terms in the equation are missing here for this implementation
def control_torque(sigma, omega):
    return -K * sigma - P * omega + ut.cross_product_operator(omega) @ inertia @ omega


# The integration (with Euler's method)
time_step = 0.01
time = np.arange(0, 150, time_step)
sigmas = np.zeros((len(time), 3))
omegas = np.zeros((len(time), 3))
controls = np.zeros((len(time), 3))
sigmas[0] = sigma0
omegas[0] = omega0
for i in range(len(time) - 1):
    # note: comment out the next line to run the simulation in the absence of control torques
    controls[i] = control_torque(sigmas[i], omegas[i])
    omegas[i+1] = omegas[i] + time_step*omega_dot(omegas[i], controls[i])
    sigmas[i+1] = sigmas[i] + time_step*sigma_dot(sigmas[i], omegas[i])


# function to dry up plots
def _title(title, ylabel):
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel(ylabel)


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
plt.show()
