import numpy as np
import matplotlib.pyplot as plt
import util as ut
import transformations as tr


def _title(title, ylabel):
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel(ylabel)


inertia = np.diag([140, 100, 80])
omega = np.array([0.1, 0.01, -0.05])
sigma = np.array([0.60, -0.4, 0.2])
K = 7.11
P = np.array([16.7, 6.7, 15.67])


def omega_dot(omega, control):
    return np.linalg.inv(inertia) @ (-ut.cross_product_operator(omega) @ inertia @ omega) + control


def sigma_dot(sigma, omega):
    A = (1-sigma @ sigma) * np.identity(3) + 2 * ut.cross_product_operator(sigma) + 2 * np.outer(sigma, sigma)
    return (1/4) * A  @ omega


def control_torque(sigma, omega):
    return -K * sigma - P * omega + ut.cross_product_operator(omega) @ inertia @ omega


time = np.arange(0, 75, 0.01)
sigmas = np.zeros((len(time), 3))
omegas = np.zeros((len(time), 3))
controls = np.zeros((len(time), 3))
sigmas[0] = sigma
omegas[0] = omega
for i in range(len(time) - 1):
    controls[i] = control_torque(sigmas[i], omegas[i])
    omegas[i+1] = omegas[i] + 0.01*omega_dot(omegas[i], controls[i])
    sigmas[i+1] = sigmas[i] + 0.01*sigma_dot(sigmas[i], omegas[i])


plt.figure()
plt.plot(time, omegas)
_title('angular velocity components', 'angular velocity (rad/s)')
plt.figure()
plt.plot(time, sigmas)
_title('mrp components', 'mrp component values')
plt.show()

# get prv's
angle = np.zeros(len(time))
e = np.zeros((len(time), 3))
for i in range(len(time)):
    angle[i], e[i] = tr.dcm_to_prv(tr.mrp_to_dcm(sigmas[i]))

plt.figure()
plt.plot(time, angle)
_title('prv angle', 'prv angle (rad)')
plt.figure()
plt.plot(time, e)
_title('prv unit vector', 'prv unit vector components')
plt.show()
