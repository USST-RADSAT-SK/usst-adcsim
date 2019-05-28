import numpy as np
import matplotlib.pyplot as plt
import adcsim.util as ut
from adcsim.CubeSat_model_examples import CubeSatSolarPressureEx1
import adcsim.transformations as tr
from adcsim.animation import AnimateAttitude, DrawingVectors

# Compare this simulation too
# https://www.youtube.com/watch?v=1n-HMSCDYtM

# 1.0 Define differential equations
inertia = np.diag([140, 100, 80])


def omega_dot(omega):
    return np.linalg.inv(inertia) @ (-ut.cross_product_operator(omega) @ inertia @ omega)


def sigma_dot(sigma, omega):
    A = (1-sigma @ sigma) * np.identity(3) + 2 * ut.cross_product_operator(sigma) + 2 * np.outer(sigma, sigma)
    return (1/4) * A  @ omega


# 2.0 Integrate differential equations with eulers method
time_step = 0.01
time = np.arange(0, 500, time_step)
sigmas = np.zeros((len(time), 3))
omegas = np.zeros((len(time), 3))
sigmas[0] = np.array([0.60, -0.4, 0.2])
omegas[0] = np.array([0, 1, 0])
for i in range(len(time) - 1):
    omegas[i+1] = omegas[i] + time_step*omega_dot(omegas[i])
    if i == 500:
        omegas[i+1] += np.array([0.005, 0.005, 0.005])
    sigmas[i+1] = sigmas[i] + time_step*sigma_dot(sigmas[i], omegas[i])


# 3.0 plot angular velocites
plt.figure()
plt.plot(time, omegas)
plt.title('angular velocity components')

# 3.1 plot attitude representation components
plt.figure()
plt.plot(time, sigmas)
plt.title('attitude representation components')


# 4.0 Animate this result so we can actually see what is going on
cubesat = CubeSatSolarPressureEx1()
# need the 3x3 matrix representation of attitude (Directional Cosine Matrix (DCM)) for the animation routine.
dcm = np.zeros((len(time), 3, 3))
for i in range(len(time)):
    dcm[i] = tr.mrp_to_dcm(sigmas[i])

ref = DrawingVectors(dcm[::100], 'axes', color=['C0', 'C1', 'C2'], label=['Body x', 'Body y', 'Body z'], length=0.2)
a = AnimateAttitude(dcm[::100], draw_vector=ref, cubesat_model=cubesat)
a.animate()
