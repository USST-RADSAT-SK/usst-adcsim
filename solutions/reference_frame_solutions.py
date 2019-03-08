import numpy as np
import matplotlib.pyplot as plt
import util as ut
import transformations as tr


# Solution Q1 (this is one of many solutions)
v = np.array([1, 0.5, 0.1])  # create a vector that represents euler angle rotation
dcm_rn = tr.euler_angles_to_dcm(v, type='3-2-1')  # find dcm corresponding to euler angle rotation


# Solution Q2
# note: this is how the for loop in sim.py could potentially be altered
time_step = 0.01
end_time = 300
time = np.arange(0, end_time, time_step)
sigmas = np.zeros((len(time), 3))
sigmas_br = np.zeros((len(time), 3))
omegas = np.zeros((len(time), 3))
controls = np.zeros((len(time), 3))
sigmas[0] = sigma0
omegas[0] = omega0
dcm_br = tr.mrp_to_dcm(sigmas[0]) @ dcm_rn.T
sigmas_br[0] = tr.dcm_to_mrp(dcm_br)
for i in range(len(time) - 1):
    # note: comment out the next line to run the simulation in the absence of control torques
    controls[i] = control_torque(sigmas_br[i], omegas[i])

    # if the torque is greater than the max torque, then truncate it to the max torque (along each axis)
    # controls[i][abs(controls[i]) > max_torque] = max_torque*np.sign(controls[i])[abs(controls[i]) > max_torque]

    omegas[i+1] = omegas[i] + time_step*omega_dot(omegas[i], controls[i])
    sigmas[i+1] = sigmas[i] + time_step*sigma_dot(sigmas[i], omegas[i])

    dcm_br = tr.mrp_to_dcm(sigmas[i+1]) @ dcm_rn.T
    sigmas_br[i+1] = tr.dcm_to_mrp(dcm_br)

    # switch mrp's if needed
    if np.linalg.norm(sigmas[i+1]) > 1:
        sigmas[i+1] = -sigmas[i+1]


