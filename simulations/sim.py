import numpy as np
import matplotlib.pyplot as plt
import transformations as tr
import control_laws as cl
import state_propagations as st
import integrators as it
import integral_considerations as ic
import find_reference_frame as rf
import attitude_estimations as ae


# declare the bodies inertia, initial attitude, initial angular velocity, control torque constants, and max torque
# limitation
inertia = np.diag([2*(10**-3), 8*(10**-3), 8*(10**-3)])
inertia_inv = np.linalg.inv(inertia)
omega0 = np.array([0.7, 0.2, -0.15])
sigma0 = np.array([0.60, -0.4, 0.2])
K = 7.11 * (10**-5)
P = np.array([18.67, 20.67, 30.67])*(10**-5)
max_torque = None


# create reference frame
v = np.array([0.5, 0.5, 0.1])  # create a vector that represents euler angle rotation
dcm_rn = tr.euler_angles_to_dcm(v, type='3-2-1')  # find dcm corresponding to euler angle rotation

# create inertial sun and magnetic field vectors for attitude determination
sun_vec = np.random.random(3)
sun_vec = sun_vec/np.linalg.norm(sun_vec)
mag_vec = np.random.random(3)
mag_vec = mag_vec/np.linalg.norm(mag_vec)


# The integration
time_step = 0.01
end_time = 300
time = np.arange(0, end_time, time_step)
states = np.zeros((len(time), 2, 3))
controls = np.zeros((len(time), 3))
states[0] = [sigma0, omega0]
dcm = np.zeros((len(time), 3, 3))
dcm[0] = tr.mrp_to_dcm(states[0][0])
for i in range(len(time) - 1):
    # do attitude determination
    sigma_estimated = ae.mrp_triad_with_noise(states[i][0], sun_vec, mag_vec, 0.01, 0.01)

    # get reference frame
    sigma_br = rf.get_mrp_br(dcm_rn, sigma_estimated)  # note: angular velocities need to be added to reference frame

    # get control torques
    controls[i] = cl.control_torque(sigma_br, states[i][1], inertia, K, P, i, controls[i-1], max_torque=max_torque)

    # propagate attitude state
    states[i+1] = it.rk4(st.state_dot, time_step, states[i], controls[i], inertia, inertia_inv)

    # do 'tidy' up things at the end of integration (needed for many types of attitude coordinates)
    states[i+1] = ic.mrp_switching(states[i+1])
    dcm[i+1] = tr.mrp_to_dcm(states[i+1][0])


if __name__ == "__main__":
    omegas = states[:, 1]
    sigmas = states[:, 0]


    def _plot(data, title, ylabel):
        plt.figure()
        plt.plot(time, data)
        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel(ylabel)


    _plot(omegas, 'angular velocity components', 'angular velocity (rad/s)')
    _plot(sigmas, 'mrp components', 'mrp component values')

    # get prv's
    def get_prvs(data):
        angle = np.zeros(len(time))
        e = np.zeros((len(time), 3))
        for i in range(len(time)):
            angle[i], e[i] = tr.dcm_to_prv(tr.mrp_to_dcm(data[i]))
        return angle, e


    angle, e = get_prvs(sigmas)

    # The prv's are obtained and plotted here because they are an intuitive attitude coordinate system
    # and the prv angle as a function of time is the best way to visualize your attitude error.
    _plot(angle, 'prv angle reference', 'prv angle (rad)')

    # plot the control torque
    _plot(controls, 'control torque components', 'Torque (Nm)')

    # plot the mrp magnitude
    _plot(np.linalg.norm(sigmas, axis=1), 'mrp magnitude', '')

    from animation import AnimateAttitude

    a = AnimateAttitude(dcm[::100], dcm_rn)
    a.animate()

    plt.show()
