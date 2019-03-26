import numpy as np
import matplotlib.pyplot as plt
import transformations as tr
import control_laws as cl
import state_propagations as st
import integrators as it
import integral_considerations as ic
import util as ut


# declare the bodies inertia, initial attitude, initial angular velocity, control torque constants, and max torque
# limitation
inertia = np.diag([2*(10**-3), 8*(10**-3), 8*(10**-3)])
inertia_inv = np.linalg.inv(inertia)
omega0 = np.array([0.1, 0.1, 0.1])
sigma0 = np.array([0.20, -0.4, 0.2])
K = 10000
max_torque = None


# The integration
time_step = 0.1
end_time = 4000
time = np.arange(0, end_time, time_step)
states = np.zeros((len(time), 2, 3))
controls = np.zeros((len(time), 3))
states[0] = [sigma0, omega0]
dcm = np.zeros((len(time), 3, 3))
dcm[0] = tr.mrp_to_dcm(states[0][0])
mag = np.zeros((len(time), 3))
m = np.zeros((len(time), 3))


# create magnetic field vector
mag_vec_def = np.load('C:/Users/CurtisDesktop/PycharmProjects/U-of-Colorado_course/magfields.npy')[:400]
xp = np.arange(0, end_time, 10)
mag_vec_magnitude = 50 * (10**-6)


mag_vec = np.zeros((len(time), 3))
for i in range(len(time) - 1):
    # get magnetometer measurement
    # mag_vec[i+1] = dcm_mag_vec @ mag_vec[i]
    mag_vec[i] = [np.interp(i*0.1, xp, mag_vec_def[:, 0]), np.interp(i*0.1, xp, mag_vec_def[:, 1]), np.interp(i*0.1, xp, mag_vec_def[:, 2])]
    mag_vec[i] = mag_vec_magnitude*mag_vec[i]/np.linalg.norm(mag_vec[i])
    mag[i] = dcm[i] @ mag_vec[i]

    # get control torques
    m[i] = cl.b_dot(mag[i], mag[i-1], K, i, m[i-1], time_step*1)
    controls[i] = ut.cross_product_operator(m[i]) @ mag[i]

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
    _plot(controls, 'control torque components', 'Torque (Nm)')
    _plot(m, 'coil magnetic moments', '(A*m^2)')

    from animation import AnimateAttitude
    num = 200
    a = AnimateAttitude(dcm[::num], draw_vector=mag_vec[::num])
    a.animate_and_plot(time[::num], m[::num], 'Magnetic moment', 'A*m^2', draw_vec_type='double')

    plt.show()
