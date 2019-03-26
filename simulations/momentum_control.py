import numpy as np
import matplotlib.pyplot as plt
import transformations as tr
import state_propagations as st
import integrators as it
import integral_considerations as ic
import find_reference_frame as rf
import util as ut

time_step = 0.02
end_time = 600
time = np.arange(0, end_time, time_step)


# declare the bodies inertia, initial attitude, initial angular velocity, control torque constants, and max torque
# limitation
inertia = np.diag([2*(10**-3), 8*(10**-3), 8*(10**-3)])
inertia_inv = np.linalg.inv(inertia)
# original
# omega0 = np.array([0.7, 0.2, -0.15])
# sigma0 = np.array([0.60, -0.4, 0.2])
# new
omega0 = np.array([0, 0, 0])
sigma0 = np.array([0.60, -0.4, 0.2])
K = 10000000
max_torque = None


# create reference frame
v = np.array([-0.7, 0.2, -0.1])  # create a vector that represents euler angle rotation
dcm_rn = tr.euler_angles_to_dcm(v, type='3-2-1')  # find dcm corresponding to euler angle rotation


# declare reaction wheel inertia
mass = 0.06
r = 12*(10**-3)
h = 22*(10**-3)
same = (1/12)*mass*(3*r*r + h*h)
reaction_wheel_inertia = np.array([mass*r*r, same, same])
inertia_rw = inertia.copy()
inertia_rw[1, 1] += reaction_wheel_inertia[1]
inertia_rw[2, 2] += reaction_wheel_inertia[2]
inertia_inv_rw = np.linalg.inv(inertia_rw)

# create magnetic field vector
mag_vec_def = np.array([1, 2, 2])
mag_vec_def = mag_vec_def/np.linalg.norm(mag_vec_def) * 50 * (10**-6)

# mag_vec_def = np.load('magfields.npy')[:400]
# xp = np.linspace(0, end_time, len(mag_vec_def))
# mag_vec_magnitude = 50 * (10**-6)


# The integration
states = np.zeros((len(time), 2, 3))
real_controls = np.zeros((len(time), 3))
states[0] = [sigma0, omega0]
wheel_angular_vel = np.zeros((len(time), 3))  # initial condition for wheels
wheel_angular_vel[0] = [10, 10, 10]
wheel_angular_accel = np.zeros((len(time), 3))  # initial condition for wheels
dcm = np.zeros((len(time), 3, 3))
dcm[0] = tr.mrp_to_dcm(states[0][0])
mag = np.zeros((len(time), 3))
m = np.zeros((len(time), 3))
for i in range(len(time) - 1):
    # mag_vec = np.array([np.interp(i*0.1, xp, mag_vec_def[:, 0]), np.interp(i*0.1, xp, mag_vec_def[:, 1]), np.interp(i*0.1, xp, mag_vec_def[:, 2])])
    # mag_vec = mag_vec_magnitude*mag_vec/np.linalg.norm(mag_vec)
    # mag[i] = dcm[i] @ mag_vec
    mag[i] = dcm[i] @ mag_vec_def

    # get reference frame
    sigma_br = rf.get_mrp_br(dcm_rn, states[i][0])  # note: angular velocities need to be added to reference frame

    # get the hs variable needed for control torque calculation and state propagation
    hs = reaction_wheel_inertia[0] * (wheel_angular_vel[i])

    # get magnetic moment and wheel angular acceleration that create approx zero control torque
    m[i] = -K*(ut.cross_product_operator(mag[i]) @ hs)
    omega_dot = (states[i+1][1, 0] - states[i][1, 0])/time_step
    wheel_angular_accel[i] = (ut.cross_product_operator(m[i]) @ mag[i])/reaction_wheel_inertia[0] - omega_dot

    # get real controls corresponding to this
    real_controls[i] = ut.cross_product_operator(m[i]) @ mag[i] + reaction_wheel_inertia[0]*(wheel_angular_accel[i] + omega_dot)

    # propagate attitude state
    states[i+1] = it.rk4(st.state_dot, time_step, states[i], real_controls[i], inertia, inertia_inv)

    # get wheel angular velocity
    wheel_angular_vel[i+1] = wheel_angular_vel[i] + wheel_angular_accel[i] * time_step

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
    _plot(real_controls, 'control torque components', 'Torque (Nm)')

    # plot the mrp magnitude
    _plot(np.linalg.norm(sigmas, axis=1), 'mrp magnitude', '')

    # plot wheel speed and accelerations
    _plot(wheel_angular_vel*9.5493, 'reaction wheel angular velocities', 'rpm')
    _plot(wheel_angular_accel, 'reaction wheel angular accelerations', 'rad/s/s')
    _plot(m, 'coil magnetic moments', '(A*m^2)')

    from animation import AnimateAttitude
    num = 200
    a = AnimateAttitude(dcm[::num])
    a.animate_and_2_plots(time[::num], wheel_angular_vel[::num]*9.5493, time[::num], m[::num],
                          title1='Reaction Wheel Angular Velocities', ylabel1='rpm', title2='Magnetic moment',
                          ylabel2='A*m^2')

    plt.show()
