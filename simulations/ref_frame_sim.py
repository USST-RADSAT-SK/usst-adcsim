import numpy as np
import matplotlib.pyplot as plt
import transformations as tr
import control_laws as cl
import state_propagations as st
import integrators as it
import integral_considerations as ic
import find_reference_frame as rf


time_step = 0.01
end_time = 500
time = np.arange(0, end_time, time_step)


# declare the bodies inertia, initial attitude, initial angular velocity, control torque constants, and max torque
# limitation
inertia = np.diag([140, 100, 80])
inertia_inv = np.linalg.inv(inertia)
omega0 = np.array([0, 0, 0])
sigma0 = np.array([0.60, -0.4, 0.2])
K = 40
P = np.array([18.67, 20.67, 10.67])
max_torque = None


# create inertial sun and magnetic field vectors for attitude determination
sun_vec = np.random.random(3)
sun_vec = sun_vec/np.linalg.norm(sun_vec)
mag_vec = np.random.random(3)
mag_vec = mag_vec/np.linalg.norm(mag_vec)


# get initial reference frame
dcm_rn = np.zeros((len(time), 3, 3))
v = np.array([0.5, 0.5, 0.1])  # create a vector that represents euler angle rotation
dcm_rn[0] = tr.euler_angles_to_dcm(v, type='3-2-1')  # find dcm corresponding to euler angle rotation


# The integration

# states_br = np.zeros((len(time), 2, 3))
states_bn = np.zeros((len(time), 2, 3))
states_bn[0] = [sigma0, omega0]
controls = np.zeros((len(time), 3))
# states_br[0] = [sigma0, omega0]
dcm_br = np.zeros((len(time), 3, 3))
# dcm_br[0] = tr.mrp_to_dcm(states_br[0][0])
dcm_bn = np.zeros((len(time), 3, 3))
for i in range(len(time) - 1):
    # do attitude determination (this needs to change now that the state is defined in the reference frame)
    # sigma_estimated = ae.mrp_triad_with_noise(states_br[i][0], sun_vec, mag_vec, 0.01, 0.01)
    # sigma_estimated = ae.mrp_triad_with_noise(states_br[i][0], sun_vec, mag_vec, 0.0, 0.0)
    # sigma_estimated = states_br[i][0]

    dcm_bn[i] = tr.mrp_to_dcm(states_bn[i][0])
    dcm_rn[i] = rf.get_dcm_rn(time[i])
    dcm_br[i] = dcm_bn[i] @ dcm_rn[i].T
    sigma_br = tr.dcm_to_mrp(dcm_br[i])

    # get reference frame velocities
    omega_r = dcm_br[i] @ rf.get_omega_r(time[i])  # convert from reference frame to body frame
    omega_dot_r = dcm_br[i] @ rf.get_omega_r_dot(time[i])  # convert from reference frame to body frame
    # next step: numerically approximate these from reference frame position wrt time?

    # get control torques
    controls[i] = cl.control_torque_ref_frame(sigma_br, states_bn[i][1], omega_r, omega_dot_r, inertia, K, P, i,
                                              controls[i - 1], max_torque=max_torque)

    # propagate attitude state
    states_bn[i + 1] = it.rk4(st.state_dot_ref_frame, time_step, states_bn[i], controls[i], omega_r, inertia,
                              inertia_inv)

    # do 'tidy' up things at the end of integration (needed for many types of attitude coordinates)
    states_bn[i + 1] = ic.mrp_switching(states_bn[i + 1])

dcm_bn[i+1] = tr.mrp_to_dcm(states_bn[i+1][0])
dcm_rn[i+1] = rf.get_dcm_rn(time[i+1])
dcm_br[i+1] = dcm_bn[i+1] @ dcm_rn[i+1].T


if __name__ == "__main__":

    def _plot(data, title, ylabel):
        plt.figure()
        plt.plot(time, data)
        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel(ylabel)

    omegas = states_bn[:, 1]
    sigmas = states_bn[:, 0]

    _plot(omegas, 'angular velocity components', 'angular velocity (rad/s)')
    _plot(sigmas, 'mrp components', 'mrp component values')


    # get prv's
    def get_prvs(data):
        angle = np.zeros(len(time))
        e = np.zeros((len(time), 3))
        for i in range(len(time)):
            angle[i], e[i] = tr.dcm_to_prv(tr.mrp_to_dcm(data[i]))
        return angle, e

    # angle, e = get_prvs(sigmas)

    # The prv's are obtained and plotted here because they are an intuitive attitude coordinate system
    # and the prv angle as a function of time is the best way to visualize your attitude error.
    # _plot(angle, 'prv angle reference', 'prv angle (rad)')

    # plot the control torque
    # _plot(controls, 'control torque components', 'Torque (Nm)')

    # plot the mrp magnitude
    # _plot(np.linalg.norm(sigmas, axis=1), 'mrp magnitude', '')

    from animation import AnimateAttitude, DrawingVectors
    num = 200
    ref1 = DrawingVectors(dcm_bn[::num], 'axes', color=['C0', 'C1', 'C2'], label=['Body x', 'Body y', 'Body z'], length=4)
    ref2 = DrawingVectors(dcm_rn[::num], 'axes', color=['r', 'y', 'b'], label=['Ref x', 'Ref y', 'Ref z'], length=4)
    a = AnimateAttitude(dcm_bn[::num], draw_vector=[ref1, ref2])
    a.animate()

    plt.show()
