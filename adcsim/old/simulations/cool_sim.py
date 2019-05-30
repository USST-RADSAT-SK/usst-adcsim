import numpy as np
import matplotlib.pyplot as plt
from adcsim import integrators as it, transformations as tr, state_propagations as st, integral_considerations as ic

# declare the bodies inertia, initial attitude, initial angular velocity, control torque constants, and max torque
# limitation
inertia = np.diag([2*(10**-3), 5*(10**-3), 8*(10**-3)])
inertia_inv = np.linalg.inv(inertia)
omega0 = np.array([2, 3, 1])
sigma0 = np.array([0.60, -0.4, 0.2])

# The integration
time_step = 0.01
end_time = 100
time = np.arange(0, end_time, time_step)
states = np.zeros((len(time), 2, 3))
controls = np.zeros((len(time), 3))
states[0] = [sigma0, omega0]
dcm = np.zeros((len(time), 3, 3))
dcm[0] = tr.mrp_to_dcm(states[0][0])
for i in range(len(time) - 1):
    if i == 500:
        controls[i] = [0.005, 0.005, 0.005]

    # propagate attitude state
    states[i+1] = it.rk4(st.state_dot_mrp, time_step, states[i], controls[i], inertia, inertia_inv)

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


    #_plot(omegas, 'angular velocity components', 'angular velocity (rad/s)')
    #_plot(sigmas, 'mrp components', 'mrp component values')


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

    from adcsim.animation import AnimateAttitude, DrawingVectors, AdditionalPlots
    from adcsim.CubeSat_model_examples import CubeSatSolarPressureEx1
    num = 20
    body = DrawingVectors(dcm[::num], 'axes', ['C0', 'C1', 'C2'], ['Body x', 'Body y', 'Body z'], 4)
    plot1 = AdditionalPlots(time[::num], omegas[::num], title='Angular Velocity', ylabel='Rad/s')
    a = AnimateAttitude(dcm[::num], draw_vector=body, additional_plots=plot1, cubesat_model=CubeSatSolarPressureEx1())
    a.animate_and_plot()

    plt.show()
