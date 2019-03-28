import numpy as np
import matplotlib.pyplot as plt
import transformations as tr
import state_propagations as st
import integrators as it
import integral_considerations as ic
import disturbance_torques as dt
import util as ut
from skyfield.api import load, EarthSatellite

time_step = 10
end_time = 100000
time = np.arange(0, end_time, time_step)

# declare the bodies inertia, initial attitude, initial angular velocity, control torque constants, and max torque
# limitation
inertia = np.diag([2*(10**-2), 4*(10**-2), 5*(10**-3)])
inertia_inv = np.linalg.inv(inertia)
omega0 = np.array([0, 0, 0])
sigma0 = np.array([0, 0, 0])

# declare orbit stuff
line1 = '1 44031U 98067PX  19083.14584174  .00005852  00000-0  94382-4 0  9997'
line2 = '2 44031  51.6393  63.5548 0003193 165.0023 195.1063 15.54481029  8074'
satellite = EarthSatellite(line1, line2)
ts = load.timescale()
day = 24
hour = 18
minute = 35
second = 0
lons = np.zeros(len(time))
lats = np.zeros(len(time))
alts = np.zeros(len(time))
positions = np.zeros((len(time), 3))
velocities = np.zeros((len(time), 3))
t = ts.utc(2019, 3, day, hour, minute, second)
geo = satellite.at(t)
positions[0] = geo.position.m
velocities[0] = geo.velocity.km_per_s * 1000
subpoint = geo.subpoint()
lons[0] = subpoint.longitude.radians * 180 / np.pi
lats[0] = subpoint.latitude.radians * 180 / np.pi
alts[0] = subpoint.elevation.m

# initialize attitude so that z direction of body frame is aligned with nadir
dcm0 = ut.initial_align_gravity_stabilization(positions[0], velocities[0])
sigma0 = tr.dcm_to_mrp(dcm0)
# initialize angular velocity so that it is approximately the speed of rotation around the earth
omega0_body = np.array([0, -0.00113, 0])
omega0 = dcm0.T @ omega0_body

# The integration
states = np.zeros((len(time), 2, 3))
controls = np.zeros((len(time), 3))
states[0] = [sigma0, omega0]
dcm = np.zeros((len(time), 3, 3))
dcm[0] = tr.mrp_to_dcm(states[0][0])
nadir = np.zeros((len(time), 3))
for i in range(len(time) - 1):
    # propagate orbit
    second += 1*time_step
    if second >= 60:
        minute += 1
        second = 0
    if minute >= 60:
        hour += 1
        minute = 0
    if hour >= 24:
        day += 1
        hour = 0
    t = ts.utc(2019, 3, day, hour, minute, second)
    geo = satellite.at(t)
    positions[i+1] = geo.position.m
    velocities[i+1] = geo.velocity.km_per_s * 1000
    subpoint = geo.subpoint()
    lons[i+1] = subpoint.longitude.radians * 180/np.pi
    lats[i+1] = subpoint.latitude.radians * 180/np.pi
    alts[i+1] = subpoint.elevation.m

    # get unit vector towards nadir in body frame (for gravity gradient torque)
    R0 = np.linalg.norm(positions[i])
    nadir[i] = -positions[i]/R0
    ue = dcm[i] @ nadir[i]

    # get disturbance torque
    controls[i] = dt.gravity_gradient(ue, R0, inertia)

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

    from animation import AnimateAttitude, DrawingVectors, AdditionalPlots
    num = 10
    vec1 = DrawingVectors(nadir[::num], 'single', color='b', label='nadir', length=5)
    ref1 = DrawingVectors(dcm[::num], 'axes', color=['C0', 'C1', 'C2'], label=['Body x', 'Body y', 'Body z'], length=4)
    plot1 = AdditionalPlots(lons[::num], lats[::num], groundtrack=True)
    a = AnimateAttitude(dcm[::num], draw_vector=[vec1, ref1], additional_plots=plot1)
    a.animate_and_plot()

    plt.show()
