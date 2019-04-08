import numpy as np
import matplotlib.pyplot as plt
import transformations as tr
import state_propagations as st
import integrators as it
import integral_considerations as ic
import disturbance_torques as dt
import util as ut
from skyfield.api import load, EarthSatellite, utc
from astropy.coordinates import get_sun
from astropy.time import Time
from CubeSat_model_examples import CubeSatSolarPressureEx1
from datetime import datetime, timedelta
from atmospheric_density import AirDensityModel
from magnetic_field_model import magnetic_field

# declare time step for integration
time_step = 10
end_time = 10000
time = np.arange(0, end_time, time_step)

# create the CubeSat model
cubesat = CubeSatSolarPressureEx1(inertia=np.diag([2*(10**-2), 4*(10**-2), 5*(10**-3)]),
                                  center_of_mass=np.array([0, 0, 0]))

# load class to get atmospheric density
air_density = AirDensityModel()

# declare memory
states = np.zeros((len(time), 2, 3))
dcm = np.zeros((len(time), 3, 3))
controls = np.zeros((len(time), 3))
nadir = np.zeros((len(time), 3))
sun_vec = np.zeros((len(time), 3))
sun_vec_body = np.zeros((len(time), 3))
lons = np.zeros(len(time))
lats = np.zeros(len(time))
alts = np.zeros(len(time))
positions = np.zeros((len(time), 3))
velocities = np.zeros((len(time), 3))
aerod = np.zeros((len(time), 3))
gravityd = np.zeros((len(time), 3))
solard = np.zeros((len(time), 3))
density = np.zeros(len(time))
mag_field = np.zeros((len(time), 3))
mag_field_body = np.zeros((len(time), 3))

# declare all orbit stuff
line1 = '1 44031U 98067PX  19083.14584174  .00005852  00000-0  94382-4 0  9997'
line2 = '2 44031  51.6393  63.5548 0003193 165.0023 195.1063 15.54481029  8074'
satellite = EarthSatellite(line1, line2)
ts = load.timescale()
time_track = datetime(2019, 3, 24, 18, 35, 1, tzinfo=utc)
t = ts.utc(time_track)
geo = satellite.at(t)
subpoint = geo.subpoint()

# declare initial conditions
positions[0] = geo.position.m
velocities[0] = geo.velocity.km_per_s * 1000
lons[0] = subpoint.longitude.radians * 180 / np.pi
lats[0] = subpoint.latitude.radians * 180 / np.pi
alts[0] = subpoint.elevation.m
# initialize attitude so that z direction of body frame is aligned with nadir
dcm0 = ut.initial_align_gravity_stabilization(positions[0], velocities[0])
sigma0 = tr.dcm_to_mrp(dcm0)
# initialize angular velocity so that it is approximately the speed of rotation around the earth
omega0_body = np.array([0, -0.00113, 0])
omega0 = dcm0.T @ omega0_body
states[0] = [sigma0, omega0]
dcm[0] = tr.mrp_to_dcm(states[0][0])

# the integration
for i in range(len(time) - 1):
    print(i)

    # get magnetic field (for show right now)
    mag_field[i] = magnetic_field(time_track, lats[i], lons[i], alts[i])
    mag_field_body[i] = dcm[i] @ mag_field[i]  # in the body frame

    # get unit vector towards nadir in body frame (for gravity gradient torque)
    R0 = np.linalg.norm(positions[i])
    nadir[i] = -positions[i]/R0
    ue = dcm[i] @ nadir[i]

    # get sun vector in GCRS (for solar pressure torque)
    sun_vec[i] = get_sun(Time(time_track)).cartesian.xyz.value
    sun_vec[i] = sun_vec[i]/np.linalg.norm(sun_vec[i])
    sun_vec_body[i] = dcm[i] @ sun_vec[i]

    # get atmospheric density and velocity vector in body frame (for aerodynamic torque)
    density[i] = air_density.air_mass_density(date=time_track, alt=alts[i]/1000, g_lat=lats[i], g_long=lons[i])
    vel_body = dcm[i] @ velocities[i]

    # get disturbance torque
    aerod[i] = dt.aerodynamic_torque(vel_body, density[i], cubesat)
    solard[i] = dt.solar_pressure(sun_vec_body[i], cubesat)
    gravityd[i] = dt.gravity_gradient(ue, R0, cubesat)
    controls[i] = aerod[i] + solard[i] + gravityd[i]

    # propagate orbit
    time_track = time_track + timedelta(seconds=time_step)
    t = ts.utc(time_track)
    geo = satellite.at(t)
    positions[i+1] = geo.position.m
    velocities[i+1] = geo.velocity.km_per_s * 1000
    subpoint = geo.subpoint()
    lons[i+1] = subpoint.longitude.radians * 180/np.pi
    lats[i+1] = subpoint.latitude.radians * 180/np.pi
    alts[i+1] = subpoint.elevation.m

    # propagate attitude state
    states[i+1] = it.rk4(st.state_dot, time_step, states[i], controls[i], cubesat.inertia, cubesat.inertia_inv)

    # do 'tidy' up things at the end of integration (needed for many types of attitude coordinates)
    states[i+1] = ic.mrp_switching(states[i+1])
    dcm[i+1] = tr.mrp_to_dcm(states[i+1][0])


if __name__ == "__main__":
    omegas = states[:, 1]
    sigmas = states[:, 0]


    def _plot(data, title='', ylabel=''):
        plt.figure()
        plt.plot(time, data)
        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel(ylabel)


    # _plot(omegas, 'angular velocity components', 'angular velocity (rad/s)')
    # _plot(sigmas, 'mrp components', 'mrp component values')
    # plot the control torque
    # _plot(controls, 'control torque components', 'Torque (Nm)')
    # plot the mrp magnitude
    # _plot(np.linalg.norm(sigmas, axis=1), 'mrp magnitude', '')

    _plot(aerod, 'aerodynamic disturbance')
    _plot(gravityd, 'gravity gradient disturbance')
    _plot(solard, 'solar radiation pressure disturbance')

    from animation import AnimateAttitude, DrawingVectors, AdditionalPlots
    num = 10
    vec1 = DrawingVectors(nadir[::num], 'single', color='b', label='nadir', length=0.5)
    vec2 = DrawingVectors(sun_vec[::num], 'single', color='y', label='sun', length=0.5)
    vec3 = DrawingVectors(velocities[::num], 'single', color='g', label='velocity', length=0.5)
    vec4 = DrawingVectors(mag_field[::num], 'single', color='r', label='magnetic field', length=0.5)
    ref1 = DrawingVectors(dcm[::num], 'axes', color=['C0', 'C1', 'C2'], label=['Body x', 'Body y', 'Body z'], length=0.2)
    plot1 = AdditionalPlots(time[::num], controls[::num], labels=['X', 'Y', 'Z'])
    plot2 = AdditionalPlots(lons[::num], lats[::num], groundtrack=True)
    a = AnimateAttitude(dcm[::num], draw_vector=[vec1, vec2, vec3, vec4, ref1], additional_plots=plot2, cubesat_model=cubesat)
    a.animate_and_plot()

    plt.show()
