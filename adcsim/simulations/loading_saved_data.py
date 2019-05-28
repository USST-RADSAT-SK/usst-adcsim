import numpy as np
import matplotlib.pyplot as plt
from adcsim import disturbance_torques as dt, integrators as it, transformations as tr, util as ut, \
    state_propagations as st, integral_considerations as ic
from skyfield.api import load, EarthSatellite, utc
from astropy.coordinates import get_sun
from astropy.time import Time
import astropy.units as u
from adcsim.CubeSat_model_examples import CubeSatSolarPressureEx1
from adcsim.hysteresis_rod import HysteresisRod
from datetime import datetime, timedelta
from adcsim.atmospheric_density import AirDensityModel
from adcsim.magnetic_field_model import GeoMag
from adcsim.animation import AnimateAttitudeInside, DrawingVectors, AdditionalPlots
from tqdm import tqdm
import xarray as xr

# declare time step for integration
time_step = 0.1
end_time = 30000
time = np.arange(0, end_time, time_step)

# create the CubeSat model
rod1 = HysteresisRod(0.4, 2.5, 12, volume=0.09*np.pi*(0.0005)**2, mass=0.001,
                     scale_factor=10**-2, axes_alignment=np.array([1.0, 0, 0]))
rod2 = HysteresisRod(0.4, 2.5, 12, volume=0.09*np.pi*(0.0005)**2, mass=0.001,
                     scale_factor=10**-2, axes_alignment=np.array([0, 1.0, 0]))
cubesat = CubeSatSolarPressureEx1(inertia=np.diag([3e-2, 5e-2, 8e-3]), magnetic_moment=np.array([0, 0, 1.0]),
                                  hyst_rods=[rod1, rod2])

# create atmospheric density model
air_density = AirDensityModel()

# create magnetic field model
geomag = GeoMag()

# declare memory
states = np.zeros((len(time), 2, 3))
dcm_bn = np.zeros((len(time), 3, 3))
dcm_on = np.zeros((len(time), 3, 3))
dcm_bo = np.zeros((len(time), 3, 3))
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
magneticd = np.zeros((len(time), 3))
density = np.zeros(len(time))
mag_field = np.zeros((len(time), 3))
mag_field_body = np.zeros((len(time), 3))
solar_power = np.zeros(len(time))
is_eclipse = np.zeros(len(time))
hyst_rod = np.zeros((len(time), 3))

# load saved data
save_data = xr.open_dataset('saved_data.nc')
time_tracks = [datetime(2019, 3, 24, 18, 35, 1, tzinfo=utc) + timedelta(seconds=time_step*i) for i in range(len(time))]
saved_data = save_data.interp(time=time_tracks)
sun_vec = saved_data.sun.values
mag_field = saved_data.mag.values
density = saved_data.atmos.values
lons = saved_data.lons.values
lats = saved_data.lats.values
alts = saved_data.alts.values
positions = saved_data.positions.values
velocities = saved_data.velocities.values


# initialize attitude so that z direction of body frame is aligned with nadir
dcm0 = ut.initial_align_gravity_stabilization(positions[0], velocities[0])
sigma0 = tr.dcm_to_mrp(dcm0)
# initialize angular velocity so that it is approximately the speed of rotation around the earth
omega0_body = np.array([0, -0.1, 0.05])
omega0 = dcm0.T @ omega0_body
states[0] = [sigma0, omega0]
dcm_bn[0] = tr.mrp_to_dcm(states[0][0])
dcm_on[0] = ut.inertial_to_orbit_frame(positions[0], velocities[0])
dcm_bo[0] = dcm_bn[0] @ dcm_on[0].T

# get initial b field so that hysteresis rods can be initialized properly
mag_field[0] = geomag.GeoMag(np.array([lats[0], lons[0], alts[0]]), time_tracks[0], output_format='inertial')
mag_field_body[0] = (dcm_bn[0] @ mag_field[0]) * 10 ** -9  # in the body frame in units of T
# cubesat.hyst_rods[0].h_current = mag_field_body[0][0]/cubesat.hyst_rods[0].u0

# the integration
for i in tqdm(range(len(time) - 1)):
    # keep track of dcms for various frame
    dcm_bn[i] = tr.mrp_to_dcm(states[i][0])
    dcm_on[i] = ut.inertial_to_orbit_frame(positions[i], velocities[i])
    dcm_bo[i] = dcm_bn[i] @ dcm_on[i].T

    # get magnetic field in inertial frame (for show right now)
    mag_field_body[i] = (dcm_bn[i] @ mag_field[i]) * 10**-9  # in the body frame in units of T

    # get unit vector towards nadir in body frame (for gravity gradient torque)
    R0 = np.linalg.norm(positions[i])
    nadir[i] = -positions[i]/R0
    ue = dcm_bn[i] @ nadir[i]

    # get sun vector in inertial frame (GCRS) (for solar pressure torque)
    sun_vec_norm = sun_vec[i] / np.linalg.norm(sun_vec[i])
    sun_vec_body[i] = dcm_bn[i] @ sun_vec_norm  # in the body frame

    # get atmospheric density and velocity vector in body frame (for aerodynamic torque)
    vel_body = dcm_bn[i] @ dt.get_air_velocity(velocities[i], positions[i])

    # check if satellite is in eclipse (spherical earth approximation)
    theta = np.arcsin(6378000 / (6378000+alts[i]))
    angle_btw = np.arccos(nadir[i] @ sun_vec_norm)  # note: these vectors will be normalized already.
    if angle_btw < theta:
        is_eclipse[i] = 1

    # get disturbance torque
    aerod[i] = dt.aerodynamic_torque(vel_body, density[i], cubesat)
    if not is_eclipse[i]:
        solard[i] = dt.solar_pressure(sun_vec_body[i], sun_vec[i], positions[i], cubesat)
    gravityd[i] = dt.gravity_gradient(ue, R0, cubesat)
    magneticd[i] = dt.total_magnetic(mag_field_body[i], cubesat)
    hyst_rod[i] = dt.hysteresis_rod_torque(mag_field_body[i], cubesat)
    controls[i] = aerod[i] + solard[i] + gravityd[i] + magneticd[i] + hyst_rod[i]

    # calculate solar power
    if not is_eclipse[i]:
        solar_power[i] = dt.solar_panel_power(sun_vec_body[i], sun_vec[i], positions[i], cubesat)

    # propagate attitude state
    states[i+1] = it.rk4(st.state_dot, time_step, states[i], controls[i], cubesat.inertia, cubesat.inertia_inv)

    # do 'tidy' up things at the end of integration (needed for many types of attitude coordinates)
    states[i+1] = ic.mrp_switching(states[i+1])


if __name__ == "__main__":
    # omegas = states[:, 1]
    # sigmas = states[:, 0]
    #
    #
    def _plot(data, title='', ylabel=''):
        plt.figure()
        plt.plot(time, data)
        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel(ylabel)

    _plot(hyst_rod)

    # _plot(omegas, 'angular velocity components', 'angular velocity (rad/s)')
    # _plot(sigmas, 'mrp components', 'mrp component values')
    # plot the control torque
    # _plot(controls, 'control torque components', 'Torque (Nm)')
    # plot the mrp magnitude
    # _plot(np.linalg.norm(sigmas, axis=1), 'mrp magnitude', '')

    # Calculate angles between body axis and magnetic field
    mag_angles = np.zeros((len(time), 3))
    for i in range(len(time)):
        mag_angles[i] = np.arccos(dcm_bn[i] @ mag_field[i] / np.linalg.norm(mag_field[i]))


    cubesat.hyst_rods[0].plot_limiting_cycle(-150, 150)
    plt.plot(cubesat.hyst_rods[0].h, cubesat.hyst_rods[0].b, color='red', linestyle='--')
    plt.show()

    #_plot(aerod, 'aerodynamic disturbance')
    #_plot(gravityd, 'gravity gradient disturbance')
    #_plot(solard, 'solar radiation pressure disturbance')
    #_plot(magneticd, 'residual magnetic disturbance')

    from adcsim.animation import AnimateAttitude, DrawingVectors, AdditionalPlots
    num = 1000
    vec1 = DrawingVectors(nadir[::num], 'single', color='b', label='nadir', length=0.5)
    vec2 = DrawingVectors(sun_vec[::num], 'single', color='y', label='sun', length=0.5)
    vec3 = DrawingVectors(velocities[::num], 'single', color='g', label='velocity', length=0.5)
    vec4 = DrawingVectors(mag_field[::num], 'single', color='r', label='magnetic field', length=0.5)
    ref1 = DrawingVectors(dcm_bn[::num], 'axes', color=['C0', 'C1', 'C2'], label=['Body x', 'Body y', 'Body z'], length=0.2)
    reforbit = DrawingVectors(dcm_on[::num], 'axes', color=['C0', 'C1', 'C2'], label=['Orbit x', 'Orbit y', 'Orbit z'], length=0.2)
    ref2 = DrawingVectors(dcm_bo[::num], 'axes', color=['C0', 'C1', 'C2'], label=['Body x', 'Body y', 'Body z'], length=0.2)
    plot1 = AdditionalPlots(time[::num], controls[::num], labels=['X', 'Y', 'Z'])
    plot2 = AdditionalPlots(lons[::num], lats[::num], groundtrack=True)
    plot3 = AdditionalPlots(time[::num], is_eclipse[::num])
    # a = AnimateAttitude(dcm_bo[::num], draw_vector=ref2, additional_plots=plot2, cubesat_model=cubesat)
    a = AnimateAttitude(dcm_bn[::num], draw_vector=[ref1, vec1, vec4], additional_plots=plot2,
                        cubesat_model=cubesat)
    a.animate_and_plot()

    plt.show()
