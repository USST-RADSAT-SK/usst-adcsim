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
from scipy.interpolate.interpolate import interp1d

save_every = 1

# declare time step for integration
time_step = 0.1
end_time = 300
time = np.arange(0, end_time, time_step)

# create the CubeSat model
rod1 = HysteresisRod(0.4, 2.5, 12, volume=0.09*np.pi*(0.0005)**2, mass=0.001, integration_size=len(time),
                     scale_factor=10**-2, axes_alignment=np.array([1.0, 0, 0]))
rod2 = HysteresisRod(0.4, 2.5, 12, volume=0.09*np.pi*(0.0005)**2, mass=0.001, integration_size=len(time),
                     scale_factor=10**-2, axes_alignment=np.array([0, 1.0, 0]))
cubesat = CubeSatSolarPressureEx1(inertia=np.diag([3e-2, 5e-2, 8e-3]), magnetic_moment=np.array([0, 0, 1.0]),
                                  hyst_rods=[rod1, rod2])

# create atmospheric density model
air_density = AirDensityModel()

# create magnetic field model
geomag = GeoMag()

# declare memory
le = int(len(time)/save_every)
states = np.zeros((le+1, 2, 3))
dcm_bn = np.zeros((le, 3, 3))
dcm_on = np.zeros((le, 3, 3))
dcm_bo = np.zeros((le, 3, 3))
controls = np.zeros((le, 3))
nadir = np.zeros((le, 3))
sun_vec = np.zeros((le, 3))
sun_vec_body = np.zeros((le, 3))
lons = np.zeros(le)
lats = np.zeros(le)
alts = np.zeros(le)
positions = np.zeros((le, 3))
velocities = np.zeros((le, 3))
aerod = np.zeros((le, 3))
gravityd = np.zeros((le, 3))
solard = np.zeros((le, 3))
magneticd = np.zeros((le, 3))
density = np.zeros(le)
mag_field = np.zeros((le, 3))
mag_field_body = np.zeros((le, 3))
solar_power = np.zeros(le)
is_eclipse = np.zeros(le)
hyst_rod = np.zeros((le, 3))

# load saved data
save_data = xr.open_dataset('saved_data.nc')
time_track = datetime(2019, 3, 24, 18, 35, 1, tzinfo=utc)
final_time = time_track + timedelta(seconds=time_step*len(time))
saved_data = save_data.sel(time=slice(None, final_time))
x = np.linspace(0, len(time), len(saved_data.time))
a = len(saved_data.time)
ab = np.concatenate((saved_data.sun.values, saved_data.mag.values, saved_data.atmos.values.reshape(a, 1),
                     saved_data.lons.values.reshape(a, 1), saved_data.lats.values.reshape(a, 1),
                     saved_data.alts.values.reshape(a, 1), saved_data.positions.values, saved_data.velocities.values),
                    axis=1)
interp_data = interp1d(x, ab.T)


def interp_info(i):
    ac = interp_data(i)
    return ac[0: 3], ac[3: 6], ac[6], ac[7], ac[8], ac[9], ac[10: 13], ac[13: 16]

# initialize attitude so that z direction of body frame is aligned with nadir
sun_vec[0], mag_field[0], density[0], lons[0], lats[0], alts[0], positions[0], velocities[0] = interp_info(0)
dcm0 = ut.initial_align_gravity_stabilization(positions[0], velocities[0])
sigma0 = tr.dcm_to_mrp(dcm0)
# initialize angular velocity so that it is approximately the speed of rotation around the earth
omega0_body = np.array([0, -0.1, 0.05])
omega0 = dcm0.T @ omega0_body
states[0] = state = [sigma0, omega0]
dcm_bn[0] = tr.mrp_to_dcm(states[0][0])
dcm_on[0] = ut.inertial_to_orbit_frame(positions[0], velocities[0])
dcm_bo[0] = dcm_bn[0] @ dcm_on[0].T

# get initial b field so that hysteresis rods can be initialized properly
mag_field[0] = geomag.GeoMag(np.array([lats[0], lons[0], alts[0]]), time_track, output_format='inertial')
mag_field_body[0] = (dcm_bn[0] @ mag_field[0]) * 10 ** -9  # in the body frame in units of T
cubesat.hyst_rods[0].h[0] = mag_field_body[0][0]/cubesat.hyst_rods[0].u0

# the integration
k = 0
for i in tqdm(range(len(time) - 1)):
    if not i % save_every:
        # do the interpolation for various things
        sun_vec[k], mag_field[k], density[k], lons[k], lats[k], alts[k], positions[k], velocities[k] = interp_info(i)

        # keep track of dcms for various frame
        dcm_bn[k] = tr.mrp_to_dcm(state[0])
        dcm_on[k] = ut.inertial_to_orbit_frame(positions[k], velocities[k])
        dcm_bo[k] = dcm_bn[k] @ dcm_on[k].T

        # get magnetic field in inertial frame (for show right now)
        mag_field_body[k] = (dcm_bn[k] @ mag_field[k]) * 10**-9  # in the body frame in units of T

        # get unit vector towards nadir in body frame (for gravity gradient torque)
        R0 = np.linalg.norm(positions[k])
        nadir[k] = -positions[k]/R0
        ue = dcm_bn[k] @ nadir[k]

        # get sun vector in inertial frame (GCRS) (for solar pressure torque)
        sun_vec_norm = sun_vec[k] / np.linalg.norm(sun_vec[k])
        sun_vec_body[k] = dcm_bn[k] @ sun_vec_norm  # in the body frame

        # get atmospheric density and velocity vector in body frame (for aerodynamic torque)
        vel_body = dcm_bn[k] @ dt.get_air_velocity(velocities[k], positions[k])

        # check if satellite is in eclipse (spherical earth approximation)
        theta = np.arcsin(6378000 / (6378000+alts[k]))
        angle_btw = np.arccos(nadir[k] @ sun_vec_norm)  # note: these vectors will be normalized already.
        if angle_btw < theta:
            is_eclipse[k] = 1

        # get disturbance torque
        aerod[k] = dt.aerodynamic_torque(vel_body, density[k], cubesat)
        if not is_eclipse[k]:
            solard[k] = dt.solar_pressure(sun_vec_body[k], sun_vec[k], positions[k], cubesat)
        gravityd[k] = dt.gravity_gradient(ue, R0, cubesat)
        magneticd[k] = dt.total_magnetic(mag_field_body[k], cubesat)
        hyst_rod[k] = dt.hysteresis_rod_torque(mag_field_body[k], i, cubesat)
        controls[k] = aerod[k] + solard[k] + gravityd[k] + magneticd[k] + hyst_rod[k]

        # calculate solar power
        if not is_eclipse[k]:
            solar_power[k] = dt.solar_panel_power(sun_vec_body[k], sun_vec[k], positions[k], cubesat)

        # propagate attitude state
        states[k+1] = it.rk4(st.state_dot, time_step, state, controls[k], cubesat.inertia, cubesat.inertia_inv)

        # do 'tidy' up things at the end of integration (needed for many types of attitude coordinates)
        states[k+1] = state = ic.mrp_switching(states[k+1])
        k += 1
    else:
        # do the interpolation for various things
        sun_veci, mag_fieldi, densityi, lonsi, latsi, altsi, positionsi, velocitiesi = interp_info(i)

        # keep track of dcms for various frame
        dcm_bni = tr.mrp_to_dcm(state[0])
        dcm_oni = ut.inertial_to_orbit_frame(positionsi, velocitiesi)
        dcm_boi = dcm_bni @ dcm_oni.T

        # get magnetic field in inertial frame (for show right now)
        mag_field_bodyi = (dcm_bni @ mag_fieldi) * 10**-9  # in the body frame in units of T

        # get unit vector towards nadir in body frame (for gravity gradient torque)
        R0 = np.linalg.norm(positionsi)
        nadiri = -positionsi/R0
        ue = dcm_bni @ nadiri

        # get sun vector in inertial frame (GCRS) (for solar pressure torque)
        sun_vec_norm = sun_veci / np.linalg.norm(sun_veci)
        sun_vec_bodyi = dcm_bni @ sun_vec_norm  # in the body frame

        # get atmospheric density and velocity vector in body frame (for aerodynamic torque)
        vel_body = dcm_bni @ dt.get_air_velocity(velocitiesi, positionsi)

        # check if satellite is in eclipse (spherical earth approximation)
        theta = np.arcsin(6378000 / (6378000+altsi))
        angle_btw = np.arccos(nadiri @ sun_vec_norm)  # note: these vectors will be normalized already.
        if angle_btw < theta:
            is_eclipsei = 1
        else:
            is_eclipsei = 0

        # get disturbance torque
        aerodi = dt.aerodynamic_torque(vel_body, densityi, cubesat)
        if not is_eclipsei:
            solardi = dt.solar_pressure(sun_vec_bodyi, sun_veci, positionsi, cubesat)
        else:
            solardi = np.zeros(3)
        gravitydi = dt.gravity_gradient(ue, R0, cubesat)
        magneticdi = dt.total_magnetic(mag_field_bodyi, cubesat)
        hyst_rodi = dt.hysteresis_rod_torque(mag_field_bodyi, i, cubesat)
        controlsi = aerodi + solardi + gravitydi + magneticdi + hyst_rodi

        # calculate solar power
        if not is_eclipsei:
            solar_poweri = dt.solar_panel_power(sun_vec_bodyi, sun_veci, positionsi, cubesat)

        # propagate attitude state
        state = it.rk4(st.state_dot, time_step, state, controlsi, cubesat.inertia, cubesat.inertia_inv)

        # do 'tidy' up things at the end of integration (needed for many types of attitude coordinates)
        state = ic.mrp_switching(state)
states = np.delete(states, 1, axis=0)

if __name__ == "__main__":
    # omegas = states[:, 1]
    # sigmas = states[:, 0]
    #
    #
    def _plot(data, title='', ylabel=''):
        plt.figure()
        plt.plot(time[::save_every], data)
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
