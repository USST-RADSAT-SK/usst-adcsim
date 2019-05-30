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
import time as tim

# declare time step for integration
time_step = 0.1
end_time = 300
time = np.arange(0, end_time, time_step)

# create the CubeSat model
rod1 = HysteresisRod(0.35, 0.73, 1.59, volume=0.09*np.pi*(0.0005)**2, mass=0.001, integration_size=len(time),
                     scale_factor=1, axes_alignment=np.array([1.0, 0, 0]))
rod2 = HysteresisRod(0.35, 0.73, 1.59, volume=0.09*np.pi*(0.0005)**2, mass=0.001, integration_size=len(time),
                     scale_factor=1, axes_alignment=np.array([0, 1.0, 0]))
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
lons[0] = subpoint.longitude.degrees
lats[0] = subpoint.latitude.degrees
alts[0] = subpoint.elevation.m
# initialize attitude so that z direction of body frame is aligned with nadir
dcm0 = ut.initial_align_gravity_stabilization(positions[0], velocities[0])
sigma0 = tr.dcm_to_mrp(dcm0)
omega0_body = np.array([0, -0.1, 0.05])
omega0 = dcm0.T @ omega0_body
states[0] = [sigma0, omega0]
dcm_bn[0] = tr.mrp_to_dcm(states[0][0])

# get initial b field so that hysteresis rods can be initialized properly
mag_field[0] = geomag.GeoMag(np.array([lats[0], lons[0], alts[0]]), time_track, output_format='inertial')
mag_field_body[0] = (dcm_bn[0] @ mag_field[0]) * 10 ** -9  # in the body frame in units of T
for rod in cubesat.hyst_rods:
    axes = np.argwhere(rod.axes_alignment == 1)[0][0]
    rod.h[0] = mag_field_body[0][axes]/cubesat.hyst_rods[0].u0
    rod.b[0] = rod.b_field_bottom(rod.h[0])

# the integration
for i in tqdm(range(len(time) - 1)):
    # propagate orbit
    t = ts.utc(time_track)
    geo = satellite.at(t)
    positions[i] = geo.position.m
    velocities[i] = geo.velocity.km_per_s * 1000
    subpoint = geo.subpoint()
    lons[i] = subpoint.longitude.degrees
    lats[i] = subpoint.latitude.degrees
    alts[i] = subpoint.elevation.m

    # keep track of dcms for various frame
    dcm_bn[i] = tr.mrp_to_dcm(states[i][0])

    # get magnetic field in inertial frame (for show right now)
    t1 = tim.time()
    mag_field[i] = geomag.GeoMag(np.array([lats[i], lons[i], alts[i]]), time_track, output_format='inertial')
    mag_field_body[i] = (dcm_bn[i] @ mag_field[i]) * 10**-9  # in the body frame in units of T

    # get disturbance torque
    hyst_rod[i] = dt.hysteresis_rod_torque_save(mag_field_body[i], i, cubesat)
    controls[i] = hyst_rod[i]

    # propagate attitude state
    states[i+1] = it.rk4(st.state_dot_mrp, time_step, states[i], controls[i], cubesat.inertia, cubesat.inertia_inv)

    # do 'tidy' up things at the end of integration (needed for many types of attitude coordinates)
    states[i+1] = ic.mrp_switching(states[i+1])

    # iterate time
    time_track = time_track + timedelta(seconds=time_step)


rod1.plot_limiting_cycle(-50, 50)
plt.plot(rod1.h, rod1.b, color='red', linestyle='--')
rod2.plot_limiting_cycle(-50, 50)
plt.plot(rod2.h, rod2.b, color='red', linestyle='--')
