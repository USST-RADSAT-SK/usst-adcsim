"""
This script is meant to make all attitude simulations performed faster. Since we are using the same Two Line Element
for all of our simulations so far, the orbit and environment (magnetic field, sun vector) is always the same. Running
this script will save this information to a netcdf file so that it can just be loaded in simulation scripts.

This saved data can be used in your simulation so long as it goes further in time that the simulation you want to run.

The time_step you can use for this script also does not need to be as short as in most of your attitude simulations. A
10 second time step is fine, but it could probably be even shorter.

In the simulations this data is always interpolated.
"""
import numpy as np
from skyfield.api import load, EarthSatellite, utc
from astropy.coordinates import get_sun
from astropy.time import Time
from datetime import datetime, timedelta
from adcsim.atmospheric_density import AirDensityModel
from adcsim.magnetic_field_model import GeoMag
from tqdm import tqdm
import astropy.units as u
import xarray as xr

# declare time step for integration
time_step = 10
end_time = 3000000
time = np.arange(0, end_time, time_step)

# create atmospheric density model
air_density = AirDensityModel()

# create magnetic field model
geomag = GeoMag()

# declare memory
sun_vec = np.zeros((len(time), 3))
lons = np.zeros(len(time))
lats = np.zeros(len(time))
alts = np.zeros(len(time))
positions = np.zeros((len(time), 3))
velocities = np.zeros((len(time), 3))
density = np.zeros(len(time))
mag_field = np.zeros((len(time), 3))

# declare all orbit stuff
line1 = '1 44031U 98067PX  19083.14584174  .00005852  00000-0  94382-4 0  9997'
line2 = '2 44031  51.6393  63.5548 0003193 165.0023 195.1063 15.54481029  8074'
satellite = EarthSatellite(line1, line2)
ts = load.timescale()
time_track = datetime(2019, 3, 24, 18, 35, 1, tzinfo=utc)
time_tracks = [datetime(2019, 3, 24, 18, 35, 1, tzinfo=utc) + timedelta(seconds=time_step*i) for i in range(len(time))]
t = ts.utc(time_track)
geo = satellite.at(t)
subpoint = geo.subpoint()

# declare initial conditions
positions[0] = geo.position.m
velocities[0] = geo.velocity.km_per_s * 1000
lons[0] = subpoint.longitude.degrees
lats[0] = subpoint.latitude.degrees
alts[0] = subpoint.elevation.m

# get initial b field so that hysteresis rods can be initialized properly
mag_field[0] = geomag.GeoMag(np.array([lats[0], lons[0], alts[0]]), time_track, output_format='inertial')

# the integration
for i in tqdm(range(len(time) - 1)):
    # get magnetic field in inertial frame (for show right now)
    mag_field[i] = geomag.GeoMag(np.array([lats[i], lons[i], alts[i]]), time_tracks[i], output_format='inertial')

    # get sun vector in inertial frame (GCRS) (for solar pressure torque)
    sun_obj = get_sun(Time(time_tracks[i])).cartesian.xyz
    sun_vec[i] = sun_obj.to(u.meter).value

    # get atmospheric density and velocity vector in body frame (for aerodynamic torque)
    density[i] = air_density.air_mass_density(date=time_tracks[i], alt=alts[i]/1000, g_lat=lats[i], g_long=lons[i])

    # propagate orbit
    t = ts.utc(time_tracks[i])
    geo = satellite.at(t)
    positions[i+1] = geo.position.m
    velocities[i+1] = geo.velocity.km_per_s * 1000
    subpoint = geo.subpoint()
    lons[i+1] = subpoint.longitude.degrees
    lats[i+1] = subpoint.latitude.degrees
    alts[i+1] = subpoint.elevation.m

mag_field[i+1] = geomag.GeoMag(np.array([lats[i+1], lons[i+1], alts[i+1]]), time_tracks[i+1], output_format='inertial')
sun_obj = get_sun(Time(time_tracks[i+1])).cartesian.xyz
sun_vec[i+1] = sun_obj.to(u.meter).value
density[i+1] = air_density.air_mass_density(date=time_tracks[i+1], alt=alts[i+1]/1000, g_lat=lats[i+1], g_long=lons[i+1])

# import matplotlib.pyplot as plt
#
#
# def _plot(data, title='', ylabel=''):
#     plt.figure()
#     plt.plot(time, data)
#     plt.title(title)
#     plt.xlabel('Time (s)')
#     plt.ylabel(ylabel)
#
#
# _plot(mag_field)
# _plot(sun_vec)
# _plot(density)
# _plot(positions)
# _plot(velocities)
# _plot(lons)
# _plot(lats)
# _plot(alts)


a = xr.Dataset({'sun': (['time', 'cord'], sun_vec),
                'mag': (['time', 'cord'], mag_field), 'atmos': ('time', density), 'lons': ('time', lons),
                'lats': ('time', lats), 'alts': ('time', alts), 'positions': (['time', 'cord'], positions),
                'velocities': (['time', 'cord'], velocities)},
               coords={'time': time_tracks, 'cord': ['x', 'y', 'z']})
a.to_netcdf('saved_data.nc')
