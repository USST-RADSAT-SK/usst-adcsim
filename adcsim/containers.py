import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from skyfield.api import utc
from scipy.interpolate.interpolate import interp1d
from collections import namedtuple

from adcsim.CubeSat_model import CubeSat

class OrbitData:
    def __init__(self, sim_params: dict, saved_data: xr.Dataset):
        start_time = np.datetime64(sim_params['start_time'].replace('/', '-').replace(' ', 'T'))
        final_time = start_time + np.timedelta64(round(sim_params['duration'] * 1e9), 'ns')
        if start_time == saved_data.time.values[0]:
            start_index = 0
        else:
            start_index = np.where(saved_data.time.values < start_time)
            if len(start_index[0]) > 0:
                start_index = start_index[0][-1]
            else:
                raise ValueError('Simulation start time preceeds orbit start time')
        final_index = np.where(saved_data.time.values > final_time)
        if len(final_index[0]) > 0:
            final_index = final_index[0][0] + 1
        else:
            raise ValueError('Simulation final time exceeds orbit final time')
        orbit_data = saved_data.isel(time=slice(start_index, final_index))
        t = orbit_data.time.values.astype('float')
        t = (t - t[0]) * 1e-9
        ab = np.concatenate((orbit_data.sun.values, orbit_data.mag.values, orbit_data.atmos.values.reshape(-1, 1),
                             orbit_data.lons.values.reshape(-1, 1), orbit_data.lats.values.reshape(-1, 1),
                             orbit_data.alts.values.reshape(-1, 1), orbit_data.positions.values,
                             orbit_data.velocities.values),
                            axis=1)
        self._interp_data = interp1d(t, ab.T)

    def set_time(self, t: float):
        interpolated = self._interp_data(t)
        self.sun_vec = interpolated[0:3]
        self.mag_field = interpolated[3:6]
        self.density = interpolated[6]
        self.lons = interpolated[7]
        self.lats = interpolated[8]
        self.alts = interpolated[9]
        self.positions = interpolated[10:13]
        self.velocities = interpolated[13:16]

class AttitudeData:
    class _AttitudeData:
        def __init__(self, cubesat: CubeSat):
            self.states = np.zeros((2, 3))
            self.dcm_bn = np.zeros((3, 3))
            self.dcm_on = np.zeros((3, 3))
            self.dcm_bo = np.zeros((3, 3))
            self.controls = np.zeros(3)
            self.nadir = np.zeros(3)
            self.sun_vec = np.zeros(3)
            self.sun_vec_body = np.zeros(3)
            self.density = 0.0
            self.aerod = np.zeros(3)
            self.gravityd = np.zeros(3)
            self.solard = np.zeros(3)
            self.magneticd = np.zeros(3)
            self.mag_field = np.zeros(3)
            self.mag_field_body = np.zeros(3)
            self.solar_power = 0.0
            self.is_eclipse = 0.0
            self.hyst_rod = np.zeros((len(cubesat.hyst_rods), 3))
            self.h_rods = np.zeros((len(cubesat.hyst_rods)))
            self.b_rods = np.zeros((len(cubesat.hyst_rods)))
            self.lons = 0.0
            self.lats = 0.0
            self.alts = 0.0
            self.positions = np.zeros(3)
            self.velocities = np.zeros(3)

    def __init__(self, cubesat):
        self.temp = self._AttitudeData(cubesat)
        self.save = self._AttitudeData(cubesat)

    def interp_orbit_data(self, orbit: OrbitData, t: float, save: bool=False):
        orbit.set_time(t)
        if save:
            self.save.sun_vec = orbit.sun_vec
            self.save.mag_field = orbit.mag_field
            self.save.density = orbit.density
            self.save.lons = orbit.lons
            self.save.lats = orbit.lats
            self.save.alts = orbit.alts
            self.save.positions = orbit.positions
            self.save.velocities = orbit.velocities
        else:
            self.temp.sun_vec = orbit.sun_vec
            self.temp.mag_field = orbit.mag_field
            self.temp.density = orbit.density
            self.temp.lons = orbit.lons
            self.temp.lats = orbit.lats
            self.temp.alts = orbit.alts
            self.temp.positions = orbit.positions
            self.temp.velocities = orbit.velocities
