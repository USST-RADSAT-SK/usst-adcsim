import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from skyfield.api import utc
from scipy.interpolate.interpolate import interp1d

from adcsim.CubeSat_model import CubeSat

class OrbitData:
    def __init__(self, sim_params: dict, saved_data: xr.Dataset):
        num_simulation_data_points = int(sim_params['end_time_index'] // sim_params['time_step']) + 1
        start_time = datetime.strptime(sim_params['start_time'], "%Y/%m/%d %H:%M:%S")
        start_time = start_time.replace(tzinfo=utc)
        final_time = start_time + timedelta(seconds=sim_params['time_step']*num_simulation_data_points)
        orbit_data = saved_data.sel(time=slice(None, final_time))
        num_saved_data_points = len(orbit_data.time)
        a = num_saved_data_points
        t = orbit_data.time.values.astype('float')
        t = (t - t[0]) * 1e-9
        ab = np.concatenate((orbit_data.sun.values, orbit_data.mag.values, orbit_data.atmos.values.reshape(a, 1),
                             orbit_data.lons.values.reshape(a, 1), orbit_data.lats.values.reshape(a, 1),
                             orbit_data.alts.values.reshape(a, 1), orbit_data.positions.values, orbit_data.velocities.values),
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

    def interp_orbit_data(self, orbit: OrbitData, t: float):
        orbit.set_time(t)
        self.sun_vec = orbit.sun_vec
        self.mag_field = orbit.mag_field
        self.density = orbit.density
        self.lons = orbit.lons
        self.lats = orbit.lats
        self.alts = orbit.alts
        self.positions = orbit.positions
        self.velocities = orbit.velocities

