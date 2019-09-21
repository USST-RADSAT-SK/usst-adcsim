import numpy as np
from adcsim import disturbance_torques as dt, integrators as it, transformations as tr, util as ut, \
    state_propagations as st, integral_considerations as ic
from skyfield.api import utc
from adcsim.CubeSat_model_examples import CubeSatModel
from adcsim.hysteresis_rod import HysteresisRod
from adcsim.simulations.sim import sim_attitude
from datetime import datetime, timedelta
from tqdm import tqdm
import xarray as xr
from scipy.interpolate.interpolate import interp1d
import os

# create initial simulation parameters dict
spin = -1 / 36 * np.pi  # maximum 5 degree per axis spin by requirement 3.08
sim_params = {
    'time_step': 0.01,
    'save_every': 10,
    'end_time_index': 21000,
    'start_time': '2019/03/24 18:35:01',
    'omega0_body': (np.pi/180) * np.array([-5, -5, -5]),
    'sigma0': [0.6440095705520482, 0.39840861883760637, 0.18585931442943798]
}

# create inital cubesat parameters dict (the raw data is way to large to do manually like above)
rod1 = HysteresisRod(br=0.35, bs=0.73, hc=1.59, volume=0.075/(100**3), axes_alignment=np.array([1.0, 0, 0]))
rod2 = HysteresisRod(br=0.35, bs=0.73, hc=1.59, volume=0.075/(100**3), axes_alignment=np.array([0, 1.0, 0]))
cubesat = CubeSatModel(inertia=np.diag([0.008, 0.008, 0.002]), magnetic_moment=np.array([0, 0, 0.5]),
                       hyst_rods=[rod1, rod2])
cubesat_params = cubesat.asdict()


sim_attitude(sim_params, cubesat_params, 'run1', aerodynamic_torque=False)
