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
sim_params = [{
    'time_step': 0.1,
    'save_every': 10,
    'duration': 3600,
    'start_time': '2019/03/24 18:35:01',
    'omega0_body': (np.pi/180) * np.array([-2, 3, 3.5]),
    'sigma0': [0.6440095705520482, 0.39840861883760637, 0.18585931442943798],
    'disturbance_torques': ['gravity', 'magnetic', 'hysteresis', 'aerodynamic', 'solar'],
    'calculate_power': False
}]

# create inital cubesat parameters dict (the raw data is way to large to do manually like above)
rod1 = HysteresisRod(br=0.35, bs=0.73, hc=1.59, volume=0.075/(100**3), axes_alignment=np.array([1.0, 0, 0]))
rod2 = HysteresisRod(br=0.35, bs=0.73, hc=1.59, volume=0.075/(100**3), axes_alignment=np.array([0, 1.0, 0]))
cubesat = CubeSatModel(inertia=np.diag([8*(10**-3), 8*(10**-3), 2*(10**-3)]), magnetic_moment=np.array([0, 0, 0.5]),
                       hyst_rods=[rod1, rod2])
cubesat_params = [cubesat.asdict()]

# make a copy but set initial angular velocity to 0 (import to use dict.copy() to avoid errors)
sim_params += [sim_params[0].copy()]
sim_params[1]['omega-_body'] = np.zeros(3)
cubesat_params += [cubesat_params[0].copy()]

# wrapper for multiprocessing to access sim_attitude
def sim_wrapper(i: int):
    sim_attitude(sim_params[i], cubesat_params[i], f'run{i}')


if __name__ == '__main__':
    from multiprocessing import Pool

    sim_params = sim_params[:1]
    n = min(len(sim_params), 8)

    if n > 1:
        # run simulations in parallel, as many as you have processors
        with Pool(n) as pool:
            m = pool.map_async(sim_wrapper, range(len(sim_params)))
            m.get()
    else:
        sim_wrapper(0)

