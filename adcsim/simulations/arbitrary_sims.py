from adcsim.simulations.sim import sim_attitude
import numpy as np
from adcsim.hysteresis_rod import HysteresisRod
from adcsim.CubeSat_model_examples import CubeSatModel
from multiprocessing import Pool
import os
import copy


# create initial simulation parameters dict
sim_params = {
    'time_step': 0.01,
    'save_every': 10,
    'end_time_index': 5000,
    'start_time': '2019/03/24 18:35:01',
    'final_time': '2019/03/24 18:35:51',
    'omega0_body': np.array([-2, 3, 3.5]),
    'sigma0': [0.6440095705520482, 0.39840861883760637, 0.18585931442943798]
}

# create inital cubesat parameters dict (the raw data is way to large to do manually like above)
rod1 = HysteresisRod(br=0.35, bs=0.73, hc=1.59, volume=0.075/(100**3), axes_alignment=np.array([1.0, 0, 0]))
rod2 = HysteresisRod(br=0.35, bs=0.73, hc=1.59, volume=0.075/(100**3), axes_alignment=np.array([0, 1.0, 0]))
cubesat = CubeSatModel(inertia=np.diag([8*(10**-3), 8*(10**-3), 2*(10**-3)]), magnetic_moment=np.array([0, 0, 1.5]),
                       hyst_rods=[rod1, rod2])
cubesat_params = cubesat.asdict()


# USING MULTIPLE CORES
if __name__ == "__main__":
    pool = Pool(os.cpu_count())  # the number of cores differ on different computers

    cp = []

    # First simulation (parameters what is shown above)
    cp.append([copy.deepcopy(sim_params), copy.deepcopy(cubesat_params), 'first'])

    # Second simulation (alter the parameters as desired before appending it to cp)
    cubesat_params['hyst_rods'][0]['axes_alignment'] = np.array([0, 0, 1.0])
    cubesat_params['magnetic_moment'] = np.array([1.5, 0, 0])
    cp.append([copy.deepcopy(sim_params), copy.deepcopy(cubesat_params), 'second'])

    # Third simulation (alter the parameters as desired before appending it to cp)
    sim_params['omega0_body'] = [0, -0.06666, 0]
    cp.append([copy.deepcopy(sim_params), copy.deepcopy(cubesat_params), 'third'])

    # Second simulation (alter the parameters as desired before appending it to cp)
    cubesat_params['hyst_rods'][0]['axes_alignment'] = np.array([1.0, 0, 0])
    cubesat_params['magnetic_moment'] = np.array([0, 0, 1.5])
    cp.append([copy.deepcopy(sim_params), copy.deepcopy(cubesat_params), 'fourth'])

    # ...
    # ...
    # ...
    # Infinity simulation

    pool.starmap(sim_attitude, cp)
