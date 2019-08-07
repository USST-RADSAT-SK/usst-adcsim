from adcsim.simulations.sim import sim_attitude
import numpy as np
from adcsim.hysteresis_rod import HysteresisRod
from adcsim.CubeSat_model_examples import CubeSatModel

# create initial simulation parameters dict
sim_params = {
    'time_step': 0.01,
    'save_every': 10,
    'end_time_index': 50,
    'start_time': '2019/03/24 18:35:01',
    'final_time': '2019/03/24 18:35:51',
    'omega0_body': [-2, 3, 3.5],
    'sigma0': [0.6440095705520482, 0.39840861883760637, 0.18585931442943798]
}

# create inital cubesat parameters dict (the raw data is way to large to do manually like above)
rod1 = HysteresisRod(br=0.35, bs=0.73, hc=1.59, volume=0.075/(100**3), axes_alignment=np.array([1.0, 0, 0]))
rod2 = HysteresisRod(br=0.35, bs=0.73, hc=1.59, volume=0.075/(100**3), axes_alignment=np.array([0, 1.0, 0]))
cubesat = CubeSatModel(inertia=np.diag([8*(10**-3), 8*(10**-3), 2*(10**-3)]), magnetic_moment=np.array([0, 0, 1.5]),
                       hyst_rods=[rod1, rod2])
cubesat_params = cubesat.asdict()

# iterate over the desired permanent magnet strengths and run the simulation each time
perm_strengths = np.arange(0.25, 5 + 0.25, 0.25)
for p in perm_strengths:
    cubesat_params['magnetic_moment'][-1] = p
    sim_attitude(sim_params, cubesat_params, f'run_magmoment_{p}')
