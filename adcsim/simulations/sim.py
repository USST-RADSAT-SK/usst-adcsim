import numpy as np
from adcsim import disturbance_torques as dt, integrators as it, transformations as tr, util as ut, \
    state_propagations as st, integral_considerations as ic
from adcsim.CubeSat_model import CubeSat
from tqdm import tqdm
import xarray as xr
from adcsim.dcm_convert.dcm_to_stk import dcm_to_stk_simple
from adcsim.containers import AttitudeData, OrbitData
import os
from datetime import datetime, timedelta
from skyfield.api import utc

def sim_attitude(sim_params, cubesat_params, file_name, save=True, ret=False):
    if isinstance(sim_params, str):
        sim_params = eval(sim_params)

    save_every = sim_params['save_every']  # only save the data every number of iterations


    # declare time step for integration
    time_step = sim_params['time_step']
    end_time = sim_params['end_time_index']
    time = np.arange(0, end_time, time_step)
    le = int(len(time)/save_every)

    num_simulation_data_points = int(sim_params['end_time_index'] // sim_params['time_step']) + 1
    start_time = datetime.strptime(sim_params['start_time'], "%Y/%m/%d %H:%M:%S")
    start_time = start_time.replace(tzinfo=utc)
    final_time = start_time + timedelta(seconds=sim_params['time_step']*num_simulation_data_points)

    # create the CubeSat model
    cubesat = CubeSat.fromdict(cubesat_params)

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
    hyst_rod = np.zeros((le, len(cubesat.hyst_rods), 3))
    h_rods = np.zeros((le, len(cubesat.hyst_rods)))
    b_rods = np.zeros((le, len(cubesat.hyst_rods)))

    # def interp_info(i):
    #     ac = interp_data(i)
    #     return ac[0: 3], ac[3: 6], ac[6], ac[7], ac[8], ac[9], ac[10: 13], ac[13: 16]

    # load saved orbit and environment data
    with xr.open_dataset(os.path.join(os.path.dirname(__file__), '../../orbit_pre_process.nc')) as saved_data:
        orbit = OrbitData(sim_params, saved_data)

    # allocate space for attitude data
    attitude = AttitudeData(cubesat)

    # initialize attitude so that z direction of body frame is aligned with nadir
    # sun_vec[0], mag_field[0], density[0], lons[0], lats[0], alts[0], positions[0], velocities[0] = interp_info(0)
    sigma0 = np.array(sim_params['sigma0'])
    dcm0 = tr.mrp_to_dcm(sigma0)
    omega0_body = np.array(sim_params['omega0_body'])
    omega0 = dcm0.T @ omega0_body
    states[0] = state = attitude.states = [sigma0, omega0]
    dcm_bn[0] = attitude.dcm_bn = tr.mrp_to_dcm(states[0][0])
    dcm_on[0] = attitude.dcm_on = ut.inertial_to_orbit_frame(positions[0], velocities[0])
    dcm_bo[0] = attitude.dcm_bo = dcm_bn[0] @ dcm_on[0].T

    # Put hysteresis rods in an initial state that is reasonable. (Otherwise you can get large magnetization from the rods)
    mag_field_body[0] = attitude.mag_field_body = (dcm_bn[0] @ mag_field[0]) * 10 ** -9  # in the body frame in units of T
    for rod in cubesat.hyst_rods:
        rod.define_integration_size(le+1)
        axes = np.argwhere(rod.axes_alignment == 1)[0][0]
        rod.h[0] = rod.h_current = mag_field_body[0][axes]/cubesat.hyst_rods[0].u0
        rod.b[0] = rod.b_current = rod.b_field_bottom(rod.h_current)

    # initialize the disturbance torque object
    disturbance_torques = dt.DisturbanceTorques(**{torque: True for torque in sim_params['disturbance_torques']})

    # the integration
    k = 0
    for i in tqdm(range(len(time) - 1)):
        # # do the interpolation for various things
        # sun_vec[k], mag_field[k], density[k], lons[k], lats[k], alts[k], positions[k], velocities[k] = interp_info(i)
        #
        # # keep track of dcms for various frame
        # dcm_bn[k] = tr.mrp_to_dcm(state[0])
        # dcm_on[k] = ut.inertial_to_orbit_frame(positions[k], velocities[k])
        # dcm_bo[k] = dcm_bn[k] @ dcm_on[k].T
        #
        # # get magnetic field in inertial frame (for show right now)
        # mag_field_body[k] = (dcm_bn[k] @ mag_field[k]) * 10**-9  # in the body frame in units of T
        #
        # # get unit vector towards nadir in body frame (for gravity gradient torque)
        # R0 = np.linalg.norm(positions[k])
        # nadir[k] = -positions[k]/R0
        # ue = dcm_bn[k] @ nadir[k]
        #
        # # get sun vector in inertial frame (GCRS) (for solar pressure torque)
        # sun_vec_norm = sun_vec[k] / np.linalg.norm(sun_vec[k])
        # sun_vec_body[k] = dcm_bn[k] @ sun_vec_norm  # in the body frame
        #
        # # get atmospheric density and velocity vector in body frame (for aerodynamic torque)
        # vel_body = dcm_bn[k] @ dt.get_air_velocity(velocities[k], positions[k])
        #
        # # check if satellite is in eclipse (spherical earth approximation)
        # theta = np.arcsin(6378000 / (6378000+alts[k]))
        # angle_btw = np.arccos(nadir[k] @ sun_vec_norm)  # note: these vectors will be normalized already.
        # if angle_btw < theta:
        #     is_eclipse[k] = 1
        #
        # # get disturbance torque
        # aerod[k] = dt.aerodynamic_torque(vel_body, density[k], cubesat)
        # if not is_eclipse[k]:
        #     solard[k] = dt.solar_pressure(sun_vec_body[k], sun_vec[k], positions[k], cubesat)
        # gravityd[k] = dt.gravity_gradient(ue, R0, cubesat)
        # magneticd[k] = dt.total_magnetic(mag_field_body[k], cubesat)
        # hyst_rod[k] = dt.hysteresis_rod_torque_save(mag_field_body[k], k, cubesat)
        # controls[k] = aerod[k] + solard[k] + gravityd[k] + magneticd[k] + hyst_rod[k].sum(axis=0)
        #
        # # calculate solar power
        # if not is_eclipse[k]:
        #     solar_power[k] = dt.solar_panel_power(sun_vec_body[k], sun_vec[k], positions[k], cubesat)

        # propagate attitude state
        disturbance_torques.propagate_hysteresis = True  # should propagate the hysteresis history, bringing it up to the current position
        state = it.rk4(st.state_dot_mrp, time[i], state, time_step, attitude, orbit, cubesat, disturbance_torques)
        # controls[k] = ...

        # do 'tidy' up things at the end of integration (needed for many types of attitude coordinates)
        state = ic.mrp_switching(state)
        if not i % save_every:
            states[k] = state
            dcm_bn[k] = attitude.dcm_bn
            dcm_on[k] = attitude.dcm_on
            dcm_bo[k] = attitude.dcm_bo
            controls[k] = attitude.controls
            nadir[k] = attitude.nadir
            sun_vec[k] = attitude.sun_vec
            sun_vec_body[k] = attitude.sun_vec_body
            lons[k] = attitude.lons
            lats[k] = attitude.lats
            alts[k] = attitude.alts
            positions[k] = attitude.positions
            velocities[k] = attitude.velocities
            aerod[k] = attitude.aerod
            gravityd[k] = attitude.gravityd
            solard[k] = attitude.solard
            magneticd[k] = attitude.magneticd
            density[k] = attitude.density
            mag_field[k] = attitude.mag_field
            mag_field_body[k] = attitude.mag_field_body
            solar_power[k] = attitude.solar_power
            is_eclipse[k] = attitude.is_eclipse
            hyst_rod[k] = attitude.hyst_rod
            k += 1
            disturbance_torques.save_hysteresis = True
            if k >= le:
                break
    states = np.delete(states, 1, axis=0)
    for i, rod in enumerate(cubesat.hyst_rods):
        b_rods[:, i] = rod.b = np.delete(rod.b, 1, axis=0)
        h_rods[:, i] = rod.h = np.delete(rod.h, 1, axis=0)


    omegas = states[:, 1]
    sigmas = states[:, 0]

    # save the data
    sim_params_dict = {'time_step': time_step, 'save_every': save_every, 'end_time_index': end_time,
                       'start_time': start_time.strftime('%Y/%m/%d %H:%M:%S'),
                       'final_time': final_time.strftime('%Y/%m/%d %H:%M:%S'), 'omega0_body': omega0_body.tolist(),
                       'sigma0': sigma0.tolist()}
    a = xr.Dataset({'sun': (['time', 'cord'], sun_vec),
                    'mag': (['time', 'cord'], mag_field),
                    'atmos': ('time', density),
                    'lons': ('time', lons),
                    'lats': ('time', lats),
                    'alts': ('time', alts),
                    'positions': (['time', 'cord'], positions),
                    'velocities': (['time', 'cord'], velocities),
                    'dcm_bn': (['time', 'dcm_mat_dim1', 'dcm_mat_dim2'], dcm_bn),
                    'dcm_bo': (['time', 'dcm_mat_dim1', 'dcm_mat_dim2'], dcm_bo),
                    'angular_vel': (['time', 'cord'], omegas),
                    'controls': (['time', 'cord'], controls),
                    'gg_torque': (['time', 'cord'], gravityd),
                    'aero_torque': (['time', 'cord'], aerod),
                    'solar_torque': (['time', 'cord'], solard),
                    'magnetic_torque': (['time', 'cord'], magneticd),
                    'hyst_rod_torque': (['time', 'hyst_rod', 'cord'], hyst_rod),
                    'hyst_rod_magnetization': (['time', 'hyst_rod'], b_rods),
                    'hyst_rod_external_field': (['time', 'hyst_rod'], h_rods),
                    'nadir': (['time', 'cord'], nadir),
                    'solar_power': ('time', solar_power)},
                   coords={'time': np.arange(0, le, 1), 'cord': ['x', 'y', 'z'], 'hyst_rod': [f'rod{i}' for i in range(len(cubesat.hyst_rods))]},
                   attrs={'simulation_parameters': str(sim_params_dict), 'cubesat_parameters': str(cubesat.asdict()),
                          'description': 'University of kentucky attitude propagator software '
                                         '(they call it SNAP) recreation'})
    # Note: the simulation and cubesat parameter dictionaries are saved as strings for the nc file. If you wish
    # you could just eval(a.cubesat_parameters) to get the dictionary back.
    if save:
        a.to_netcdf(os.path.join(os.path.dirname(__file__), f'../../{file_name}.nc'))
        dcm_to_stk_simple(time[::save_every], dcm_bn, os.path.join(os.path.dirname(__file__), f'../../{file_name}.a'))
    if ret:
        return a


# function to continue a simulation from the simulation data time for a given number of additional iterations
def continue_sim(sim_dataset, num_iter, file_name):
    import copy
    original_params = eval(sim_dataset.simulation_parameters)
    sim_params = copy.deepcopy(original_params)
    sim_params['end_time_index'] = num_iter
    sim_params['start_time'] = sim_params['final_time']
    last_data = sim_dataset.isel(time=-1)
    sim_params['omega0_body'] = last_data.dcm_bn.values @ last_data.angular_vel.values
    sim_params['sigma0'] = tr.dcm_to_mrp(last_data.dcm_bn.values)
    cubesat_params = eval(sim_dataset.cubesat_parameters)
    new_data = sim_attitude(sim_params, cubesat_params, 'easter_egg', save=False, ret=True)
    new_data['time'] = np.arange(len(sim_dataset.time), len(sim_dataset.time) + sim_params['end_time_index']*sim_params['save_every'], 1)
    a = xr.concat([sim_dataset, new_data], dim='time')
    true_sim_params = eval(a.simulation_parameters)
    true_sim_params['end_time_index'] = original_params['end_time_index'] + num_iter
    true_sim_params['final_time'] = eval(new_data.simulation_parameters)['final_time']
    a.attrs['simulation_parameters'] = str(true_sim_params)
    a.to_netcdf(os.path.join(os.path.dirname(__file__), f'../../{file_name}.nc'))


if __name__ == "__main__":

    # run a short simulation
    from adcsim.hysteresis_rod import HysteresisRod
    from adcsim.CubeSat_model_examples import CubeSatModel
    sim_params = {
        'time_step': 0.01,
        'save_every': 10,
        'end_time_index': 20,
        'start_time': '2019/03/24 18:35:01',
        'omega0_body': (np.pi / 180) * np.array([-2, 3, 3.5]),
        'sigma0': [0.6440095705520482, 0.39840861883760637, 0.18585931442943798]
    }
    # create inital cubesat parameters dict (the raw data is way to large to do manually like above)
    rod1 = HysteresisRod(br=0.35, bs=0.73, hc=1.59, volume=0.075 / (100 ** 3), axes_alignment=np.array([1.0, 0, 0]))
    rod2 = HysteresisRod(br=0.35, bs=0.73, hc=1.59, volume=0.075 / (100 ** 3), axes_alignment=np.array([0, 1.0, 0]))
    cubesat = CubeSatModel(inertia=np.diag([8 * (10 ** -3), 8 * (10 ** -3), 2 * (10 ** -3)]),
                           magnetic_moment=np.array([0, 0, 1.5]),
                           hyst_rods=[rod1, rod2])
    cubesat_params = cubesat.asdict()

    data = sim_attitude(sim_params, cubesat_params, 'test0', save=True, ret=True)

    # run the simulation longer
    continue_sim(data, 30, 'test1')