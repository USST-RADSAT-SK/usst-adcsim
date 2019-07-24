"""
Chunks of code that can be used for post processing of data after it is saved to a netcdf file
"""
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from adcsim.CubeSat_model_examples import CubeSatModel
from adcsim.animation import AnimateAttitude, DrawingVectors, AdditionalPlots
import os


def nc_loop(mag_str):

    # load data from the run
    data = xr.open_dataset(os.path.join(os.path.dirname(__file__), '../../magnet_runs/' + str(mag_str) + '_G_magnet.nc'))
    sim_params = eval(data.simulation_parameters)
    time = np.arange(0, sim_params['end_time_index'], sim_params['time_step'])[:-1]
    le = int(len(time)/sim_params['save_every'])

    # declare a CubeSat (This is only for animations, the cubesat does not need to match the one used in the run,
    # you may want it to look the same tho)
    cubesat = CubeSatModel()


    def _plot(datas, title='', ylabel=''):
        plt.figure()
        plt.plot(time[::sim_params['save_every']], datas)
        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel(ylabel)
        outputfile = '..\\..\\' + '\\magnet_runs\\' + str(mag_str) + '\\' + str(mag_str) + '_' + title
        plt.savefig(os.path.join(os.path.dirname(__file__), outputfile))


    _plot(data.angular_vel.values, 'angular velocity components', 'angular velocity (rad/s)')
    _plot(np.linalg.norm(data.angular_vel.values, axis=1), 'angular velocity magnitude', 'angular velocity (rad/s)')
    _plot(data.controls.values, 'control torque components', 'Torque (Nm)')

    # Calculate angles between body axis and magnetic field
    mag_angles = np.zeros((le, 3))
    dcm_bn = data.dcm_bn.values
    mag = data.mag.values
    for i in range(le):
        mag_angles[i] = np.arccos(dcm_bn[i] @ mag[i] / np.linalg.norm(mag[i]))

    _plot(mag_angles, 'angles between magnetic field and body frame', 'rad')

for i in range(0,2):
    nc_loop(i/2)