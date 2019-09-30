import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from adcsim.CubeSat_model import CubeSat
from adcsim.CubeSat_model_examples import CubeSatModel
from adcsim.hysteresis_rod import HysteresisRod
from adcsim.animation import AnimateAttitude, DrawingVectors, AdditionalPlots
import os


if __name__ == '__main__':
    # Get the two datasets to compare
    in_file1 = '../../sim1.nc'
    in_file2 = '../../sim2.nc'
    data1 = xr.open_dataset(os.path.join(os.path.dirname(__file__), in_file1))
    data2 = xr.open_dataset(os.path.join(os.path.dirname(__file__), in_file2))

    assert data1.dims == data2.dims
    assert eval(data1.simulation_parameters) == eval(data2.simulation_parameters)

    sim_params = eval(data1.simulation_parameters)
    more_params = eval(data1.cubesat_parameters)
    time = np.arange(0, sim_params['duration'], sim_params['time_step'])
    time = time[::sim_params['save_every']]

    diff = data2 - data1
    for key in diff.data_vars:
        dims = diff[key].shape
        if len(dims) == 1:
            plt.figure()
            plt.title('data2 Compared to data1, ' + key)
            plt.plot(time, np.divide(diff[key], data1[key]) * 100)
            plt.xlabel('Index')
            plt.ylabel('(data2 - data1) / data1')
        elif len(dims) == 2:
            for i in range(dims[1]):
                plt.figure()
                plt.title('data2 Compared to data1, ' + key + ', ' + diff[key].dims[1] + ' ' + i)
                plt.plot(time, np.divide(diff[key][:, i], data1[key][:, i]) * 100)
                plt.xlabel('Index')
                plt.ylabel('(data2 - data1) / data1')
        elif len(dims) == 3:
            for i in range(dims[1]):
                for j in range(dims[2]):
                    plt.figure()
                    title_str = \
                        'data2 Compared to data1, ' + key + ',\n' + diff[key].dims[1] + '=' + str(i) + ', ' \
                        + diff[key].dims[2] + '=' + str(j)
                    plt.title(title_str)
                    plt.plot(time, np.divide(diff[key][:, i, j], data1[key][:, i, j]) * 100)
                    plt.xlabel('Time (s)')
                    plt.ylabel('(data2 - data1) / data1 (%)')
