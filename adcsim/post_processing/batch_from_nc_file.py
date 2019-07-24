"""
Chunks of code that can be used for post processing of data after it is saved to a netcdf file
"""
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from adcsim.CubeSat_model_examples import CubeSatModel
from adcsim.animation import AnimateAttitude, DrawingVectors, AdditionalPlots
import os




def from_nc_loop(mag_size, inputfile, ts):
    # data = xr.open_dataset(os.path.join(os.path.dirname(__file__), '../../run0.nc'))
    # filename = '..\\..\\' + '\\magnet_runs\\' + inputfile

    filename = '..\\..\\' + '\\time_step_check\\' + inputfile

    # print(os.path.join(os.path.dirname(__file__), filename))
    data = xr.open_dataset(os.path.join(os.path.dirname(__file__), filename))
    # print(data)
    sim_params = eval(data.simulation_parameters)
    time = np.arange(0, sim_params['end_time_index'], sim_params['time_step'])[:-1]
    le = int(len(time) / sim_params['save_every'])

    # declare a CubeSat (This is only for animations, the cubesat does not need to match the one used in the run,
    # you may want it to look the same tho)
    cubesat = CubeSatModel()


    # output_path = '..\\..\\output_mag_runs\\'

    output_path = '..\\..\\time_step_check\\'


    def _plot(datas, title='', ylabel=''):
        plt.figure()
        plt.plot(time[::sim_params['save_every']], datas)
        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel(ylabel)
        outputfile = output_path + str(mag_size) + '_' + str(ts) + 's_' + title.replace(' ', '_') + '.jpg'
        plt.savefig(os.path.join(os.path.dirname(__file__), outputfile))
        plt.close()

    _plot(data.angular_vel.values, 'angular velocity components', 'angular velocity (rad/s)')
    _plot(np.linalg.norm(data.angular_vel.values, axis=1), 'angular velocity magnitude', 'angular velocity (rad/s)')
    _plot(data.controls.values, 'control torque components', 'Torque (Nm)')
    #
    # Calculate angles between body axis and magnetic field
    mag_angles = np.zeros((le, 3))
    dcm_bn = data.dcm_bn.values
    mag = data.mag.values
    for i in range(le):
        mag_angles[i] = np.arccos(dcm_bn[i] @ mag[i] / np.linalg.norm(mag[i]))

    _plot(mag_angles, 'angles between magnetic field and body frame', 'rad')
    print(str(mag_size) + ' complete.')
    #
    # output_path = '..\\..\\' + '\\output_mag_runs\\' + str(mag_size) + '_magnet' + '\\'
    #
    # ang_vel_comp.plt.savefig(os.path.join(os.path.dirname(__file__), output_path + str(mag_size) + 'vel_components'))

    # # The Animation
    # num = 50
    # start = 0
    # end = -1
    # vec2 = DrawingVectors(data.sun.values[start:end:num], 'single', color='y', label='sun', length=0.5)
    # vec1 = DrawingVectors(data.nadir.values[start:end:num], 'single', color='b', label='nadir', length=0.5)
    # vec3 = DrawingVectors(data.velocities.values[start:end:num], 'single', color='g', label='velocity', length=0.5)
    # vec4 = DrawingVectors(data.mag.values[start:end:num], 'single', color='r', label='magnetic field', length=0.5)
    # ref1 = DrawingVectors(data.dcm_bn.values[start:end:num], 'axes', color=['C0', 'C1', 'C2'],
    #                       label=['Body x', 'Body y', 'Body z'],
    #                       length=0.2)
    # ref2 = DrawingVectors(data.dcm_bo.values[start:end:num], 'axes', color=['C0', 'C1', 'C2'],
    #                       label=['Body x', 'Body y', 'Body z'],
    #                       length=0.2)
    # plot1 = AdditionalPlots(time[start:end:num], data.controls.values[start:end:num], labels=['X', 'Y', 'Z'])
    # plot2 = AdditionalPlots(data.lons.values[start:end:num], data.lats.values[start:end:num], groundtrack=True)
    # a = AnimateAttitude(data.dcm_bn.values[start:end:num], draw_vector=[ref1, vec4, vec1], additional_plots=plot2,
    #                     cubesat_model=cubesat)
    # a.animate_and_plot()

    # plt.show()


# from_nc_loop(10.0, '10.0_G_magnet.nc')
for i in range(11, 20):
    in_file = str(i/2) + '_G_0.01_ts.nc'
    from_nc_loop(i/2, in_file, 0.01)
    in_file = str(i / 2) + '_G_0.001_ts.nc'
    from_nc_loop(i / 2, in_file, 0.001)


# from_nc_loop(10.0, '10.0_G_0.01_ts.nc', 0.01)
# from_nc_loop(10.0, '10.0_G_0.001_ts.nc',0.001)

# for i in range(0,20):
#     from_nc_loop(i/2, str(i/2) + '_G_magnet.nc')