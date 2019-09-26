"""
Chunks of code that can be used for post processing of data after it is saved to a netcdf file
"""
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from adcsim.CubeSat_model import CubeSat
from adcsim.CubeSat_model_examples import CubeSatModel
from adcsim.hysteresis_rod import HysteresisRod
from adcsim.animation import AnimateAttitude, DrawingVectors, AdditionalPlots
import os
####################################
in_file = '../../run1.nc'  # name and location of input .nc file; default is '../../run0.c'
save_graphs = False  # change to true if you want to save copies of various graphs; default False
output_folder = '../../'  # change to save graphs to another folder; default is adcsim folder: '../../'
                          # 'output_folder = '../../from_nc_graphs/' is an example
display_graphs = True  # change to False if you are opening multiple files in a loop; default True
display_animation = False  # change to false if you don't want to see animation; default True
animation_speed = 50  # how many seconds per frame of animation; default is 50
####################################

# load data from the run
data = xr.open_dataset(os.path.join(os.path.dirname(__file__), in_file))
sim_params = eval(data.simulation_parameters)
more_params = eval(data.cubesat_parameters)
time = np.arange(0, sim_params['duration'], sim_params['time_step'])
le = int(len(time)/sim_params['save_every'])
time = time[::sim_params['save_every']]

if save_graphs:
    output_prefix = f'{output_folder}{more_params["magnetic_moment"][-1]:.2f}G_'  # fancy filename
    if not os.path.exists(os.path.join(os.path.dirname(__file__), output_folder)):
        os.makedirs(os.path.join(os.path.dirname(__file__), output_folder))

# load the cubesat
try:
    cubesat = CubeSat.fromdict(data.cubesat_parameters)
except:  # if people are using old simulation .nc files then the above line may not work for them
    cubesat = CubeSatModel()

if save_graphs:
    print(f'Saving {more_params["magnetic_moment"][-1]:.2f}G.')

# can now recreate the rods easily
rods = HysteresisRod.from_cubesat_parameters_data(data.cubesat_parameters, data.hyst_rod_external_field, data.hyst_rod_magnetization)
for r, rod in enumerate(rods):
    rod.plot_limiting_cycle()
    if save_graphs:
        plt.savefig(f'{output_prefix}hyst_rod_{r}.jpg')
    if not display_graphs:
        plt.close()


def _plot(datas, title='', ylabel=''):
    plt.figure()
    plt.plot(time, datas)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel(ylabel)
    if save_graphs:
        plt.savefig(f'{output_prefix}{title.replace(" ", "_")}.jpg')
    if not display_graphs:
        print('\t' + title + " done.")
        plt.close()


_plot(data.angular_vel.values, 'angular velocity components', 'angular velocity (rad/s)')
_plot(np.linalg.norm(data.angular_vel.values, axis=1), 'angular velocity magnitude', 'angular velocity (rad/s)')
_plot(data.controls.values, 'control torque components', 'Torque (Nm)')

plt.figure()
plt.plot(time, np.linalg.norm(data.magnetic_torque.values, axis=1), label='magnetic')
plt.plot(time, np.linalg.norm(np.sum(data.hyst_rod_torque.values, axis=1), axis=1), label='hysteresis')
plt.plot(time, np.linalg.norm(data.gg_torque, axis=1), label='gravity')
plt.plot(time, np.linalg.norm(data.aero_torque, axis=1), label='aerodynamic')
plt.plot(time, np.linalg.norm(data.solar_torque, axis=1), label='solar')
plt.yscale('log')
plt.legend()

# Calculate angles between body axis and magnetic field
mag_angles = np.zeros((le, 3))
dcm_bn = data.dcm_bn.values
mag = data.mag.values

if sim_params['save_every'] == 1:
    for i in range(le - 1):
        mag_angles[i] = np.arccos(dcm_bn[i] @ mag[i] / np.linalg.norm(mag[i]))
    mag_angles[-1] = mag_angles[-2]
else:
    for i in range(le):
        mag_angles[i] = np.arccos(dcm_bn[i] @ mag[i] / np.linalg.norm(mag[i]))

_plot(mag_angles, 'angles between magnetic field and body frame', 'rad')

# The Animation
if display_animation:
    num = animation_speed
    start = 0
    end = -1
    vec2 = DrawingVectors(data.sun.values[start:end:num], 'single', color='y', label='sun', length=0.5)
    vec1 = DrawingVectors(data.nadir.values[start:end:num], 'single', color='b', label='nadir', length=0.5)
    vec3 = DrawingVectors(data.velocities.values[start:end:num], 'single', color='g', label='velocity', length=0.5)
    vec4 = DrawingVectors(data.mag.values[start:end:num], 'single', color='r', label='magnetic field', length=0.5)
    ref1 = DrawingVectors(data.dcm_bn.values[start:end:num], 'axes', color=['C0', 'C1', 'C2'], label=['Body x', 'Body y', 'Body z'], length=0.2)
    ref2 = DrawingVectors(data.dcm_bo.values[start:end:num], 'axes', color=['C0', 'C1', 'C2'], label=['Body x', 'Body y', 'Body z'], length=0.2)
    plot1 = AdditionalPlots(time[start:end:num], data.controls.values[start:end:num], labels=['X', 'Y', 'Z'])
    plot2 = AdditionalPlots(data.lons.values[start:end:num], data.lats.values[start:end:num], groundtrack=True)
    plot3 = AdditionalPlots(data.hyst_rod_external_field.values[:, 1][start:end:num], data.hyst_rod_magnetization.values[:, 1][start:end:num], hyst_curve=rods[1])
    a = AnimateAttitude(data.dcm_bn.values[start:end:num], draw_vector=[ref1, vec4, vec1], additional_plots=[plot2, plot3],
                        cubesat_model=cubesat)
    a.animate_and_plot()

plt.show()
