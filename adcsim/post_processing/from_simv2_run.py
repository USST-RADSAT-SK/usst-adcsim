"""
Chunks of code that can be used for post processing of data after it is ran in simv2.
"""
import matplotlib.pyplot as plt
from adcsim.animation import AnimateAttitude, DrawingVectors, AdditionalPlots

omegas = states[:, 1]
sigmas = states[:, 0]

def _plot(datas, title='', ylabel=''):
    plt.figure()
    plt.plot(time[::save_every], datas)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel(ylabel)


_plot(omegas, 'angular velocity components', 'angular velocity (rad/s)')
_plot(np.linalg.norm(omegas, axis=1), 'angular velocity magnitude', 'angular velocity (rad/s)')
_plot(controls, 'control torque components', 'Torque (Nm)')

# Calculate angles between body axis and magnetic field
mag_angles = np.zeros((le, 3))
for i in range(le):
    mag_angles[i] = np.arccos(dcm_bn[i] @ mag_field[i] / np.linalg.norm(mag_field[i]))

_plot(mag_angles, 'angles between magnetic field and body frame', 'rad')

# The Animation
num = 10
start = 0
end = -1
vec2 = DrawingVectors(sun_vec[start:end:num], 'single', color='y', label='sun', length=0.5)
vec3 = DrawingVectors(velocities[start:end:num], 'single', color='g', label='velocity', length=0.5)
vec4 = DrawingVectors(mag_field[start:end:num], 'single', color='r', label='magnetic field', length=0.5)
ref1 = DrawingVectors(dcm_bn[start:end:num], 'axes', color=['C0', 'C1', 'C2'], label=['Body x', 'Body y', 'Body z'],
                      length=0.2)
ref2 = DrawingVectors(dcm_bo[start:end:num], 'axes', color=['C0', 'C1', 'C2'], label=['Body x', 'Body y', 'Body z'],
                      length=0.2)
plot1 = AdditionalPlots(time[start:end:num], controls[start:end:num], labels=['X', 'Y', 'Z'])
plot2 = AdditionalPlots(lons[start:end:num], lats[start:end:num], groundtrack=True)
a = AnimateAttitude(dcm_bn[start:end:num], draw_vector=[ref1, vec4], additional_plots=plot2,
                    cubesat_model=cubesat)
a.animate_and_plot()

plt.show()
