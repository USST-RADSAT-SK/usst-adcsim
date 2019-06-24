# USST ADCS simulation software
This project is an an attitude propagator made in-house by the USST. The project allows for the simulation of various 
ADCS designs, so that their behaviour and performance can be analyzed.

## U-of-Colarado_course
This effort started with taking a online course offered by the University of Colorado. The required mechanics theory
is given in this course, and the instructor gives explicit instructions for how to go about 
implementing attitude propagation software.

#### The course can be found at:
https://www.coursera.org/specializations/spacecraft-dynamics-control
#### The course slides can be found in the dropbox at:
/home/CubeSat/Subteams/Attitude Control/Attitude Control Lead/Learning_materials/UofColorado_attitude_course

## Setup Instructions
These setup instructions require Anaconda and git to be installed on your computer (no need for Pycharm).

Open up the 'Anaconda Prompt' program (or your favorite command line program that has Anaconda). Then create a python environment with:

```$ conda create -n your_env_name python=3.7```

Activate the python environement you just created with:

```$ activate your_env_name```

Install all dependencies with two commands:

```$ conda install numpy pandas xarray matplotlib astropy scipy cartopy```

```$ pip install skyfield tqdm```

Then, ensure that you have git lfs installed with:

```$ git lfs install```

clone this projects github repository with:

```$ git clone https://github.com/usst-adcs/U-of-Colorado_course.git```

change directories to U-of-Colorado_course and do:

```$ pip install -e .```

You should now be able to run the best version of the simulation script (simv2.py) with:

```$ python adcsim\simulations\simv2.py``` 

And observe the results of the simulation by running the post processing script:

```$ python adcsim\post_processing\from_nc_file.py```

To change the parameters of the simulation you can edit the two files in a text editor and run them through the command line with the above commands or use the Pycharm IDE if you wish. 


## Changing the parameters of the simulation

First, a list of parameters:

legend: Parameter name in the code; parameter description; units of parameter (if applicable); additional notes (if applicable)
* time_step; the numerical integration time step for the integration of the attitude; seconds; The time_step must be 
slow enough for the given simulation, otherwise the solution can make no sense. It varies from simulation to simulation.
* end_time; the ending time of the simulation; seconds; the start time is hardcoded to be 0
* omega0; the initial angular velocity in the inertial frame; rad/s 
* sigma0; the initial attitude of the CubeSat represented with MRP attitude coordinates
* Parameters of the CubeSat model. There are many different models that could be created, and this can be done by 
creating a class in the 'CubeSat_model_examples.py" file that inherits from the 'CubeSat' class. There are a few examples in this 
file. The CubeSatAerodynamicEx1 model is the model that is up to date with our current best knowledge of the satellite.
The common user with likely just use this one.
    * center_of_mass; the center of mass of the CubeSat (array of length 3 specifying the distance from the center of 
    geometry and the center of mass in each axis); meters; This should be close to zero for our final design.
    * inertia; The 3x3 inertia matrix; kg*m^2; This property has a huge impact on the attitude of the CubeSat. 
    This is actually the only parameter we change for the gravity gradient design.
    * magnetic_moment; the magnetic moment of the permanent magnetic on the CubeSat (array of length 3 specifying 
    the strength of the magnetic moment in each axes); A*m^2
    * residual_magnetic_moment; the magnetic moment from unexpected sources on the CubeSat (e.g. electronics) 
    (array of length 3 specifying the strength of the magnetic moment in each axes); A*m^2; This parameter can be set to 
    zero for now, but down the road we may want to investigate how unexpected magnetic moments can effect the attitude.
    * Parameters of the hysteresis rods
        * br, bs, and hc; These three parameters fully define the hysteresis curve of the material 
        (for explaination see: /home/CubeSat/Subteams/Attitude Control/Attitude Control Lead/Learning_materials/Literature/HysteresisRods/passive_magnetic_design_intro.pdf)
        * volume; the volume of the rod; meters^3, The larger the volume the greater the damping effect the rods will have
        * axes_alignment; the axes of the CubeSat body that the hysteresis rod is lined up with; The approximation our code 
        as others codes makes is that the rods can only be magnetized along this axes (that is actually why they are made to be rods I think).

All of these parameters declarations can be found and changed in the simv2.py script somewhere.

A list of additional things a user could change:

* The data that is being saved to the netcdf file. E.g. the user might wish to save each disturbance torque individually 
so that they can use them in post processing, rather than just saving the sum of them. This would be done by editing
the creation of the Dataset before it is saved.
* Only a few elements of the simulation are plotted in the post_processing script. So, the user should feel free to add 
and comment out blocks of code in this script as they need, or write their own entirely new post processing scripts. 
E.g. Someone may be interested in looking at solar power generation. 
* The animation can be changed to suit your needs

### changing the animation:

* the parameter 'start' changes what index of the simulation the matplotlib animation will begin at.
* the parameter 'end' changes what index of the simulation the matplotlib animation will end at
* the parameter 'num' will change how fast the animation goes (faster with a higher number).
* You will notice that there are a number of arrows in the 3D animation window (e.g. the body frame axes, the magnetic 
field). Additional arrows can be added (e.g. the velocity vector direction) by creating an instance of the 
DrawingVectors class and then passing this instance to the 'draw_vector' keyword when creating an instance of the 
AnimateAttitude class. You should be able to see this and figure out how to create the DrawingVectors instances from the code that
is already in from_nc_file.py.
* 2D plots animated along side the 3D animation can also be added (e.g. the ground track). This can be done by 
creating an instance of the AdditionalPlots class and passing them to the 'additional_plots' keyword when creating an 
instance of the AnimateAttitude class. Again, You should be able to see this and figure out how to create the AnimateAttitude 
instances from the code that is already in from_nc_file.py.


#### To install the pysofa dependency on windows:

This is needed to run the pre_process_orbit.py scipt. Most users with not need to do this.

```$ git clone https://github.com/usst-adcs/pysofa-compiled.git```

and then:

```$ pip install .```

when you are inside the directory of the pysofa repo.