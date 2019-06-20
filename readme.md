# USST ADCS simulation software
This project is an an attitude propagator made in-house by the USST. The project allows for the simulation of various 
ADCS designs, so that their behaviour and performance can be analyzed.

## U-of-Colarado_course
This effort started with taking a online course offered by the University of Colorado. The required mechanics theory
is given in this course, and the instruction gives explicit instructions for how to go about 
implementing attitude propagation software.

#### The course can be found at:
https://www.coursera.org/specializations/spacecraft-dynamics-control
#### The course slides can be found in the dropbox at:
/home/CubeSat/Subteams/Attitude Control/Attitude Control Lead/Learning_materials/UofColorado_attitude_course

## Installing Dependances:
```conda install numpy pandas xarray matplotlib astropy```

```pip install skyfield tqdm```

#### To install the pysofa dependency on windows:

```git clone https://github.com/usst-adcs/pysofa-compiled.git```

and then:

```pip install .```

when you are inside the directory of the pysofa repo.