from distutils.core import setup

setup(
    name='U-of_Colorado_course',
    version='0.1dev',
    description='ADCS simulations for the USST CubeSat team',
    url='https://github.com/usst-adcs/U-of-Colorado_course',
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    packages=['adcsim', 'adcsim/Python_NRLMSISE', 'adcsim/dcm_convert',],
    long_description=open('readme.md').read(),
)