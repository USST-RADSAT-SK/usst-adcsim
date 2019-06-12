## functions to convert a dcm array to a stk .a file
completely dependant on simv2.py file and format

dcm.py: numpy binary storage of dcm_bn array from adcim\simulations\simv2.py
dcm_to_stk_a.py: functions to convert dcm_bn array to stk .a file from simv2.py
foo_01.a: stk .a file from dcm_to_stk_simple()
foo_02.a: stk .a file from xdcm_to_stk()
quaternions.py: numpy binary storage of dcm_bn array converted to quaternions
stk_xarray_conversion.py: given file for stk <-> xarray conversions from python-attitude-simulations

NOTE: I have no idea if this works since I have no knowledge of STK or the actual file requirements