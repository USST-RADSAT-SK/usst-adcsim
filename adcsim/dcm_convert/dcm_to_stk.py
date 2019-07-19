import numpy as np
from adcsim.transformations import dcm_to_quaternions
import xarray as xr
from adcsim.dcm_convert.stk_xarray_conversion import xarray_to_stk

# add to simv.py
# from adcsim.dcm_convert.dcm_to_stk_a import xdcm_to_stk

# and add after a.to_netcdf('run5.nc')
# xdcm_to_stk(time[::save_every], dcm_bn, "foo.a")
# or
# dcm_to_stk_simple(time[::save_every], dcm_bn, "foo.a")



def dcm_to_stk_simple(t_time, dcm, outfile):
    # a quick and dirty function to convert dcm to stk .a file

    assert(len(t_time) == len(dcm)), "unequal arrays!"

    version = "stk.v.11.0"
    NumberOfAttitudePoints = len(dcm)
    BlockingFactor = 20
    InterpolationOrder = 1
    CentralBody = "Earth"
    ScenarioEpoch = "24 Mar 2019 18:35:01.000000"
    CoordinateAxes = "ICRF"
    format = 'AttitudeTimeQuaternions'

    meta_lines = [
        f'{version}\n',
        f'BEGIN Attitude\n',
        f'NumberOfAttitudePoints {NumberOfAttitudePoints}\n',
        f'BlockingFactor {BlockingFactor}\n',
        f'InterpolationOrder {InterpolationOrder}\n',
        f'CentralBody {CentralBody}\n',
        f'ScenarioEpoch {ScenarioEpoch}\n',
        f'CoordinateAxes {CoordinateAxes}\n',
        f'{format}\n',
    ]

    header = ''.join(meta_lines)
    footer = '\nEND Attitude'

    out = np.zeros((len(t_time), 5))
    for i, dcm_line in enumerate(dcm):
        quat = dcm_to_quaternions(dcm_line)
        out[i] = [t_time[i], quat[1], quat[2], quat[3], quat[0]]  # note: STK puts the scalar quaternion last, not first

    np.savetxt(outfile, out, fmt='%.16e', comments='', header=header, footer=footer)


def xdcm_to_stk(t_time, dcm, outfile):  # convert dcm using xarray conversions
    assert (len(t_time) == len(dcm)), "unequal arrays!"

    quat = np.zeros((len(dcm), 4))

    for i, nums in enumerate(dcm):
        quat[i] = dcm_to_quaternions(nums)

    x_dcm = xr.Dataset({"time": (['s'], t_time),
                       "quaternions": (['a', 'b'], quat)},
                       attrs={"version": "stk.v.11.0",
                             "NumberOfAttitudePoints": len(quat),
                             "BlockingFactor": 20,
                             "InterpolationOrder": 1,
                             "CentralBody": "Earth",
                             "ScenarioEpoch": "24 Mar 2019 18:35:01.000000",
                             "CoordinateAxes": "ICRF",
                             "format": "AttitudeTimeQuaternions"}
                       )

    xarray_to_stk(x_dcm, outfile)


if __name__ == "__main__":

    save_every = 10  # only save the data every number of iterations
    # declare time step for integration
    time_step = 0.01
    end_time = 1000
    time = np.arange(0, end_time, time_step)

    data = np.load("dcm.npy", 'r')

    dcm_to_stk_simple(time[::save_every], data, "foo_01.a")
    xdcm_to_stk(time[::save_every], data, "foo_02.a")
