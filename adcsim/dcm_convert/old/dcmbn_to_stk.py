import numpy as np
from adcsim.transformations import dcm_to_quaternions
import xarray as xr


def xarray_to_stk(dataset: xr.Dataset, path_to_stk_file: str):
    """
    Converts an xarray dataset back into an STK attitude (.a) or ephemeris (.e) file.
    """
    if dataset.attrs['format'] == 'AttitudeTimeQuatAngVels':
        meta_lines = [
            f'{dataset.attrs["version"]}\n',
            f'BEGIN Attitude\n',
            f'NumberOfAttitudePoints {dataset.attrs["NumberOfAttitudePoints"]}\n',
            f'BlockingFactor {dataset.attrs["BlockingFactor"]}\n',
            f'InterpolationOrder {dataset.attrs["InterpolationOrder"]}\n',
            f'CentralBody {dataset.attrs["CentralBody"]}\n',
            f'ScenarioEpoch {dataset.attrs["ScenarioEpoch"]}\n',
            f'CoordinateAxes {dataset.attrs["CoordinateAxes"]}\n',
            f'{dataset.attrs["format"]}\n',
        ]

        header = ''.join(meta_lines)
        footer = '\nEND Attitude'
        data = np.column_stack((dataset.time.values, dataset.quaternions.values, dataset.angular_velocities.values))
        np.savetxt(path_to_stk_file, data, fmt='%.16e', comments='', header=header, footer=footer)
    elif dataset.attrs['format'] == 'AttitudeTimeQuaternions':
        meta_lines = [
            f'{dataset.attrs["version"]}\n',
            f'BEGIN Attitude\n',
            f'NumberOfAttitudePoints {dataset.attrs["NumberOfAttitudePoints"]}\n',
            f'BlockingFactor {dataset.attrs["BlockingFactor"]}\n',
            f'InterpolationOrder {dataset.attrs["InterpolationOrder"]}\n',
            f'CentralBody {dataset.attrs["CentralBody"]}\n',
            f'ScenarioEpoch {dataset.attrs["ScenarioEpoch"]}\n',
            f'CoordinateAxes {dataset.attrs["CoordinateAxes"]}\n',
            f'{dataset.attrs["format"]}\n',
        ]

        header = ''.join(meta_lines)
        footer = '\nEND Attitude'
        data = np.column_stack((dataset.time.values, dataset.quaternions.values))
        np.savetxt(path_to_stk_file, data, fmt='%.16e', comments='', header=header, footer=footer)
    elif dataset.attrs['format'] == 'EphemerisTimePosVel':
        meta_lines = [
            f'{dataset.attrs["version"]}\n',
            f'BEGIN Ephemeris\n',
            f'NumberOfEphemerisPoints {dataset.attrs["NumberOfEphemerisPoints"]}\n',
            f'ScenarioEpoch {dataset.attrs["ScenarioEpoch"]}\n',
            f'InterpolationMethod {dataset.attrs["InterpolationMethod"]}\n',
            f'InterpolationSamplesM1 {dataset.attrs["InterpolationSamplesM1"]}\n',
            f'CentralBody {dataset.attrs["CentralBody"]}\n',
            f'CoordinateSystem {dataset.attrs["CoordinateSystem"]}\n',
            f'Begin SegmentBoundaryTimes\n'
        ]

        for i in range(dataset.dims['SegmentBoundaryTimes']):
            meta_lines.append(f'{dataset.SegmentBoundaryTimes.isel(SegmentBoundaryTimes=i).item():.16e}\n')

        meta_lines += [
            f'END SegmentBoundaryTimes\n',
            f'{dataset.attrs["format"]}\n'
        ]

        header = ''.join(meta_lines)
        footer = '\nEND Ephemeris'
        data = np.column_stack((dataset.time.values, dataset.positions.values, dataset.velocities.values))
        np.savetxt(path_to_stk_file, data, fmt='%.16e', comments='', header=header, footer=footer)
    else:
        raise ValueError(f'Unknown format {dataset.attrs["format"]}')


# converts a dcm to a quaternion export file for stk
def export_stk_file(dataset, out_file):

    # declare time step for integration
    time_step = 0.01
    end_time = 1000
    ttime = np.arange(0, end_time, time_step)
    ctime = np.array([ttime]).T

    quat = np.zeros((len(dataset), 4))

    for i, data in enumerate(dataset):
        quat[i] = dcm_to_quaternions(data)

    # np.savetxt("time_steps.txt", ttime)

    xdcm = xr.Dataset({"time": (['s'], ctime),
                       "quaternions": (['a', 'b'], quat)},
                      attrs={"version": "stk.v.11.0",
                             "NumberOfAttitudePoints": len(quat),
                             "BlockingFactor": 20,
                             "InterpolationOrder": 1,
                             "CentralBody": "Earth",
                             "ScenarioEpoch": "1 Jan 2019 18:00:00.000000",
                             "CoordinateAxes": "Fixed",
                             "format": "AttitudeTimeQuaternions"}
                     )
    xdcm["time"].to_index()
    # print(ctime)
    # print(xdcm.quaternions.values)

    xarray_to_stk(xdcm, "stk_file_001")


if __name__ == "__main__":

    save_every = 10  # only save the data every number of iterations

    # declare time step for integration
    time_step = 0.01
    end_time = 1000
    time = np.arange(0, end_time, time_step)
    ttime = time[::10]
    data = np.load("dcm.npy", 'r')
    export_stk_file(data, "quaternions.txt")

    # with open("temp.txt", 'w') as f:
    #     for i in range(20):
    #         f.write(str([i]))
    #
    # with open("temp.txt", 'r') as f:
    #     llist = []
    #     for line in f:
    #         llist.append(line.rstrip().strip())
    #     print(llist)

    # with open("dcm_data.txt", 'r') as f:
    #     data_list = []
    #     data = f.readline()
    #     data.rstrip()
    #     print(data)
        # for line in f:
        #     temp = line.rstrip()
        #     data_list.append(temp)
        #     print(data_list)
        # print(data_list[::10])
        # export_stk_file(data, "sim_txt_100.txt")
    # np.save("quaternions", quat)

    # meta_lines = [
    #     f'{dataset.attrs["version"]}\n',
    #     f'BEGIN Attitude\n',
    #     f'NumberOfAttitudePoints {dataset.attrs["NumberOfAttitudePoints"]}\n',
    #     f'BlockingFactor {dataset.attrs["BlockingFactor"]}\n',
    #     f'InterpolationOrder {dataset.attrs["InterpolationOrder"]}\n',
    #     f'CentralBody {dataset.attrs["CentralBody"]}\n',
    #     f'ScenarioEpoch {dataset.attrs["ScenarioEpoch"]}\n',
    #     f'CoordinateAxes {dataset.attrs["CoordinateAxes"]}\n',
    #     f'{dataset.attrs["format"]}\n',
    # ]

    # coords = {'lon': (['x', 'y'], lon),
    #           'lat': (['x', 'y'], lat),
    #            time': pd.date_range('2014-09-06', periods=3),
    #            reference_time': pd.Timestamp('2014-09-05')}
