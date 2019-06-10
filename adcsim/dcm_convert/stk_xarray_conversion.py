import numpy as np
import xarray as xr
import re

def stk_to_xarray(path_to_stk_file: str):
    """
    Converts an attitude (.a) or ephemeris (.e) file from STK into an xarray dataset.
    """

    supported_attitude_formats = ['AttitudeTimeQuatAngVels', 'AttitudeTimeQuaternions']
    supported_ephemeris_formats = ['EphemerisTimePosVel']

    supported_extra_data = ['SegmentBoundaryTimes']

    metadata_re = {
        'version': '(stk\.v\.\d+\.\d+)\s*\n',
        'WrittenBy': '# WrittenBy\s+(STK_v\d+\.\d+\.\d+)\s*\n',
        'NumberOfAttitudePoints': 'NumberOfAttitudePoints\s+(\d+)\s*\n',
        'NumberOfEphemerisPoints': 'NumberOfEphemerisPoints\s+(\d+)\s*\n',
        'BlockingFactor': 'BlockingFactor\s+(\d+)\s*\n',
        'InterpolationOrder': 'InterpolationOrder\s+(\d+)\s*\n',
        'InterpolationMethod': 'InterpolationMethod\s+(\w+)\s*\n',
        'InterpolationSamplesM1': 'InterpolationSamplesM1\s+(\d+)\s*\n',
        'CentralBody': 'CentralBody\s+(\w+)\s*\n',
        'ScenarioEpoch': 'ScenarioEpoch\s+(.+?)\s*\n',
        'Epoch in JDate format': '# Epoch in JDate format:\s+(\d+\.\d+)\s*\n',
        'Epoch in YYDDD format': '# Epoch in YYDDD format:\s+(\d+\.\d+)\s*\n',
        'Time of first point': '# Time of first point:\s*(.*?)\s*\n',
        'CoordinateAxes': 'CoordinateAxes\s+(\w+)\s*\n',
        'CoordinateSystem': 'CoordinateSystem\s+(\w+)\s*\n',
    }

    attrs = {}
    section = []  # contains what section(s) we are in, specified by the BEGIN and END keywords
    with open(path_to_stk_file, 'r') as file:

        read_data = False
        extra_data = {}  # read extra data like SegmentBoundaryTimes into a list, convert to array when done
        while not read_data:
            metadata = ''
            for line in file:
                # check for BEGIN-END sections
                m = re.match('\s*BEGIN\s+(\w+)\s*\n', line)
                if m is not None:
                    match = m.group(1)
                    section.append(match)
                    if match in supported_extra_data:
                        extra_data[match] = []
                    continue
                m = re.match('\s*END\s+(\w+)\s*\n', line)
                if m is not None:
                    match = m.group(1)
                    if match != section[-1]:
                        raise Warning(f'Mismatched BEGIN/END statements {match}')
                    if match in supported_extra_data:
                        extra_data[match] = np.array(extra_data[match])
                    section.pop(-1)
                    continue

                # check for known extra data keywords
                if len(section) > 0 and section[-1] == 'SegmentBoundaryTimes':
                    if len(line.strip()) > 0:
                        extra_data['SegmentBoundaryTimes'].append(float(line.strip()))
                    continue

                # check for the start of the main data
                if any([f == line.strip() for f in supported_attitude_formats + supported_ephemeris_formats]):
                    attrs['format'] = line.strip()
                    read_data = True
                    break

                # sanity check: we shouldn't reach any lines that are all numbers without triggering something else
                if is_dataline(line):  # if it's data, we missed the format string
                    raise ValueError(f'Unknown attitude format')

                # if we made it here and the line isn't empty this should be metadata or a comment
                if len(line.split()) > 0:
                    metadata += line

        # parse the metadata
        for key, value in metadata_re.items():
            m = re.search(value, metadata)
            if m is not None:
                attrs[key] = m.group(1)

        if attrs['format'] == 'AttitudeTimeQuatAngVels':
            data = np.loadtxt(file, dtype=float, ndmin=2, comments=['END'])
            coords = {'time': ('time', data[:, 0])}
            data_vars = {
                'quaternions': (['time', 'q'], data[:, 1:5]),
                'angular_velocities': (['time', 'xyz'], data[:, 5:])
            }
            ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
        elif attrs['format'] == 'AttitudeTimeQuaternions':
            data = np.loadtxt(file, dtype=float, ndmin=2, comments=['END'])
            coords = {'time': ('time', data[:, 0])}
            data_vars = {'quaternions': (['time', 'q'], data[:, 1:])}
            ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
        elif attrs['format'] == 'EphemerisTimePosVel':
            data = np.loadtxt(file, dtype=float, ndmin=2, comments=['END'])
            ds = []
            istart = 0
            for i in range(len(extra_data['SegmentBoundaryTimes'])):
                # find the ending index for the current segment
                if i == len(extra_data['SegmentBoundaryTimes']) - 1:
                    iend = data.shape[0]
                else:
                    iend = np.where(data[:, 0] == extra_data['SegmentBoundaryTimes'][i + 1])[0][0]
                coords = {'time': ('time', data[istart:iend, 0])}
                data_vars = {
                    'positions': (['time', 'xyz'], data[istart:iend, 1:4]),
                    'velocities': (['time', 'xyz'], data[istart:iend, 4:])
                }
                for extra in extra_data:
                    data_vars[extra] = (extra, extra_data[extra])
                ds.append(xr.Dataset(data_vars=data_vars, coords=coords))
                istart = iend + 1
            ds = xr.merge(ds)
            ds.attrs = attrs
        else:
            raise ValueError(f'Unknown format {attrs["format"]}')

        return ds


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




def is_dataline(line):
    # assumes this is a dataline if every whitespace-separated token can be turned to a float
    tokens = line.split()
    if len(tokens) == 0:
        return False
    try:
        for token in tokens:
            float(token)
        return True
    except ValueError:
        return False


if __name__ == '__main__':
    filename = 'CubeSatICRF.e'
    ds = stk_to_xarray(f'data//{filename}')
    print(ds)
    ds.to_netcdf(f'data//{filename[:-1]}nc')