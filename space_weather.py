import numpy as np
import xarray as xr
import re
import datetime as dt
from urllib.request import urlopen

url_last_five_years = 'http://celestrak.com/SpaceData/SW-Last5Years.txt'
url_all = 'http://celestrak.com/SpaceData/SW-All.txt'


def create_space_weather_netcdf(url_string='http://celestrak.com/SpaceData/SW-Last5Years.txt',
                                output_name='cssi_space_weather.nc'):
    # load the data
    data = urlopen(url_string)
    # create empty lists
    date = []
    bsrn = []  # Bartels Solar Rotation Number
    nd = []  # number of days within Bartel 27-day cycle
    kp = []
    kp_sum = []
    ap = []
    ap_avg = []
    cp = []
    c9 = []
    isn = []
    f107_adj = []
    q_flux = []
    ctr81_adj = []  # adj -> adjusted to 1 AU
    lst81_adj = []
    f107_obs = []  # obs -> observed (unadjusted)
    ctr81_obs = []
    lst81_obs = []

    while not (b'BEGIN OBSERVED' in data.readline()):
        pass  # loops until we get to the line where data starts
    line_byte = data.readline()  # grab the first line
    while not (b'END OBSERVED' in line_byte):
        line_str = line_byte.decode('utf-8')
        values = re.findall(r'[\d.]+', line_str)
        if not len(values) == 33:  # some lines near the end are missing values, skip them for now
            # TODO: be smart and don't skip them because in some lines the f107 and ap values are still there
            #   but some other values are missing -- these are lines near the end of the time range
            #   if we wanted to do this the lines would have to be read one character at a time
            #   in order to locate the missing values
            line_byte = data.readline()
            continue
        date.append(dt.datetime(int(values[0]), int(values[1]), int(values[2])))
        bsrn.append(int(values[3]))
        nd.append(int(values[4]))
        kp.append(np.array([int(v) for v in values[5:13]]))
        kp_sum.append(int(values[13]))
        ap.append(np.array([int(v) for v in values[14:22]]))
        ap_avg.append(int(values[22]))
        cp.append(float(values[23]))
        c9.append(int(values[24]))
        isn.append(int(values[25]))
        f107_adj.append(float(values[26]))
        q_flux.append(int(values[27]))
        ctr81_adj.append(float(values[28]))
        lst81_adj.append(float(values[29]))
        f107_obs.append(float(values[30]))
        ctr81_obs.append(float(values[31]))
        lst81_obs.append(float(values[32]))

        line_byte = data.readline()  # read in next line

    # TODO: add in predicted values, if desired
    # while not (b'BEGIN DAILY PREDICTED' in data.readline()):
    #     pass
    # line_byte = data.readline()
    # while not (b'END DAILY PREDICTED' in line_byte):
    #
    #     line_byte = data.readline()

    data.close()
    three_hour_interval = ['0000-0300', '0300-0600', '0600-0900', '0900-1200',
                           '1200-1500', '1500-1800', '1800-2100', '2100-0000']

    # TODO: add units and descriptions to the dateset
    cssi_space_weather = xr.Dataset(
        data_vars={'bsrn': ('date', bsrn),
                   'nd': ('date', nd),
                   'kp': (('date', 'three_hour_interval'), kp),
                   'kp_sum': ('date', kp_sum),
                   'ap': (('date', 'three_hour_interval'), ap),
                   'ap_avg': ('date', ap_avg),
                   'cp': ('date', cp),
                   'c9': ('date', c9),
                   'isn': ('date', isn),
                   'f107_adj': ('date', f107_adj),
                   'q_flux': ('date', q_flux),
                   'ctr81_adj': ('date', ctr81_adj),
                   'lst81_adj': ('date', lst81_adj),
                   'f107_obs': ('date', f107_obs),
                   'ctr81_obs': ('date', ctr81_obs),
                   'lst81_obs': ('date', lst81_obs)},
        coords={'date': date,
                'three_hour_interval': three_hour_interval})

    # make the netcdf file
    cssi_space_weather.to_netcdf(output_name)


if __name__ == '__main__':
    create_space_weather_netcdf(url_last_five_years, 'cssi_space_weather.nc')
