"""
The code in this file loads space weather information (which is needed to calculate atmospheric densities) from
celestrak's website and saves it to a netcdf file.
"""


import numpy as np
import xarray as xr
import re
import datetime as dt
from urllib.request import urlopen

url_last_five_years = 'http://celestrak.com/SpaceData/SW-Last5Years.txt'
url_all = 'http://celestrak.com/SpaceData/SW-All.txt'


def create_space_weather_netcdf(url_string='http://celestrak.com/SpaceData/SW-Last5Years.txt',
                                output_name='cssi_space_weather.nc'):
    """
    Create a netcdf file of space weather information. This data is needed to calculate atmospheric densities.
    :param url_string: URL to retrieve the data from
    :param output_name: Name of the saved file
    :return: None
    """
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

    datas = [[] for i in range(33)]

    while not (b'BEGIN OBSERVED' in data.readline()):
        pass  # loops until we get to the line where data starts
    line_byte = data.readline()  # grab the first line

    # get starting indexes for the first line
    line_str = line_byte.decode('utf-8')
    p = re.compile("[\d.] ")
    indexes = []
    for m in p.finditer(line_str):
        indexes.append(m.start())
    indexes.append(129)

    while not (b'END OBSERVED' in line_byte):
        line_str = line_byte.decode('utf-8')
        values = re.findall(r'[\d.]+', line_str)

        k = 0
        for i, val in enumerate(indexes):
            if line_str[val] == ' ':
                datas[i].append(' ')
                continue
            datas[i].append(values[k])
            k += 1

        line_byte = data.readline()  # read in next line

    # TODO: add in predicted values, if desired
    # while not (b'BEGIN DAILY PREDICTED' in data.readline()):
    #     pass
    # line_byte = data.readline()
    # while not (b'END DAILY PREDICTED' in line_byte):
    #
    #     line_byte = data.readline()

    for i in range(len(datas[0])):
        date.append(dt.datetime(int(datas[0][i]), int(datas[1][i]), int(datas[2][i])))
        bsrn.append(int(datas[3][i]))
        nd.append(int(datas[4][i]))
        kp.append(np.array([int(v[i]) for v in datas[5:13]]))
        kp_sum.append(int(datas[13][i]))
        ap.append(np.array([int(v[i]) for v in datas[14:22]]))
        ap_avg.append(int(datas[22][i]))
        if datas[23][i] == ' ':
            cp.append(np.nan)
        else:
            cp.append(float(datas[23][i]))
        if datas[24][i] == ' ':
            c9.append(np.nan)
        else:
            c9.append(int(datas[24][i]))
        if datas[25][i] == ' ':
            isn.append(np.nan)
        else:
            isn.append(int(datas[25][i]))
        f107_adj.append(float(datas[26][i]))
        q_flux.append(int(datas[27][i]))
        ctr81_adj.append(float(datas[28][i]))
        lst81_adj.append(float(datas[29][i]))
        f107_obs.append(float(datas[30][i]))
        ctr81_obs.append(float(datas[31][i]))
        lst81_obs.append(float(datas[32][i]))


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
