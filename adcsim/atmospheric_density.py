import xarray as xr
import os
import numpy as np
import datetime as dt
from adcsim.space_weather import create_space_weather_netcdf

from adcsim.Python_NRLMSISE.nrlmsise_00 import *
from adcsim.Python_NRLMSISE.nrlmsise_00_header import *


class AirDensityModel:
    def __init__(self, path_to_space_weather_netcdf='./cssi_space_weather.nc', update_netcdf=False):
        """
        Parameters
        ----------
        path_to_space_weather_netcdf : str
            file path of the space weather netcdf file, including the filename at the end
            i.e. /path/to/file/cssi_space_weather.nc
            By default, looks for the file in the current directory, and creates a new one if it can not be found.
        """
        if (not os.path.isfile(path_to_space_weather_netcdf)) or update_netcdf:
            # create the space weather netcdf if it can't be found, or if user wants to update it
            create_space_weather_netcdf('http://celestrak.com/SpaceData/SW-Last5Years.txt',
                                        path_to_space_weather_netcdf)
        self._space_dataset = xr.open_dataset(path_to_space_weather_netcdf)

    def air_mass_density(self, year=0, doy=0, sec=0.0, alt=0.0, g_lat=0.0, g_long=0.0, date=None):
        """
        Parameters
        ----------
        year : int
            year, currently ignored
        doy
            day of year
        sec
            seconds in day (UT)
        alt
            altitude in kilometers
        g_lat
            geodetic latitude
        g_long
            geodetic longitude

        Returns
        -------
        float
             air mass density in kg/m3
        """
        if date is not None:
            doy = date.timetuple().tm_yday
            sec = (date - date.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
            year = date.year
            date = date.date()
        else:
            date = self._get_date(year, doy)

        selected_data = self._space_dataset.sel(date=date)  # faster to call sel() once instead of 3 times
        # 81 day average of F10.7 flux (centered on doy)
        f107A = selected_data.ctr81_obs.item()  # TODO: should we be using the adjusted or observed value?
        # daily F10.7 flux for previous day
        f107 = selected_data.f107_obs.item()  # adjusted or observed? should this be from the previous day?
        # magnetic index(daily)
        ap = selected_data.ap_avg.item()  # use average value from the day
        # average magnetic index?
        ap_a = ap  # is ap_a the average value?

        output = nrlmsise_output()
        # note: lst is the local apparent solar time
        # I'm using the recommended formula in nrlmsise_00_header.py to calculate it

        input = nrlmsise_input(year=year, doy=doy, sec=sec, alt=alt, g_lat=g_lat, g_long=g_long,
                               lst=(sec/3600 + g_long/15), f107A=f107A, f107=f107, ap=ap, ap_a=ap_a)
        flags = nrlmsise_flags()
        # using the default recommended switches for now
        flags.switches[0] = 0
        for i in range(1, len(flags.switches)):
            flags.switches[i] = 1

        # call the model
        gtd7(input, flags, output)

        mass_density = output.d[5]  # total mass density in g/cm3

        return mass_density * 1000  # convert to kg/m3

    def _get_date(self, year, doy):
        """
        Parameters
        ----------
        year : int
            current year
        doy : int
            number of days into the year

        Returns
        -------
        datetime
        """
        days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        if self._is_leap_year(year):
            days_in_month[1] = 29

        # get the date
        month = 1
        day_of_month = doy
        day_counter = doy - days_in_month[0]
        while day_counter > 0:
            day_of_month = day_counter
            day_counter = day_counter - days_in_month[month]  # month number and index are offset by one
            month = month + 1
        date = dt.datetime(year, month, day_of_month)

        return date

    @staticmethod
    def _is_leap_year(year):
        """
        Taken from https://support.microsoft.com/en-ca/help/214019/method-to-determine-whether-a-year-is-a-leap-year

        Parameters
        ----------
        year : int
            current year

        Returns
        -------
        bool
        """
        if year % 4 == 0:
            if year % 100 == 0:
                if year % 400 == 0:
                    return True
                else:
                    return False
            else:
                return True
        else:
            return False


if __name__ == "__main__":
    from skyfield.api import utc
    air = AirDensityModel()

    year = 2019
    month = 3
    day = 24
    hour = 18
    minute = 35
    second = 1

    day_of_year = 31 + 28 + day
    second_of_day = second + minute*60 + hour*60*60

    time_track = dt.datetime(year, month, day, hour, minute, second, tzinfo=utc)

    print(air.air_mass_density(year, day_of_year, second_of_day, 400, 10, 10))
    print(air.air_mass_density(date=time_track, alt=400, g_lat=10, g_long=10))  # two prints should be identical
