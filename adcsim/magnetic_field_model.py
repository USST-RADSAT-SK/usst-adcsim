"""
The code in this file allows us to get the magnetic field at any point in orbit.
"""

# this file was copied from https://github.com/cmweiss/geomag and altered to avoid the need to convert to
# geodetic coordinates and back

# geomag.py
# by Christopher Weiss cmweiss@gmail.com

# Adapted from the geomagc software and World Magnetic Model of the NOAA
# Satellite and Information Service, National Geophysical Data Center
# http://www.ngdc.noaa.gov/geomag/WMM/DoDWMM.shtml
#
# Suggestions for improvements are appreciated.

# USAGE:
#
# >>> gm = geomag.GeoMag("WMM.COF")
# >>> mag = gm.GeoMag(80,0)
# >>> mag.dec
# -6.1335150785195536
# >>>

# Note: this thing loads the most recent coefficients for the magnetic field model used (which is good)

# Note: I'm not sure if I trust this code in the long run. For initial results it seems fine.

import numpy as np
import math
import os
import datetime
from adcsim import icrf_to_fixed


class GeoMag:

    def _GeoMagSpherical(self, sintheta, costheta, sinphi, cosphi, r, time=datetime.date.today()): # Geocentric terrestrial Cartesian r=(x,y,z), (meters), date
        # contains the core of the magnetic field calculation, with spherical input and output

        time = time.year+(time.timetuple().tm_yday/365.0)  # usst-adcs fix (easier way to get day of year)
        dt = time - self.epoch

        sp = sinphi
        cp = cosphi
        st = sintheta
        ct = costheta
        r *= 1e-3

        self.sp[1] = sp
        self.cp[1] = cp

        for m in range(2,self.maxord+1):
            self.sp[m] = self.sp[1]*self.cp[m-1]+self.cp[1]*self.sp[m-1]
            self.cp[m] = self.cp[1]*self.cp[m-1]-self.sp[1]*self.sp[m-1]

        aor = self.re/r
        ar = aor*aor
        br = bt = bp = bpp = 0.0
        for n in range(1,self.maxord+1):
            ar = ar*aor

            #for (m=0,D3=1,D4=(n+m+D3)/D3;D4>0;D4--,m+=D3):
            m=0
            D3=1
            #D4=(n+m+D3)/D3
            D4=(n+m+1)
            while D4>0:

                # /*
                # COMPUTE UNNORMALIZED ASSOCIATED LEGENDRE POLYNOMIALS
                # AND DERIVATIVES VIA RECURSION RELATIONS
                # */
                if (n == m):
                    self.p[m][n] = st * self.p[m-1][n-1]
                    self.dp[m][n] = st*self.dp[m-1][n-1]+ct*self.p[m-1][n-1]

                elif (n == 1 and m == 0):
                    self.p[m][n] = ct*self.p[m][n-1]
                    self.dp[m][n] = ct*self.dp[m][n-1]-st*self.p[m][n-1]

                elif (n > 1 and n != m):
                    if (m > n-2):
                        self.p[m][n-2] = 0
                    if (m > n-2):
                        self.dp[m][n-2] = 0.0
                    self.p[m][n] = ct*self.p[m][n-1]-self.k[m][n]*self.p[m][n-2]
                    self.dp[m][n] = ct*self.dp[m][n-1] - st*self.p[m][n-1]-self.k[m][n]*self.dp[m][n-2]

                # /*
                # TIME ADJUST THE GAUSS COEFFICIENTS
                # */
                self.tc[m][n] = self.c[m][n]+dt*self.cd[m][n]
                if (m != 0):
                    self.tc[n][m-1] = self.c[n][m-1]+dt*self.cd[n][m-1]

                # /*
                # ACCUMULATE TERMS OF THE SPHERICAL HARMONIC EXPANSIONS
                # */
                par = ar*self.p[m][n]

                if (m == 0):
                    temp1 = self.tc[m][n]*self.cp[m]
                    temp2 = self.tc[m][n]*self.sp[m]
                else:
                    temp1 = self.tc[m][n]*self.cp[m]+self.tc[n][m-1]*self.sp[m]
                    temp2 = self.tc[m][n]*self.sp[m]-self.tc[n][m-1]*self.cp[m]

                bt = bt-ar*temp1*self.dp[m][n]
                bp = bp + (self.fm[m] * temp2 * par)
                br = br + (self.fn[n] * temp1 * par)
                # /*
                # SPECIAL CASE:  NORTH/SOUTH GEOGRAPHIC POLES
                # */
                if (st == 0.0 and m == 1):
                    if (n == 1):
                        self.pp[n] = self.pp[n-1]
                    else:
                        self.pp[n] = ct*self.pp[n-1]-self.k[m][n]*self.pp[n-2]
                    parp = ar*self.pp[n]
                    bpp = bpp + (self.fm[m]*temp2*parp)

                D4=D4-1
                m=m+1

        if (st == 0.0):
            bp = bpp
        else:
            bp = bp/st

        return np.array([br, bt, bp])

    def GeoMag(self, location, time=datetime.datetime.today(), location_format='geodetic', output_format='geodetic'):
        """
        Calculate the magnetic field from the WMM 2019.

        Parameters
        ----------
        location : np.ndarray
            A 3-element vector specifying the location to get the magnetic field at. Interpretation of location
            depends on location_format.
        time : datetime.datetime
            Date to get the magnetic field at.
        location_format : str
            Specifies the interpretation of the location vector. Can take the following values:
            geodetic: location = [latitude (deg), longitude (deg), altitude (m)]
            cartesian: location = [x, y, z] (m) in fixed-Earth coordinate frame
            inertial: location = [x, y, z] (m) in inertial coordinate frame
        output_format : str
            Specifies the output format. Can take the following values:
            geodetic: output = [north, east, nadir] (nT)
            compass: output = [declination (deg), inclination (deg), magnitude (nT)]
            cartesian: output = [x, y, z] (nT) in fixed-Earth coordinate frame
            inertial: output = [x, y, z] (nT) in inertial coordinate frame
        """
        # convert location to spherical coordinates
        if location_format == 'geodetic':
            lat, lon, alt = np.deg2rad(location[0]), np.deg2rad(location[1]), location[2] * 1e-3  # convert from deg to rad, m to km
            slon = np.sin(lon)
            slat = np.sin(lat)
            clon = np.cos(lon)
            clat = np.cos(lat)
            slat2 = slat*slat
            clat2 = clat*clat
            q = math.sqrt(self.a2-self.c2*slat2)
            q1 = alt*q
            q2 = ((q1+self.a2)/(q1+self.b2))*((q1+self.a2)/(q1+self.b2))
            ct = slat/math.sqrt(q2*clat2+slat2)
            st = math.sqrt(1.0-(ct*ct))
            r2 = (alt*alt)+2.0*q1+(self.a4-self.c4*slat2)/(q*q)
            r = math.sqrt(r2)
            sp = slon
            cp = clon

            d = math.sqrt(self.a2*clat2+self.b2*slat2)
            ca = (alt+d)/r
            sa = self.c2*clat*slat/(r*d)

            # convert back to meters
            d *= 1e3
            r *= 1e3
            alt *= 1e3
        elif location_format == 'cartesian' or location_format == 'inertial':
            if location_format == 'inertial':
                location = icrf_to_fixed.icrf_to_fixed(time) @ location
            x, y, z = location[0], location[1], location[2]
            h = math.sqrt(x**2 + y**2)
            r = math.sqrt(x**2 + y**2 + z**2)
            sp = y / h
            cp = x / h
            st = h / r
            ct = z / r
        else:
            raise ValueError(f'Invalid location format \'{location_format}\'')

        br, bt, bp = self._GeoMagSpherical(st, ct, sp, cp, r, time)

        # convert magnetic field from spherical coordinates to output_format
        if output_format == 'geodetic' or output_format == 'compass':
            if location_format != 'geodetic':
                raise ValueError('At this time, a geodetic location must be specified to get geodetic or compass output')
            bnorth = -bt*ca-br*sa
            beast = bp
            bdown = bt*sa-br*ca
            if output_format == 'compass':
                bh = np.sqrt((bnorth*bnorth)+(beast*beast))
                ti = np.sqrt((bh*bh)+(bdown*bdown))
                dec = np.rad2deg(math.atan2(beast,bnorth))
                dip = np.rad2deg(math.atan2(bdown,bh))
                return np.array([dec, dip, ti])
            return np.array([bnorth, beast, bdown])
        elif output_format == 'cartesian' or output_format == 'inertial':
            bx = cp*st*br + cp*ct*bt - sp*bp
            by = sp*st*br + sp*ct*bt + cp*bp
            bz = ct*br - st*bt
            b = np.array([bx, by, bz])
            if output_format == 'inertial':
                return icrf_to_fixed.icrf_to_fixed(time).T @ b
            return b
        else:
            raise ValueError(f'Invalid output format \'{output_format}\'')

    def __init__(self, wmm_filename=None):
        if wmm_filename is None:
            wmm_filename = os.path.join(os.path.dirname(__file__), 'WMM_2015_v2.COF')
        print(wmm_filename)

        wmm=[]
        with open(wmm_filename) as wmm_file:
            for line in wmm_file:
                linevals = line.strip().split()
                if len(linevals) == 3:
                    self.epoch = float(linevals[0])
                    self.model = linevals[1]
                    self.modeldate = linevals[2]
                elif len(linevals) == 6:
                    linedict = {'n': int(float(linevals[0])),
                                'm': int(float(linevals[1])),
                                'gnm': float(linevals[2]),
                                'hnm': float(linevals[3]),
                                'dgnm': float(linevals[4]),
                                'dhnm': float(linevals[5])}
                    wmm.append(linedict)

        z = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        self.maxord = self.maxdeg = 12
        self.tc = [z[0:13],z[0:13],z[0:13],z[0:13],z[0:13],z[0:13],z[0:13],z[0:13],z[0:13],z[0:13],z[0:13],z[0:13],z[0:13],z[0:13]]
        self.sp = z[0:14]
        self.cp = z[0:14]
        self.cp[0] = 1.0
        self.pp = z[0:13]
        self.pp[0] = 1.0
        self.p = [z[0:14],z[0:14],z[0:14],z[0:14],z[0:14],z[0:14],z[0:14],z[0:14],z[0:14],z[0:14],z[0:14],z[0:14],z[0:14],z[0:14]]
        self.p[0][0] = 1.0
        self.dp = [z[0:13],z[0:13],z[0:13],z[0:13],z[0:13],z[0:13],z[0:13],z[0:13],z[0:13],z[0:13],z[0:13],z[0:13],z[0:13],z[0:13]]
        self.a = 6378.137
        self.b = 6356.7523142
        self.re = 6371.2
        self.a2 = self.a*self.a
        self.b2 = self.b*self.b
        self.c2 = self.a2-self.b2
        self.a4 = self.a2*self.a2
        self.b4 = self.b2*self.b2
        self.c4 = self.a4 - self.b4

        self.c = [z[0:14],z[0:14],z[0:14],z[0:14],z[0:14],z[0:14],z[0:14],z[0:14],z[0:14],z[0:14],z[0:14],z[0:14],z[0:14],z[0:14]]
        self.cd = [z[0:14],z[0:14],z[0:14],z[0:14],z[0:14],z[0:14],z[0:14],z[0:14],z[0:14],z[0:14],z[0:14],z[0:14],z[0:14],z[0:14]]

        for wmmnm in wmm:
            m = wmmnm['m']
            n = wmmnm['n']
            gnm = wmmnm['gnm']
            hnm = wmmnm['hnm']
            dgnm = wmmnm['dgnm']
            dhnm = wmmnm['dhnm']
            if (m <= n):
                self.c[m][n] = gnm
                self.cd[m][n] = dgnm
                if (m != 0):
                    self.c[n][m-1] = hnm
                    self.cd[n][m-1] = dhnm

        #/* CONVERT SCHMIDT NORMALIZED GAUSS COEFFICIENTS TO UNNORMALIZED */
        self.snorm = [z[0:13],z[0:13],z[0:13],z[0:13],z[0:13],z[0:13],z[0:13],z[0:13],z[0:13],z[0:13],z[0:13],z[0:13],z[0:13]]
        self.snorm[0][0] = 1.0
        self.k = [z[0:13],z[0:13],z[0:13],z[0:13],z[0:13],z[0:13],z[0:13],z[0:13],z[0:13],z[0:13],z[0:13],z[0:13],z[0:13]]
        self.k[1][1] = 0.0
        self.fn = [0.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0]
        self.fm = [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0]
        for n in range(1,self.maxord+1):
            self.snorm[0][n] = self.snorm[0][n-1]*(2.0*n-1)/n
            j=2.0
            #for (m=0,D1=1,D2=(n-m+D1)/D1;D2>0;D2--,m+=D1):
            m=0
            D1=1
            D2=(n-m+D1)/D1
            while (D2 > 0):
                self.k[m][n] = (((n-1)*(n-1))-(m*m))/((2.0*n-1)*(2.0*n-3.0))
                if (m > 0):
                    flnmj = ((n-m+1.0)*j)/(n+m)
                    self.snorm[m][n] = self.snorm[m-1][n]*math.sqrt(flnmj)
                    j = 1.0
                    self.c[n][m-1] = self.snorm[m][n]*self.c[n][m-1]
                    self.cd[n][m-1] = self.snorm[m][n]*self.cd[n][m-1]
                self.c[m][n] = self.snorm[m][n]*self.c[m][n]
                self.cd[m][n] = self.snorm[m][n]*self.cd[m][n]
                D2=D2-1
                m=m+D1


def magnetic_field(date: datetime.datetime, lat, lon, alt, output_format='cartesian'):
    """
    Outputs magnetic field given lat, lon, alt.
    :param date: date
    :param lat: latitude
    :param lon: longitude
    :param alt: altitude
    :param output_format : str
            Specifies the output format. Can take the following values:
            geodetic: output = [north, east, nadir] (nT)
            compass: output = [declination (deg), inclination (deg), magnitude (nT)]
            cartesian: output = [x, y, z] (nT) in fixed-Earth coordinate frame
            inertial: output = [x, y, z] (nT) in inertial coordinate frame
    :return:
    """
    g = GeoMag()
    return g.GeoMag(np.array([lat, lon, alt]), date, location_format='geodetic', output_format=output_format)


if __name__ == "__main__":
    from astropy import coordinates as coords
    from astropy.time import Time
    from astropy import units as u
    from skyfield.api import load, EarthSatellite, utc
    from datetime import datetime

    line1 = '1 44031U 98067PX  19083.14584174  .00005852  00000-0  94382-4 0  9997'
    line2 = '2 44031  51.6393  63.5548 0003193 165.0023 195.1063 15.54481029  8074'
    satellite = EarthSatellite(line1, line2)
    ts = load.timescale()
    time_track = datetime(2019, 3, 24, 18, 35, 1, tzinfo=utc)
    t = ts.utc(time_track)
    geo = satellite.at(t)
    subpoint = geo.subpoint()

    now = Time(time_track)
    mag_ecef = magnetic_field(time_track, subpoint.latitude.degrees, subpoint.longitude.degrees, subpoint.elevation.m)
    mag_inertial = magnetic_field(time_track, subpoint.latitude.degrees, subpoint.longitude.degrees,
                                  subpoint.elevation.m, output_format='inertial')

    mag_ecef_obj = coords.EarthLocation.from_geocentric(mag_ecef[0], mag_ecef[1], mag_ecef[2], unit=u.meter)
    mag_gcrs = mag_ecef_obj.get_gcrs(obstime=now).cartesian.xyz.value

    print(mag_inertial)
    print(mag_gcrs)
    # This shows that the inertial system output from the magnetic_field function is very close to what astropy gives in
    # gcrs
