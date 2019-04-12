import numpy as np
import pysofa
import datetime

def icrf_to_fixed(epoch: datetime.datetime, seconds: np.ndarray=0.0):
    """
    Calculate rotation matrices for transforming vectors from the ICRF (International Celestial Reference Frame) to
    a Fixed frame that follows the attitude of the Earth. These matrices should work well for STK ephemeris (.e) and
    attitude (.a) files in the ICRF and Fixed frames.

    The UT1-UTC offset, the CIP offsets, and the polar motion corrections are all ignored to match default STK
    settings. These corrections are only necessary for arcsecond precision.

    The input date is in UTC.

    The returned array has shape (3, 3) if seconds is a scalar, and (len(seconds), 3, 3) if seconds is a vector.
    """
    sec = np.atleast_1d(seconds)

    djmjd0, date = pysofa.cal2jd(epoch.year, epoch.month, epoch.day)
    time0 = (60. * (60. * epoch.hour + epoch.minute) + epoch.second) / 86400.
    dat = pysofa.dat(epoch.year, epoch.month, epoch.day, time0)

    rc2ti = np.zeros((len(sec), 3, 3))
    for i, s in enumerate(sec):
        time = (60. * (60. * epoch.hour + epoch.minute) + epoch.second + s) / 86400.
        utc = date + time
        tai = utc + dat / 86400.
        tt = tai + 32.184 / 86400.

        # cip and cio, IAU 2006/2006A
        x, y, s = pysofa.xys06a(djmjd0, tt)

        # gcrs to cirs matrix
        rc2i = pysofa.c2ixys(x, y, s)

        # earth rotation angle
        era = pysofa.era00(djmjd0 + date, time)

        # form celestial-terrestrial matrix (no polar motion yet)
        rc2ti[i, :, :] = pysofa.rz(era, rc2i)

    if isinstance(seconds, float):
        return rc2ti[0, :, :]
    else:
        return rc2ti


if __name__ == '__main__':
    rc = np.array([7e6, 0., 0.])  # vector in celestial frame
    epoch = datetime.datetime(year=2019, month=1, day=1, hour=12)
    R = icrf_to_fixed(epoch)
    rf = R @ rc  # vector in fixed frame
    print(rf)
    print(R.T @ rf)  # rotate it back again


