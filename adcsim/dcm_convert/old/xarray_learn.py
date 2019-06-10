import xarray as xr
import numpy as np


values = np.load("quaternions.npy", 'r')
ttime = np.linspace(0,10,len(values))

xdcm = xr.Dataset({"version": "stk.v.11.0",
                   "NumberOfAttitudePoints": len(ttime),
                   "BlockingFactor": 20,
                   "InterpolationOrder": 1,
                   "CentralBody": "Earth",
                   "ScenarioEpoch": "1 Jan 2019 18:00:00.000000",
                   "CoordinateAxes": "Fixed",
                   "format": "AttitudeTimeQuaternions",
                   "time": ttime,
                   "quaternions":  (['a','b'], values)
                   })
# xdcm.time.values will not include the coordinates, just the numerical values

xdcm["time"].to_index()
# xdcm["quaternions"].to_index()
print(values)