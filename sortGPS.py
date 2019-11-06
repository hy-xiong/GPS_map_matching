"""
This module sort each daily GPS trajectories by Driver ID and Time.
Then uncover the masked true GPS location using a local search algorithm
Finally output the trajectory with columns:
	DriverID, OrderID - str
	Time - str (datetime)
	lon, lat, x, y - float
x, y are projected coordinate of lat lon in study area.
The projection CS is WGS85_UTM48

@author Haoyi Xiong
"""

import datetime
import os

import numpy as np
import pandas as pd
import pyproj
import sys
from dateutil import tz
from math import pi, sin, cos, sqrt

outDir = sys.argv[1]
fpath = sys.argv[2]

# read rawGPS
tzlocal = tz.tzoffset('Beijing', 8 * 3600)


def timeConv(x): return datetime.datetime.fromtimestamp(int(x), tzlocal)


df_GPS = pd.read_csv(fpath,
                     names=['DriverID', 'OrderID', 'Time', 'lon', 'lat'],
                     dtype={'DriverID': np.str, 'OrderID': np.str,
                            'lon': np.float64, 'lat': np.float64},
                     converters={'Time': timeConv})
df_GPS = df_GPS.sort_values(by=['DriverID', 'Time'])
dup = df_GPS[['DriverID', 'Time']]
df_GPS = df_GPS[~dup.eq(dup.shift()).all(axis=1)]

'''Estimate location at WGS84 based on location at GCJ02 using local search
Based on https://github.com/caijun/geoChina/blob/master/R/cst.R
Input: V' - gcj02 lat&lon; f: wgc -> gcj
Output: V - wgs lat&lon
Idea:
1. gcj -> wgs: V = V' - dV', we cannot get dV' from V',
   as the generation of V' from its original V is somehow random
   based on lat&lon
2. Assuming V' as input to wgs -> gcj, dV = V'' - V',
   where V'' is the estimated gcj02 based on V',
   and dV is the difference-vector between them,
   the dV should be close to that of dV' as V' is close to V.
3. We can estimate V using V_temp = V' - dV.
   Since V_temp is an appriomation of V,
   we can use V_temp as V to estimate V' now.
4. The error between estimate V' and true V' will be used to
   adjust the position of V_temp.
5. Keep iterating until the adjustment of V_temp is smaller than a threshold
'''

a = 6378245.0
f = 0.00335233
b = a * (1 - f)
ee = (a * a - b * b) / (a * a)


def GCJ02_to_WGS(lat, lon):
    clat, clon = lat, lon
    nlat, nlon = wgs2gcj(clat, clon)
    nlat, nlon = clat - (nlat - lat), clon - (nlon - lon)
    # local search
    gamma = 1e-6
    while (max(abs(nlat - clat), abs(nlon - clon)) >= gamma):
        clat, clon = nlat, nlon
        nlat, nlon = wgs2gcj(clat, clon)
        nlat, nlon = clat - (nlat - lat), clon - (nlon - lon)
    return nlat, nlon


def wgs2gcj(lat, lon):
    dLon = transformLon(lon - 105.0, lat - 35.0)
    dLat = transformLat(lon - 105.0, lat - 35.0)
    radLat = lat / 180.0 * pi
    magic = sin(radLat)
    magic = 1 - ee * magic * magic
    sqrtMagic = sqrt(magic)
    dLat = (dLat * 180.0) / ((a * (1 - ee)) / (magic * sqrtMagic) * pi)
    dLon = (dLon * 180.0) / (a / sqrtMagic * cos(radLat) * pi)
    return lat + dLat, lon + dLon


def transformLat(x, y):
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * \
          sqrt(abs(x))
    ret += (20.0 * sin(6.0 * x * pi) + 20.0 * sin(2.0 * x * pi)) * 2.0 / 3.0
    ret += (20.0 * sin(y * pi) + 40.0 * sin(y / 3.0 * pi)) * 2.0 / 3.0
    ret += (160.0 * sin(y / 12.0 * pi) + 320.0 * sin(y * pi / 30.0)) * 2.0 / 3.0
    return ret


def transformLon(x, y):
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * sqrt(abs(x))
    ret += (20.0 * sin(6.0 * x * pi) + 20.0 * sin(2.0 * x * pi)) * 2.0 / 3.0
    ret += (20.0 * sin(x * pi) + 40.0 * sin(x / 3.0 * pi)) * 2.0 / 3.0
    ret += (150.0 * sin(x / 12.0 * pi) + 300.0 * sin(x * pi / 30.0)) * 2.0 / 3.0
    return ret


xy = df_GPS.apply(lambda row: GCJ02_to_WGS(row['lat'], row['lon']),
                  axis=1, result_type='expand')
xy.columns = ['wgsLat', 'wgsLon']
df_GPS['lat'] = xy['wgsLat']
df_GPS['lon'] = xy['wgsLon']

# estimate x, y given true lat, lon and projection CS
prj1 = pyproj.Proj('+proj=latlong +datum=WGS84')
prj2 = pyproj.Proj('+proj=utm +zone=48 +datum=WGS84')
prj = lambda x, y: pyproj.transform(prj1, prj2, x, y)
xy = df_GPS.apply(lambda row: prj(row['lon'], row['lat']),
                  axis=1, result_type='expand')
xy.columns = ['x', 'y']
df_GPS = df_GPS.merge(xy, left_index=True, right_index=True)

# write GPS
df_GPS.to_csv(os.path.join(outDir, os.path.basename(fpath)), index=False,
              float_format='%.5f', date_format='%Y-%m-%d %H:%M:%S')
