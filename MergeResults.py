'''
Created on Oct 25, 2017

@author: hxiong
'''

import glob
import os
import sys

import numpy as np
import pandas as pd

Output_unPrj = sys.argv[1]
Output_prj = sys.argv[2]
MMOutDir = sys.argv[3]
dfs = []
flist = sorted(glob.glob(os.path.join(MMOutDir, 'input*_mm')), key=lambda x: int(os.path.basename(x).split('_')[1]))
for f in flist:
    df = pd.read_csv(f,
                     dtype={'DriverID': np.str, 'OrderID': np.str, 'lat': np.float64,
                            'lon': np.float64, 'x': np.float64, 'y': np.float64,
                            'road': np.int32, 'MMFlags': np.int8, 'spd': np.float64},
                     converters={'Time': pd.Timestamp},
                     error_bad_lines=False)
    dfs.append(df)
df = pd.concat(dfs)
cars = df['DriverID'].unique()
orders = df['OrderID'].unique()
# get results with all columns
df_unPrj = df.drop(['x', 'y'], axis=1)
df_unPrj.to_csv(os.path.join(Output_unPrj, '%s' % os.path.basename(MMOutDir)), index=False,
                float_format='%.5f', date_format='%Y-%m-%d %H:%M:%S')
# get results with only processed data
df_prj = df.drop(['lon', 'lat'], axis=1)
df_prj.to_csv(os.path.join(Output_prj, '%s' % os.path.basename(MMOutDir)), index=False,
              float_format='%.5f', date_format='%Y-%m-%d %H:%M:%S')
# evaluate map-matching
nPoints = df.shape[0]
nNoise = 0
nIsolated = 0
nBreak = 0
nSamePos = 0
for i, v in df['MMFlags'].items():
    if v == -2:
        nNoise += 1
    elif v == 3:
        nIsolated += 1
    elif v == 1:
        nBreak += 1
    elif v == -1:
        nSamePos += 1
nBreak -= orders.shape[0]
print('# of points, vehicles, orders: %d %d %d' % (nPoints, cars.shape[0], orders.shape[0]))
print('# of points processed: %d, avg pts per vehicle: %.1f' % (nPoints, nPoints * 1.0 / cars.shape[0]))
print('%d, %.2f%% noise road pts\n%d, %.2f%% break points\n%d, %.2f%% isolated points\n%d, %.2f%% samePosition points' %
      (nNoise, nNoise * 100.0 / nPoints, nBreak, nBreak * 100.0 / nPoints, nIsolated, nIsolated * 100.0 / nPoints,
       nSamePos, nSamePos * 100.0 / nPoints))
print('Error points lower & upper bound: %.2f%% - %.2f%%' % ((nNoise) * 100.0 / nPoints,
                                                             (nNoise + nBreak + nIsolated) * 100.0 / nPoints))
print('\nafter removing same position GPS points')
nPoints -= nSamePos
print('# of points processed: %d, avg pts per vehicle: %.1f' % (nPoints, nPoints * 1.0 / cars.shape[0]))
print('%d, %.2f%% noise road pts\n%d, %.2f%% break points\n%d, %.2f%% isolated points' %
      (nNoise, nNoise * 100.0 / nPoints, nBreak, nBreak * 100.0 / nPoints, nIsolated, nIsolated * 100.0 / nPoints))
print('Error points lower & upper bound: %.2f%% - %.2f%%' % ((nNoise) * 100.0 / nPoints,
                                                             (nNoise + nBreak + nIsolated) * 100.0 / nPoints))
