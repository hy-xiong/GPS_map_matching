import json
import os
import sys

import numpy as np
import pandas as pd

outDir = sys.argv[1]
d = sys.argv[2]
f_road = sys.argv[3]  # geojson

with open(f_road, 'r') as rd:
    road_json = json.loads(rd.read())
xmin, ymin, xmax, ymax = road_json["bbox"]

df = pd.read_csv(d,
                 dtype={'DriverID': np.str, 'OrderID': np.str, 'lat': np.float64,
                        'lon': np.float64, 'x': np.float64, 'y': np.float64},
                 converters={'Time': pd.Timestamp},
                 error_bad_lines=False)
inBoundFlag = df.apply(
    lambda row: row['lon'] >= xmin and row['lon'] <= xmax and row['lat'] >= ymin and row['lat'] <= ymax, axis=1)
df = df[inBoundFlag]
df.to_csv(os.path.join(outDir, os.path.basename(d)), index=False, float_format='%.5f', date_format='%Y-%m-%d %H:%M:%S')
