import glob
import os
import shutil
import sys

import numpy as np
import pandas as pd

f_traj = sys.argv[2]

df = pd.read_csv(os.path.join(f_traj),
                 dtype={'DriverID': np.str, 'OrderID': np.str, 'lat': np.float64,
                        'lon': np.float64, 'x': np.float64, 'y': np.float64},
                 converters={'Time': pd.Timestamp},
                 error_bad_lines=False)
OutputDir = os.path.join(sys.argv[1], os.path.basename(f_traj))
if os.path.isdir(OutputDir):
    shutil.rmtree(OutputDir)
os.mkdir(OutputDir)

nptsThreshold = 20000
# divide dataset
orderStartRowIndex = df[df['OrderID'].ne(df['OrderID'].shift())].index.values
AllFilesRowIndex = []
oneFileNPts = 0
oneFileRowIndex = []
for i in range(orderStartRowIndex.shape[0]):
    if oneFileNPts == 0:
        oneFileRowIndex.append(orderStartRowIndex[i])
    if i < orderStartRowIndex.shape[0] - 1:
        oneFileNPts += orderStartRowIndex[i + 1] - orderStartRowIndex[i]
        if oneFileNPts >= nptsThreshold:
            oneFileRowIndex.append(orderStartRowIndex[i + 1] - 1)
            AllFilesRowIndex.append(oneFileRowIndex)
            oneFileNPts = 0
            oneFileRowIndex = []
    else:
        oneFileNPts += df.shape[0] - orderStartRowIndex[i] + 1
        oneFileRowIndex.append(df.shape[0] - 1)
        AllFilesRowIndex.append(oneFileRowIndex)

# write divided dataset
for i in range(len(AllFilesRowIndex)):
    oneFileRowIndex = AllFilesRowIndex[i]
    df.loc[oneFileRowIndex[0]: oneFileRowIndex[1]].to_csv(os.path.join(OutputDir, 'input_%d' % (i + 1)), index=False,
                                                          float_format='%.5f', date_format='%Y-%m-%d %H:%M:%S')
