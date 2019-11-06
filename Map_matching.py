import json
import sys

import numpy as np
import pandas as pd

from Map_matching_util import *

pd.options.mode.chained_assignment = None

# input files
sanityCheck = int(sys.argv[1])
dir_out = sys.argv[2]
f_road = sys.argv[3]
f_trips = sys.argv[4]

bufferDist = 20.0
timeThd = 60.0

# #algorithm revise option: output map of map-matching
# evalFigDir = r'C:\Users\xiaoy\Desktop\Dissertation\Data\mmImages'
# if os.path.isdir(evalFigDir):
#     shutil.rmtree(evalFigDir)
# os.mkdir(evalFigDir)

# read GPS data
st1 = gct()
df = pd.read_csv(f_trips,
                 dtype={'DriverID': np.str, 'OrderID': np.str, 'lat': np.float64,
                        'lon': np.float64, 'x': np.float64, 'y': np.float64},
                 converters={'Time': pd.Timestamp},
                 error_bad_lines=False)
orders = df['OrderID'].unique()

# Read road network and gridize it
gridLen = 500.0  # 500 meter grid
with open(f_road, 'r') as rd:
    road_json = json.loads(rd.read())
# determine NO. of grids in X, Y
# code will be 2-digits, starting from 00 to XY
xmin, ymin, xmax, ymax = road_json["bbox"]
xLen = xmax - xmin
yLen = ymax - ymin
x_nGrids = math.ceil(xLen / gridLen)
y_nGrids = math.ceil(yLen / gridLen)
# assign road segments to grids based on intersection
roads = {}
grid_to_road = {}
segments_json = road_json["features"]

for segment_json in segments_json:
    rid = segment_json["id"]
    nextRids = segment_json["properties"]["nextAdj"]
    preRids = segment_json["properties"]["preAdj"]
    nextRids = str2IntList(nextRids)
    preRids = str2IntList(preRids)
    shape_pts = segment_json["geometry"]["coordinates"]
    roadType = segment_json["properties"]["roadType"]
    uTurnID = segment_json["properties"]["UTurnRID"]
    roads[rid] = Road(rid, nextRids, preRids, uTurnID, shape_pts, roadType)
    # determine which grids the road segment intersects with
    for k in range(len(shape_pts) - 1):
        stPt = shape_pts[k]
        edPt = shape_pts[k + 1]
        line_seg = [stPt, edPt]
        st_nX, st_nY = getGridNumForPoint(stPt, xmin, ymax, gridLen)
        ed_nX, ed_nY = getGridNumForPoint(edPt, xmin, ymax, gridLen)
        # check every grid in the bounding grid box of this line segment, see which it intersects with
        nX_min, nX_max, nY_min, nY_max = min(st_nX, ed_nX), max(st_nX, ed_nX), min(st_nY, ed_nY), max(st_nY, ed_nY)
        for nx in range(nX_min, nX_max + 1):
            for ny in range(nY_min, nY_max + 1):
                grid_bbox = [xmin + nx * gridLen, ymax - (ny + 1) * gridLen, xmin + (nx + 1) * gridLen,
                             ymax - ny * gridLen]
                grid_id = (nx, ny)
                if isLineOverlapSquare(line_seg, grid_bbox):
                    addToListDict(grid_to_road, grid_id, rid)
if not sanityCheck:
    print('gps and map reading done. Runtime: %s' % (gct() - st1))
    print('# of roads: %d' % len(roads))
    print('NO. X-grids: %d, NO. Y-grids: %d' % (x_nGrids, y_nGrids))

    print('total # of points %d' % df.shape[0])
    print('total # of orders %d' % orders.shape[0])
    print('total # of vehicles: %d' % df['DriverID'].unique().shape[0])
# process each vehicle's trajectory
trips = []
tCount = 0
pCount = 0
for order in orders:
    if not sanityCheck:
        print('process order %d: %s' % (tCount, order))
    trip = df[df['OrderID'] == order]
    xyList = trip[['x', 'y']].values.tolist()
    timeList = trip['Time'].values
    # Map-matching
    # MMFlag:
    # 3: single point subtrip
    # 2: subtrip start point
    # 1: subtrip end point
    # 0: normal point
    # -1: same position as previous point
    # -2: noise (no possible trajectory to connect it from previous normal GPS point
    #           given an upper-bound based on time and max speed.
    #           It should be removed))
    MMRoadIDs, MMFlags = map_match(xyList, timeList, bufferDist, timeThd, xmin, ymax, gridLen, roads, grid_to_road,
                                   debug=False)
    # check output MMFlags on map-matched GPS have an unexpected value pattern
    if sanityCheck:
        cleanMMFlags = list(filter(lambda x: x >= 0, MMFlags))
        problemIndex = []
        for i in range(len(MMFlags) - 1):
            if MMFlags[i] == 2 and (MMFlags[i + 1] == 3 or MMFlags[i + 1] == 2):
                problemIndex.append(i)
            elif MMFlags[i] == 1 and (MMFlags[i + 1] == 0 or MMFlags[i + 1] == 1):
                problemIndex.append(i)
            elif MMFlags[i] == 3 and (MMFlags[i + 1] == 0 or MMFlags[i + 1] == 1):
                problemIndex.append(i)
            elif MMFlags[i] == 0 and (MMFlags[i + 1] == 2 or MMFlags[i + 1] == 3):
                problemIndex.append(i)
        if len(problemIndex) > 0:
            print('Order %s has problem at point: %s (index is counted by excluding MMFlags < 0)' \
                  % (order, ', '.join(str(v) for v in problemIndex)))
    # compute average speed given matched result
    avgSpeeds = []
    destMMRoad = []
    intermediateRoads = []
    startRoad_distToEndPt = []
    destRoad_distToStartPt = []
    travelDist = []
    st = gct()
    SPSearchGridExBand = 1
    for i in range(len(MMFlags) - 1):
        avgSpeed = -1.0
        destRid = -1
        distInStartRoad = -1.0
        distInEndRoad = -1.0
        pathLen = -1.0
        TPath = []
        if MMFlags[i] == 2 or MMFlags[i] == 0:
            nextValidPt = i + 1
            # look for next map-matched point until the end of trajectory
            # assuming every trajectory point's map-matching flag is right
            while MMFlags[nextValidPt] <= -1:
                nextValidPt += 1
            startPt = xyList[i]
            startRid = MMRoadIDs[i]
            endPt = xyList[nextValidPt]
            endRid = MMRoadIDs[nextValidPt]
            destRid = endRid
            startPt_nX, startPt_nY = getGridNumForPoint(startPt, xmin, ymax, gridLen)
            endRid_nX, endRid_nY = getGridNumForPoint(endPt, xmin, ymax, gridLen)
            nX_LB = min(startPt_nX, endRid_nX) - SPSearchGridExBand
            nX_UB = max(startPt_nX, endRid_nX) + SPSearchGridExBand
            nY_LB = min(startPt_nY, endRid_nY) - SPSearchGridExBand
            nY_UB = max(startPt_nY, endRid_nY) + SPSearchGridExBand
            relatedRids = set([])
            for i_nX in range(nX_LB, nX_UB + 1):
                for i_nY in range(nY_LB, nY_UB + 1):
                    if (i_nX, i_nY) in grid_to_road:
                        relatedRids |= set(grid_to_road[(i_nX, i_nY)])
            relatedRoads = {}
            for rid in relatedRids:
                relatedRoads[rid] = roads[rid]
            # total travel distance, [travel path], [distance between start location to end of start road,
            #                                        distance between end location to start of end road]
            TDist, TPath, TSEDist = DijkstraSP(relatedRoads, startRid, startPt, endRid, endPt, weighted=True)
            if TDist == float('inf'):
                if len(TPath) > 0:
                    avgSpeed = 0.
                    distInStartRoad, distInEndRoad = TSEDist
                    pathLen = 0.
                else:
                    avgSpeed = -1.0
                    destRid = -1
            else:
                timeDif = (timeList[nextValidPt] - timeList[i]) / np.timedelta64(1, 's')
                if timeDif == 0:
                    # cut the sub-trajectory into 2
                    avgSpeed = -1.0
                    destRid = -1
                    if MMFlags[i] == 0:
                        MMFlags[i] = 1
                    else:
                        MMFlags[i] = 3
                    if MMFlags[nextValidPt] == 0:
                        MMFlags[nextValidPt] = 2
                    else:
                        MMFlags[nextValidPt] = 3
                else:
                    # convert to km/h
                    avgSpeed = TDist / timeDif * 3.6
                    pathLen = TDist
                    distInStartRoad, distInEndRoad = TSEDist
        avgSpeeds.append(avgSpeed)
        destMMRoad.append(destRid)
        startRoad_distToEndPt.append(distInStartRoad)
        destRoad_distToStartPt.append(distInEndRoad)
        travelDist.append(pathLen)
        intermediateRoads.append(' '.join(str(r) for r in TPath[1:-1]))
    # for trajectory last point
    avgSpeeds.append(-1.0)
    destMMRoad.append(-1)
    startRoad_distToEndPt.append(-1.0)
    destRoad_distToStartPt.append(-1.0)
    travelDist.append(-1.0)
    intermediateRoads.append('')
    # write output
    trip["road"] = MMRoadIDs
    trip["MMFlags"] = MMFlags
    trip["spd"] = avgSpeeds
    trip["destRoad"] = destMMRoad
    trip["startRoad_distToEndPt"] = startRoad_distToEndPt
    trip["destRoad_distToStartPt"] = destRoad_distToStartPt
    trip["travelDist"] = travelDist
    trip["roadInBetween"] = intermediateRoads
    trips.append(trip)
    tCount += 1
    pCount += trip.shape[0]
    if not sanityCheck:
        print('# order processed: %d, points processed: %d %.2f%%' % (tCount, pCount, pCount * 100.0 / df.shape[0]))
    # # plot map-matching results by each point pair
    # figDir = os.path.join(evalFigDir, 'order_%s' % order)
    # if os.path.isdir(figDir):
    #     shutil.rmtree(figDir)
    # os.mkdir(figDir)
    # visualByPt(trip, figDir, SPSearchGridExBand, xmin, ymax, gridLen, grid_to_road, roads)
# regroup all trajectory into one dataframe
newTrip = pd.concat(trips)

if newTrip.shape[0] != df.shape[0]:
    s = 'Error! New dataframe has %d records and old dataframe has %d records' % (newTrip.shape[0], df.shape[0])
    raise Exception(s)

# evaluate map-matching
if not sanityCheck:
    nPoints = newTrip.shape[0]
    nNoise = 0
    nIsolated = 0
    nBreak = 0
    nSamePos = 0
    for i, v in newTrip['MMFlags'].items():
        if v == -2:
            nNoise += 1
        elif v == 3:
            nIsolated += 1
        elif v == 1:
            nBreak += 1
        elif v == -1:
            nSamePos += 1
    nBreak -= orders.shape[0]
    print('\n')
    print('# of points processed: %d, avg pts per trip: %.1f' % (nPoints, nPoints * 1.0 / len(trips)))
    print('%d, %.2f%% no nearby road pts\n%d, %.2f%% break points\n' % (nNoise, nNoise * 100.0 / nPoints,
                                                                        nBreak, nBreak * 100.0 / nPoints))
    print('%d, %.2f%% isolated points\n%d, %.2f%% samePosition points\n' % (nIsolated, nIsolated * 100.0 / nPoints,
                                                                            nSamePos, nSamePos * 100.0 / nPoints))
    print('Error points lower & upper bound: %.2f%% - %.2f%%' % (nNoise * 100.0 / nPoints,
                                                                 (nNoise + nBreak + nIsolated) * 100.0 / nPoints))
    print('\nafter removing duplicate GPS points')
    nPoints -= nSamePos
    print('# of points processed: %d, avg pts per trip: %.1f' % (nPoints, nPoints * 1.0 / len(trips)))
    print('%d, %.2f%% no nearby road pts\n%d, %.2f%% break points\n%d, %.2f%% isolated points' %
          (nNoise, nNoise * 100.0 / nPoints, nBreak, nBreak * 100.0 / nPoints, nIsolated, nIsolated * 100.0 / nPoints))
    print('Error points lower & upper bound: %.2f%% - %.2f%%' % (nNoise * 100.0 / nPoints,
                                                                 (nNoise + nBreak + nIsolated) * 100.0 / nPoints))
# output final map match result
newTrip.to_csv(os.path.join(dir_out, '%s_mm' % os.path.basename(f_trips).split('.')[0]), index=False,
               float_format='%.5f', date_format='%Y-%m-%d %H:%M:%S')
