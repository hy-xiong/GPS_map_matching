'''
Created on Oct 21, 2017

@author: xiaoy
'''
import datetime
import math
import os

import matplotlib as mpl
import numpy as np
from scipy import spatial

mpl.use('Agg')
from matplotlib import pyplot as plt

plt.ioff()


class Road:
    def __init__(self, rid, nextRids, preRids, uTurnID, shape_pts, roadType):
        self.rid = rid
        self.nextRids = nextRids
        self.preRids = preRids
        self.shape_pts = shape_pts
        self.roadType = roadType
        self.length = 0.0
        self.uTurnID = uTurnID
        for k in range(len(shape_pts) - 1):
            self.length += EuDist(shape_pts[k], shape_pts[k + 1])


gct = datetime.datetime.now
inBetween = lambda v, bot, top: True if v >= bot - 0.001 and v <= top + 0.001 else False
minMax = lambda v1, v2: (min(v1, v2), max(v1, v2))


def ptDistToLine(pt, line):
    x1, y1 = line[0]
    x2, y2 = line[1]
    x3, y3 = pt
    k = ((y2 - y1) * (x3 - x1) - (x2 - x1) * (y3 - y1)) / ((y2 - y1) ** 2 + (x2 - x1) ** 2)
    x4 = x3 - k * (y2 - y1)
    y4 = y3 + k * (x2 - x1)
    if not (inBetween(x4, *minMax(x1, x2)) and inBetween(y4, *minMax(y1, y2))):
        dist1 = EuDist(pt, line[0])
        dist2 = EuDist(pt, line[1])
        x4, y4 = line[0] if dist1 <= dist2 else line[1]
    return EuDist((x4, y4), pt), x4, y4


def ptDistToRoad(pt, road_shape_pts, distanceOnly=True):
    minDist = 9999.0
    jun_pt = None
    jun_lineSegNum = 0
    for i in range(len(road_shape_pts) - 1):
        line = [road_shape_pts[i], road_shape_pts[i + 1]]
        dist, x, y = ptDistToLine(pt, line)
        if minDist > dist:
            minDist = dist
            jun_pt = (x, y)
            jun_lineSegNum = i
    if distanceOnly:
        return minDist
    else:
        return minDist, jun_pt, jun_lineSegNum


def EuDist(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def getGridNumForPoint(pt, xmin, ymax, gridLen):
    # left to right, top to bot
    nX = int((pt[0] - xmin) / gridLen)
    nY = int((ymax - pt[1]) / gridLen)
    return nX, nY


def addToListDict(d, v, e):
    if v not in d:
        d[v] = [e]
    else:
        if e not in d[v]:
            d[v].append(e)


def str2IntList(l):
    l = l.split(',') if len(l) > 0 else []
    return [int(s) for s in l]


def _isIntersect(lineA, lineB):
    [x1, y1], [x2, y2] = lineA
    [x3, y3], [x4, y4] = lineB
    slopeA = (y2 - y1) / (x2 - x1) if x2 - x1 != 0 else "NA"
    slopeB = (y4 - y3) / (x4 - x3) if x4 - x3 != 0 else "NA"
    if slopeA != "NA" and slopeB != "NA":
        if slopeA != slopeB:
            jun_x = (y3 - y1 + slopeA * x1 - slopeB * x3) / (slopeA - slopeB)
            jun_y = slopeA * (jun_x - x1) + y1
            if inBetween(jun_x, *minMax(x1, x2)) and inBetween(jun_y, *minMax(y1, y2)) \
                    and inBetween(jun_x, *minMax(x3, x4)) and inBetween(jun_y, *minMax(y3, y4)):
                return True
        else:
            jun_y = (x3 - x1) * slopeA + y1
            if abs(jun_y - y3) < 0.0000001 and \
                    (inBetween(x1, *minMax(x3, x4)) or inBetween(x2, *minMax(x3, x4)) \
                     or inBetween(x3, *minMax(x1, x2)) or inBetween(x4, *minMax(x1, x2))):
                return True
    elif slopeA == "NA" and slopeB != "NA":
        jun_x = x1
        jun_y = slopeB * (jun_x - x3) + y3
        if inBetween(jun_x, *minMax(x3, x4)) and inBetween(jun_y, *minMax(y3, y4)) \
                and inBetween(jun_y, *minMax(y1, y2)):
            return True
    elif slopeA != "NA" and slopeB == "NA":
        jun_x = x3
        jun_y = slopeA * (jun_x - x1) + y1
        if inBetween(jun_x, *minMax(x1, x2)) and inBetween(jun_y, *minMax(y1, y2)) \
                and inBetween(jun_y, *minMax(y3, y4)):
            return True
    else:
        if x1 == x3:
            return True
    return False


def isLineOverlapSquare(line, SquareBBox):
    # @arg line: [[x, y], [x, y]]
    # @arg SquareBBox: [xmin, ymin, xmax, ymax]
    xmin, ymin, xmax, ymax = SquareBBox
    bBot = [[xmin, ymin], [xmax, ymin]]
    bLeft = [[xmin, ymin], [xmin, ymax]]
    bRight = [[xmax, ymin], [xmax, ymax]]
    bTop = [[xmin, ymax], [xmax, ymax]]
    # check within
    if inBetween(line[0][0], xmin, xmax) and inBetween(line[1][0], xmin, xmax) \
            and inBetween(line[0][1], ymin, ymax) and inBetween(line[1][1], ymin, ymax):
        return True
    # check intersect
    elif _isIntersect(line, bBot) or _isIntersect(line, bLeft) \
            or _isIntersect(line, bRight) or _isIntersect(line, bTop):
        return True
    # otherwise outside
    else:
        return False


def findNearbyRids(pt, bufferDist, xmin, ymax, gridLen, roads, grid_to_road):
    nX, nY = getGridNumForPoint(pt, xmin, ymax, gridLen)
    relatedRids = set([])
    for i_nX in range(nX - 1, nX + 2):
        for i_nY in range(nY - 1, nY + 2):
            if (i_nX, i_nY) in grid_to_road:
                relatedRids.update(grid_to_road[(i_nX, i_nY)])
    nearbyRids = {}
    for rid in relatedRids:
        distToRoad = ptDistToRoad(pt, roads[rid].shape_pts)
        if distToRoad <= bufferDist:
            nearbyRids[rid] = distToRoad
    return nearbyRids


def getWithinRoadDist(pt, rid, pointType, roads):
    DistFromStartOrToEnd = 0.
    distToRoad, jun_pt, jun_lineSegNum = ptDistToRoad(pt, roads[rid].shape_pts, distanceOnly=False)
    if pointType == 'start':
        relatedPts = roads[rid].shape_pts[jun_lineSegNum + 1:]
        for i in range(len(relatedPts)):
            if i == 0:
                DistFromStartOrToEnd += EuDist(jun_pt, relatedPts[i])
            else:
                DistFromStartOrToEnd += EuDist(relatedPts[i - 1], relatedPts[i])
    elif pointType == 'end':
        relatedPts = roads[rid].shape_pts[:jun_lineSegNum + 1]
        for i in range(len(relatedPts)):
            if i == len(relatedPts) - 1:
                DistFromStartOrToEnd += EuDist(jun_pt, relatedPts[i])
            else:
                DistFromStartOrToEnd += EuDist(relatedPts[i], relatedPts[i + 1])
    else:
        raise Exception('ValueError: unrecongized pointType: %s' % pointType)
    return DistFromStartOrToEnd


def getMaxSpd(roadType, unit='m/s'):
    sptLim = {'motorway': 150, 'motorway_link': 150,
              "trunk": 90, "trunk_link": 90,
              "primary": 60, "primary_link": 60,
              "others": 40}
    if unit == 'm/s':
        for k in sptLim:
            sptLim[k] /= 3.6
    if roadType in sptLim:
        return sptLim[roadType]
    else:
        return sptLim['others']


def possibleNextRoadSegIDs(preRid, preDist, timeDiff, roads, maxDepth=10, ptSeqTrack=None):
    maxTimeCost = timeDiff - preDist / getMaxSpd(roads[preRid].roadType)
    possibleRids = set([preRid])
    depth = 0
    if maxTimeCost > 0:
        currentLevelRids = [preRid]
        currentLevelTimeCost = [0.]
        currentLevelEndFlag = [1]
        nextLevelRids = []
        nextLevelTimeCost = []
        nextLevelEndFlag = []
        while len(currentLevelRids) > 0:
            for i in range(len(currentLevelRids)):
                if currentLevelEndFlag[i] != 0:
                    cRid = currentLevelRids[i]
                    for nRid in roads[cRid].nextRids:
                        nextLevelRids.append(nRid)
                        nRidObj = roads[nRid]
                        nextLevelTimeCost.append(currentLevelTimeCost[i] + nRidObj.length / getMaxSpd(nRidObj.roadType))
                        if nextLevelTimeCost[-1] >= maxTimeCost:
                            nextLevelEndFlag.append(0)
                        else:
                            nextLevelEndFlag.append(1)
                        possibleRids.add(nRid)
            currentLevelRids = nextLevelRids
            currentLevelTimeCost = nextLevelTimeCost
            currentLevelEndFlag = nextLevelEndFlag
            nextLevelRids = []
            nextLevelTimeCost = []
            nextLevelEndFlag = []
            depth += 1
            if depth > maxDepth:
                break
    return possibleRids


# def isGPSLineIntersectRoad(pt1, pt2, rid, roads, extendLength=0.):
#     tpt1, tpt2 = pt1, pt2
#     if extendLength > 0.:
#         GPSVec = [pt2[0] - pt1[0], pt2[1] - pt1[1]]
#         GPSVecLen = math.sqrt(GPSVecLen[0]**2 + GPSVecLen[1]**2)
#         GPSVec = [GPSVec[0]/GPSVecLen*extendLength*0.5, GPSVec[1]/GPSVecLen*extendLength*0.5]
#         tpt2 = [tpt1[0] + GPSVec[0], tpt1[1] + GPSVec[1]]
#         tpt1 = [tpt1[0] - GPSVec[0], tpt1[1] - GPSVec[1]]
#     rid_shp = roads[rid].shape_pts
#     for m in xrange(len(roads[rid].shape_pts) - 1):
#         rpt1, rpt2 = roads[rid].shape_pts[m:m+2]
#         if _isIntersect([tpt1, tpt2], [rpt1, rpt2]):
#             return True
#     return False

def getScoreFor1Rid(pt1, pt2, rid, bufferDist, distToRoad, jun_lineSegNum, roads):
    distScore = (bufferDist - distToRoad) / bufferDist
    vGPS = (pt2[0] - pt1[0], pt2[1] - pt1[1])
    endRdPt = roads[rid].shape_pts[jun_lineSegNum + 1]
    stRdPt = roads[rid].shape_pts[jun_lineSegNum]
    vRoad = (endRdPt[0] - stRdPt[0], endRdPt[1] - stRdPt[1])
    dirScore = 1 - spatial.distance.cosine(vGPS, vRoad)
    ridScore = 0.6 * distScore + 0.4 * dirScore
    return ridScore


def getBestRidBasedOnPreviousMatchedRid(pt1, pt2, preRid, timeDiff, bufferDist, roads, pt3=None, ptSeqTrack=None):
    preGPStoRoadEndDist = getWithinRoadDist(pt1, preRid, 'start', roads)
    posRids = possibleNextRoadSegIDs(preRid, preGPStoRoadEndDist, timeDiff, roads, ptSeqTrack=ptSeqTrack)
    maxScore = 0.
    bestRid = -1
    for rid in posRids:
        distToRoad, jun_pt, jun_lineSegNum = ptDistToRoad(pt2, roads[rid].shape_pts, distanceOnly=False)
        if distToRoad <= bufferDist:
            ridScore = getScoreFor1Rid(pt1, pt2, rid, bufferDist, distToRoad, jun_lineSegNum, roads)
            if pt3 != None:
                distToRoad, jun_pt, jun_lineSegNum = ptDistToRoad(pt3, roads[rid].shape_pts, distanceOnly=False)
                if distToRoad <= bufferDist:
                    ridScore += getScoreFor1Rid(pt2, pt3, rid, bufferDist, distToRoad, jun_lineSegNum, roads)
            if maxScore < ridScore:
                maxScore = ridScore
                bestRid = rid
    return bestRid


def matchSubTrip(xyList, timeList, bufferDist, timeDiffThd, startIndex, preValidPtIndex, xmin, ymax, gridLen, roads,
                 grid_to_road, MMRids, MMPtFlags, debug=False):
    # MMFlag:
    # 3: single point subtrip
    # 2: subtrip start point
    # 1: subtrip end point
    # 0: normal point
    # -1: duplicated point same as previous point
    # -2: noise (no possible trajectory to connect it from previous normal GPS point
    #           given an upper-bound based on time and max speed.
    #           It should be removed))
    #     print 'new traj: %d, %d' % (startIndex, preValidPtIndex)
    # set map-matching flag of the beginning point as the start of a new sub-trajectory
    if startIndex == 0:
        preValidPtIndex = -1
        checkStartPtFlag = True
    else:
        if xyList[startIndex] != xyList[preValidPtIndex]:
            preValidPtIndex = -1
            checkStartPtFlag = True
        else:
            checkStartPtFlag = False
    startValidPtIndex = -1
    #     print 'start preValidPtIndex %d' % preValidPtIndex
    for i in range(startIndex, len(xyList)):
        # indicating start of subtrip
        if preValidPtIndex == -1:
            # match road only based on distance to road
            posRids = findNearbyRids(xyList[i], bufferDist, xmin, ymax, gridLen, roads, grid_to_road)
            if len(posRids) > 0:
                if i < len(xyList) - 1:
                    MMPtFlags.append(2)
                else:
                    MMPtFlags.append(3)
                MMRids.append(min(list(posRids.keys()), key=lambda x: posRids[x]))
                preValidPtIndex = i
                startValidPtIndex = i
            else:
                MMPtFlags.append(-2)
                MMRids.append(-1)
        # indicating a point has been matched to a road segment, start matching to following points
        else:
            timeDiff = (timeList[i] - timeList[preValidPtIndex]) / np.timedelta64(1, 's')
            if xyList[i] == xyList[preValidPtIndex]:
                # if same position as previous valid point, copy previous map-match point result
                if timeDiff == 0:
                    # duplicated point
                    MMPtFlags.append(-1)
                    MMRids.append(MMRids[preValidPtIndex])
                else:
                    # update the end of previous trajectory to extend it if a vehicle stay
                    # in same place as previous trajectory end point, no matter how long it is
                    if MMPtFlags[preValidPtIndex] == 1:
                        MMPtFlags[preValidPtIndex] = 0
                    elif MMPtFlags[preValidPtIndex] == 3:
                        MMPtFlags[preValidPtIndex] = 2
                    if i - len(MMRids) > 0:
                        # for points with noise in-between
                        for j in range(len(MMRids), i):
                            MMRids.append(-1)
                            MMPtFlags.append(-2)
                    MMPtFlags.append(0)
                    MMRids.append(MMRids[preValidPtIndex])
                    preValidPtIndex = i

            else:
                # if different position as previous valid point
                if timeDiff > timeDiffThd:
                    if MMPtFlags[preValidPtIndex] == 2:
                        # start point become single trajectory point
                        MMPtFlags[preValidPtIndex] = 3
                    elif MMPtFlags[preValidPtIndex] == 0:
                        # normal trajectory point become end point
                        MMPtFlags[preValidPtIndex] = 1
                    else:
                        pass
                    break
                else:
                    if i < len(xyList) - 1:
                        nextValidPt = i + 1
                        # make sure the next point used to adjust map-matching is a valid and location different point
                        posRids_next = findNearbyRids(xyList[nextValidPt], bufferDist, xmin, ymax, gridLen, roads,
                                                      grid_to_road)
                        while (len(posRids_next) == 0 or xyList[nextValidPt] == xyList[i]):
                            nextValidPt += 1
                            if nextValidPt == len(xyList):
                                break
                            posRids_next = findNearbyRids(xyList[nextValidPt], bufferDist, xmin, ymax, gridLen, roads,
                                                          grid_to_road)
                        if nextValidPt <= len(xyList) - 1:
                            npt = xyList[nextValidPt]
                        else:
                            npt = None
                    else:
                        npt = None
                    bestRid = getBestRidBasedOnPreviousMatchedRid(xyList[preValidPtIndex], xyList[i],
                                                                  MMRids[preValidPtIndex], timeDiff,
                                                                  bufferDist, roads, pt3=npt)
                    if bestRid != -1:
                        if i - len(MMRids) > 0:
                            # for points with noise in-between
                            for j in range(len(MMRids), i):
                                MMRids.append(-1)
                                MMPtFlags.append(-2)
                        MMRids.append(bestRid)
                        MMPtFlags.append(0)
                        preValidPtIndex = i
                        # to deal with beginning points matched to the opposite direction of bi-direction road
                        if checkStartPtFlag:
                            checkStartPtFlag = False
                            if MMRids[startValidPtIndex] == roads[bestRid].uTurnID:
                                for begin_i in range(startValidPtIndex, i):
                                    if MMPtFlags[begin_i] >= -1:
                                        MMRids[begin_i] = bestRid
                                        if debug:
                                            print('begining pts are fixed to right directions road')
        # deal with the last map-matched trajectory point
        if i == len(xyList) - 1:
            if MMPtFlags[preValidPtIndex] == 2:
                MMPtFlags[preValidPtIndex] = 3
            elif MMPtFlags[preValidPtIndex] == 0:
                MMPtFlags[preValidPtIndex] = 1
            else:
                pass
        if debug:
            print('pt %d processed' % (i + 1))
            print('%d matched' % len(MMRids))
            print('last rid: %d' % MMRids[-1])
            print('last flag: %d' % MMPtFlags[-1])
            print('preValidPtIndex: %d\n' % preValidPtIndex)
    return MMRids, MMPtFlags, preValidPtIndex


def map_match(xyList, timeList, bufferDist, timeDiffThd, xmin, ymax, gridLen, roads, grid_to_road, debug=False):
    st = gct()
    MMRoadIDs, MMFlags = [], []
    preValidPtIndex = -1
    preLen = 0
    currenLen = 0
    while len(MMRoadIDs) < len(xyList):
        MMRoadIDs, MMFlags, preValidPtIndex = matchSubTrip(xyList, timeList, bufferDist, timeDiffThd, len(MMRoadIDs),
                                                           preValidPtIndex,
                                                           xmin, ymax, gridLen, roads, grid_to_road, MMRoadIDs, MMFlags,
                                                           debug=debug)
        currentLen = len(MMRoadIDs)
        if currentLen == preLen:
            raise RuntimeError("Stuck in while loop")
        preLen = currentLen
        if debug:
            print(len(MMRoadIDs), len(xyList), preValidPtIndex)
            print("Map-matching runtime: %s" % (gct() - st))
    return MMRoadIDs, MMFlags


def DijkstraSP(roads, startRid, start_pt, endRid, end_pt, weighted=False):
    # implmeneted in loop since recursion is memory expensive
    # the inputs roads should be road segmemnts in those grids related to start and end point
    # sanity check
    if startRid not in roads:
        raise ValueError("Dijkstra: start road segment not in input roads")
    if endRid not in roads:
        raise ValueError("Dijkstra: end road segment not in input roads")
    # distance of mapped point from/to the mapped segment's start/end
    stWeight = getWithinRoadDist(start_pt, startRid, 'start', roads)
    edWeight = getWithinRoadDist(end_pt, endRid, 'end', roads)
    # in case start gps point and end gps point are projected onto the same road
    if startRid == endRid:
        # in case two points are projected on the wrong direction of the road
        mmPoint_dist = stWeight + edWeight - roads[startRid].length
        if mmPoint_dist < 0.0:
            return float('inf'), [startRid, endRid], [stWeight, edWeight]
        else:
            if weighted:
                return mmPoint_dist, [startRid, endRid], [stWeight, edWeight]
            else:
                return 0.0, [startRid, endRid], [stWeight, edWeight]
    else:
        # find shortest path
        current = startRid
        _SPParents = {current: current}  # parent of each node in shortest path
        _SPDistance = {rid: float('inf') for rid in list(roads.keys())}  # shortest distance to segment
        _unvisited = set(roads.keys())
        _SPDistance[current] = 0.
        while True:
            for rid in roads[current].nextRids:
                if rid in roads:
                    if weighted:
                        dist = _SPDistance[current] + roads[rid].length
                    else:
                        dist = _SPDistance[current] + 1.0
                    if dist < _SPDistance[rid]:
                        _SPDistance[rid] = dist
                        _SPParents[rid] = current
            _unvisited.remove(current)
            if current == endRid or all(_SPDistance[rid] == float('inf') for rid in _unvisited):
                break
            else:
                current = min(_unvisited, key=lambda x: _SPDistance[x])
        if current == endRid:
            if weighted:
                SPDist = _SPDistance[endRid] - roads[endRid].length + stWeight + edWeight
            else:
                SPDist = _SPDistance[endRid]
            _SPPath = [endRid]
            rid = endRid
            while True:
                p = _SPParents[rid]
                _SPPath.append(p)
                rid = p
                if rid == startRid:
                    break
            _SPPath = [rid for rid in reversed(_SPPath)]
            return SPDist, _SPPath, [stWeight, edWeight]
        else:
            return float('inf'), [], []


def visualByPt(trip, figDir, SPSearchGridExBand, xmin, ymax, gridLen, grid_to_road, roads):
    # visualize map-matched trajectory for each point in given GPS points
    xyList = trip[['x', 'y']].values.tolist()
    MMRids = trip["road"].values.tolist()
    MMFlags = trip["MMFlags"].values.tolist()
    # determine the grids bounding the trip and find related road segments
    bbox_pts = []
    for pt in xyList:
        bbox_pts.append(getGridNumForPoint(pt, xmin, ymax, gridLen))
    bbox_pts = list(zip(*bbox_pts))
    nX_min, nY_min, nX_max, nY_max = min(bbox_pts[0]) - 1, min(bbox_pts[1]) - 1, max(bbox_pts[0]) + 1, max(
        bbox_pts[1]) + 1
    relatedRoads = set()
    for nx in range(nX_min, nX_max + 1):
        for ny in range(nY_min, nY_max + 1):
            if (nx, ny) in grid_to_road:
                relatedRoads = relatedRoads | set(grid_to_road[(nx, ny)])
    # set figure size
    delta_nX = nX_max - nX_min + 1
    delta_nY = nY_max - nY_min + 1
    figSizeRatio = 1.0
    ptSizeRatio = (delta_nX + delta_nY) / 2.0 * figSizeRatio
    figSize = (delta_nX * figSizeRatio, delta_nY * figSizeRatio)
    # interpolate trajectory between matched points
    for i in range(len(MMFlags) - 1):
        if MMFlags[i] == 2 or MMFlags[i] == 0:
            nextValidPt = i + 1
            while MMFlags[nextValidPt] <= -1:
                nextValidPt += 1
            startPt = xyList[i]
            startRid = MMRids[i]
            endPt = xyList[nextValidPt]
            endRid = MMRids[nextValidPt]
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
            spRoads = {}
            for rid in relatedRids:
                spRoads[rid] = roads[rid]
            traj = DijkstraSP(spRoads, MMRids[i], xyList[i], MMRids[nextValidPt], xyList[nextValidPt])[1]
            # plot
            fig, ax = plt.subplots(1, 1, figsize=figSize)
            plotRoads = []
            for rid in relatedRoads:
                plotRoads.extend(list(zip(*roads[rid].shape_pts)))
                plotRoads.append('b')
            for rid in traj:
                plotRoads.extend(list(zip(*roads[rid].shape_pts)))
                plotRoads.append('g')
            for rid in [traj[0], traj[-1]]:
                if rid > 0:
                    plotRoads.extend(list(zip(*roads[rid].shape_pts)))
                    plotRoads.append('r')
            ax.set_xlim(xmin + nX_min * gridLen, xmin + (nX_max + 1) * gridLen)
            ax.set_ylim(ymax - (nY_max + 1) * gridLen, ymax - nY_min * gridLen)
            ax.plot(*plotRoads, lw=0.3)
            plotPts = list(zip(startPt, endPt))
            ax.scatter(plotPts[0], plotPts[1], s=0.5 * ptSizeRatio, facecolor='none', linewidth=0.04 * ptSizeRatio,
                       edgecolors=[(1, 0, 0, 0.2 + round((m + 1) * 0.8 / len(plotPts), 1)) for m in
                                   range(len(plotPts))])
            if nextValidPt - i > 1:
                plotPts = list(zip(*xyList[i + 1: nextValidPt]))
                ax.scatter(plotPts[0], plotPts[1], s=0.5 * ptSizeRatio, facecolor='none', linewidth=0.04 * ptSizeRatio,
                           edgecolors=[(1, 1, 0, 0.2 + round((m + 1) * 0.8 / len(plotPts), 1)) for m in
                                       range(len(plotPts))])
            figName = '%d-%d.png' % (i, nextValidPt)
            plt.savefig(os.path.join(figDir, figName), dpi=300)
            plt.close(fig)
