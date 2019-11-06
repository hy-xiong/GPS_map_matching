# GPS_map_matching

GPS data:
- Table
- Columns: Driver ID, Order ID, Timestamp, Lat, Lon

Road network: GeoJson format

Scripts running sequence:
1. ExtractGPSInBound.py: extract GPS within given road network bound
2. sortGPS.py: sort GPS by Driver ID & Order ID & Timestamp
3. SplitGPS.py: split daily GPS into smaller datasets by # of points
4. Map_matching.py (function lib: Map_matching_ult.py): map-matching GPS with a greedy algorithm
5. MergeResults.py: merge map-matching result back to daily result
