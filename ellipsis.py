import ellipsis as el

#ids for my point cloud
pathId = 'fe5cb352-ccda-470b-bb66-12631c028def'
timestampId = '7dd94eac-f145-4a8a-b92d-0a22a289fe21'

#extent to grab my area of interest
extent = {    'xMin':5.7908,'xMax':5.79116,'yMin':51.93303,'yMax':51.93321}
epsg = 4326

#with the fetchPoints function I can retrieve all points within my extent
df = el.path.pointCloud.timestamp.fetchPoints(pathId = pathId, timestampId = timestampId, extent = extent,  epsg = epsg)

print(df)

el.util.plotPointCloud(df, method = 'cloud')


