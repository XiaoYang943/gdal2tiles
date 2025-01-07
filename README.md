## source
- source from [gdal3.10.0](https://github.com/OSGeo/gdal/blob/master/swig/python/gdal-utils/osgeo_utils/gdal2tiles.py)
## how to run
1. install gdal whl from [geospatial-wheels](https://github.com/cgohlke/geospatial-wheels/releases)
   - For example:`GDAL-3.9.2-cp312-cp312-win_amd64.whl`
2. install anaconda、start anaconda prompt
   - cd to whl folder
   - execute `pip install GDAL-3.9.2-cp312-cp312-win_amd64.whl`
3. set app configuration
   - -p mercator -z 0-16 --processes 12 "D:\data\raster\dom\0724-5-0727-1dom.tif" "D:\data\raster\tiles\gdal2tiles\python1"
   - [detail](https://gdal.org/en/stable/programs/gdal2tiles.html#synopsis)
4. run main

## TODO
1. performance test
2. log output
