from osgeo import gdal

from src.log.log import exit_with_error

try:
    import numpy
    from PIL import Image

    import osgeo.gdal_array as gdalarray

    numpy_available = True
except ImportError:
    numpy_available = False


# 将查询数据集缩放到瓦片数据集，把高层级的四张合并为一张
def scale_query_to_tile(dsquery, dstile, options, tilefilename=""):
    """Scales down query dataset to the tile dataset"""

    querysize = dsquery.RasterXSize
    tile_size = dstile.RasterXSize
    tilebands = dstile.RasterCount

    dsquery.SetGeoTransform(
        (
            0.0,
            tile_size / float(querysize),
            0.0,
            0.0,
            0.0,
            tile_size / float(querysize),
        )
    )
    # 设置默认仿射变换参数到目标数据集
    dstile.SetGeoTransform((0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    if options.resampling == "average" and (
            options.excluded_values or options.nodata_values_pct_threshold < 100
    ):

        warp_options = "-r average"

        assert options.nodata_values_pct_threshold is not None
        warp_options += (
            f" -wo NODATA_VALUES_PCT_THRESHOLD={options.nodata_values_pct_threshold}"
        )

        if options.excluded_values:
            assert options.excluded_values_pct_threshold is not None
            warp_options += f" -wo EXCLUDED_VALUES={options.excluded_values}"
            warp_options += f" -wo EXCLUDED_VALUES_PCT_THRESHOLD={options.excluded_values_pct_threshold}"

        gdal.Warp(
            dstile,
            dsquery,
            options=warp_options,
        )

    elif options.resampling == "average":

        # Function: gdal.RegenerateOverview()
        for i in range(1, tilebands + 1):
            # Black border around NODATA
            res = gdal.RegenerateOverview(
                dsquery.GetRasterBand(i), dstile.GetRasterBand(i), "average"
            )
            if res != 0:
                exit_with_error(
                    "RegenerateOverview() failed on %s, error %d" % (tilefilename, res)
                )
    else:

        if options.resampling == "near":
            gdal_resampling = gdal.GRA_NearestNeighbour

        elif options.resampling == "bilinear":
            gdal_resampling = gdal.GRA_Bilinear

        elif options.resampling == "cubic":
            gdal_resampling = gdal.GRA_Cubic

        elif options.resampling == "cubicspline":
            gdal_resampling = gdal.GRA_CubicSpline

        elif options.resampling == "lanczos":
            gdal_resampling = gdal.GRA_Lanczos

        elif options.resampling == "mode":
            gdal_resampling = gdal.GRA_Mode

        elif options.resampling == "max":
            gdal_resampling = gdal.GRA_Max

        elif options.resampling == "min":
            gdal_resampling = gdal.GRA_Min

        elif options.resampling == "med":
            gdal_resampling = gdal.GRA_Med

        elif options.resampling == "q1":
            gdal_resampling = gdal.GRA_Q1

        elif options.resampling == "q3":
            gdal_resampling = gdal.GRA_Q3

        # Other algorithms are implemented by gdal.ReprojectImage().
        res = gdal.ReprojectImage(dsquery, dstile, None, None, gdal_resampling)
        if res != 0:
            exit_with_error(
                "ReprojectImage() failed on %s, error %d" % (tilefilename, res)
            )
