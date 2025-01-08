import os

from osgeo import gdal

from src.log.log import exit_with_error

try:
    import numpy
    from PIL import Image

    import osgeo.gdal_array as gdalarray

    numpy_available = True
except ImportError:
    # 'antialias' resampling is not available
    numpy_available = False


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

    elif options.resampling == "antialias" and numpy_available:

        if tilefilename.startswith("/vsi"):
            raise Exception(
                "Outputting to /vsi file systems with antialias mode is not supported"
            )

        # Scaling by PIL (Python Imaging Library) - improved Lanczos
        array = numpy.zeros((querysize, querysize, tilebands), numpy.uint8)
        for i in range(tilebands):
            array[:, :, i] = gdalarray.BandReadAsArray(
                dsquery.GetRasterBand(i + 1), 0, 0, querysize, querysize
            )
        if options.tiledriver == "JPEG" and tilebands == 2:
            im = Image.fromarray(array[:, :, 0], "L")
        elif options.tiledriver == "JPEG" and tilebands == 4:
            im = Image.fromarray(array[:, :, 0:3], "RGB")
        else:
            im = Image.fromarray(array, "RGBA")
        im1 = im.resize((tile_size, tile_size), Image.LANCZOS)
        if os.path.exists(tilefilename):
            im0 = Image.open(tilefilename)
            im1 = Image.composite(im1, im0, im1)

        params = {}
        if options.tiledriver == "WEBP":
            if options.webp_lossless:
                params["lossless"] = True
            else:
                params["quality"] = options.webp_quality
        elif options.tiledriver == "JPEG":
            params["quality"] = options.jpeg_quality
        im1.save(tilefilename, options.tiledriver, **params)

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
