import logging
from typing import Optional, Any, List, Tuple

from osgeo import gdal, osr

from python.tilejobinfo import TileJobInfo

Options = Any
logger = logging.getLogger("preprocess")

class GDALError(Exception):
    pass

# 预处理，重投影数据集
def reproject_dataset(
    from_dataset: gdal.Dataset,
    from_srs: Optional[osr.SpatialReference],
    to_srs: Optional[osr.SpatialReference],
    options: Optional[Options] = None,
) -> gdal.Dataset:
    """
    Returns the input dataset in the expected "destination" SRS.
    If the dataset is already in the correct SRS, returns it unmodified
    """
    if not from_srs or not to_srs:
        raise GDALError("from and to SRS must be defined to reproject the dataset")

    if (from_srs.ExportToProj4() != to_srs.ExportToProj4()) or (
        from_dataset.GetGCPCount() != 0
    ):

        if (
            from_srs.IsGeographic()
            and to_srs.GetAuthorityName(None) == "EPSG"
            and to_srs.GetAuthorityCode(None) == "3857"
        ):
            from_gt = from_dataset.GetGeoTransform(can_return_null=True)
            if from_gt and from_gt[2] == 0 and from_gt[4] == 0 and from_gt[5] < 0:
                minlon = from_gt[0]
                maxlon = from_gt[0] + from_dataset.RasterXSize * from_gt[1]
                maxlat = from_gt[3]
                minlat = from_gt[3] + from_dataset.RasterYSize * from_gt[5]
                MAX_LAT = 85.0511287798066
                adjustBounds = False
                if minlon < -180.0:
                    minlon = -180.0
                    adjustBounds = True
                if maxlon > 180.0:
                    maxlon = 180.0
                    adjustBounds = True
                if maxlat > MAX_LAT:
                    maxlat = MAX_LAT
                    adjustBounds = True
                if minlat < -MAX_LAT:
                    minlat = -MAX_LAT
                    adjustBounds = True
                if adjustBounds:
                    ct = osr.CoordinateTransformation(from_srs, to_srs)
                    west, south = ct.TransformPoint(minlon, minlat)[:2]
                    east, north = ct.TransformPoint(maxlon, maxlat)[:2]
                    return gdal.Warp(
                        "",
                        from_dataset,
                        format="VRT",
                        outputBounds=[west, south, east, north],
                        srcSRS=from_srs.ExportToWkt(),
                        dstSRS="EPSG:3857",
                    )

        to_dataset = gdal.AutoCreateWarpedVRT(
            from_dataset, from_srs.ExportToWkt(), to_srs.ExportToWkt()
        )

        if options and options.verbose:
            logger.debug(
                "Warping of the raster by AutoCreateWarpedVRT (result saved into 'tiles.vrt')"
            )
            to_dataset.GetDriver().CreateCopy("tiles.vrt", to_dataset)

        return to_dataset
    else:
        return from_dataset

# 预处理，计算波段数量
def nb_data_bands(dataset: gdal.Dataset) -> int:
    """
    Return the number of data (non-alpha) bands of a gdal dataset
    """
    alphaband = dataset.GetRasterBand(1).GetMaskBand()
    if (
        (alphaband.GetMaskFlags() & gdal.GMF_ALPHA)
        or dataset.RasterCount == 4
        or dataset.RasterCount == 2
    ):
        return dataset.RasterCount - 1
    return dataset.RasterCount

# 预处理，nodata值(被是为透明的值)
def setup_no_data_values(input_dataset: gdal.Dataset, options: Options) -> List[float]:
    in_nodata = []
    if options.srcnodata:
        nds = list(map(float, options.srcnodata.split(",")))
        if len(nds) < input_dataset.RasterCount:
            in_nodata = (nds * input_dataset.RasterCount)[: input_dataset.RasterCount]
        else:
            in_nodata = nds
    else:
        for i in range(1, input_dataset.RasterCount + 1):
            band = input_dataset.GetRasterBand(i)
            raster_no_data = band.GetNoDataValue()
            if raster_no_data is not None:
                # Ignore nodata values that are not in the range of the band data type (see https://github.com/OSGeo/gdal/pull/2299)
                if band.DataType == gdal.GDT_Byte and (
                    raster_no_data != int(raster_no_data)
                    or raster_no_data < 0
                    or raster_no_data > 255
                ):
                    # We should possibly do similar check for other data types
                    in_nodata = []
                    break
                in_nodata.append(raster_no_data)

    if options.verbose:
        logger.debug("NODATA: %s" % in_nodata)

    return in_nodata

# 预处理，输入坐标系
def setup_input_srs(
    input_dataset: gdal.Dataset, options: Options
) -> Tuple[Optional[osr.SpatialReference], Optional[str]]:
    """
    Determines and returns the Input Spatial Reference System (SRS) as an osr object and as a
    WKT representation

    Uses in priority the one passed in the command line arguments. If None, tries to extract them
    from the input dataset
    """

    input_srs = None
    input_srs_wkt = None

    if options.s_srs:
        input_srs = osr.SpatialReference()
        try:
            input_srs.SetFromUserInput(options.s_srs)
        except RuntimeError:
            raise ValueError("Invalid value for --s_srs option")
        input_srs_wkt = input_srs.ExportToWkt()
    else:
        input_srs_wkt = input_dataset.GetProjection()
        if not input_srs_wkt and input_dataset.GetGCPCount() != 0:
            input_srs_wkt = input_dataset.GetGCPProjection()
        if input_srs_wkt:
            input_srs = osr.SpatialReference()
            input_srs.ImportFromWkt(input_srs_wkt)

    if input_srs is not None:
        input_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    return input_srs, input_srs_wkt




# 预处理，判断数据集是否包含六参数
def has_georeference(dataset: gdal.Dataset) -> bool:
    return (
        dataset.GetGeoTransform() != (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        or dataset.GetGCPCount() != 0
    )

# 预处理，计算非顶层瓦片数量
def count_overview_tiles(tile_job_info: "TileJobInfo") -> int:
    tile_number = 0
    for tz in range(tile_job_info.tmaxz - 1, tile_job_info.tminz - 1, -1):
        tminx, tminy, tmaxx, tmaxy = tile_job_info.tminmax[tz]
        tile_number += (1 + abs(tmaxx - tminx)) * (1 + abs(tmaxy - tminy))

    return tile_number
