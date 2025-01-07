#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ******************************************************************************
#
# Project:  Google Summer of Code 2007, 2008 (http://code.google.com/soc/)
# Support:  BRGM (http://www.brgm.fr)
# Purpose:  Convert a raster into TMS (Tile Map Service) tiles in a directory.
#           - generate Google Earth metadata (KML SuperOverlay)
#           - generate simple HTML viewer based on Google Maps and OpenLayers
#           - support of global tiles (Spherical Mercator) for compatibility
#               with interactive web maps a la Google Maps
# Author:   Klokan Petr Pridal, klokan at klokan dot cz
#
###############################################################################
# Copyright (c) 2008, Klokan Petr Pridal
# Copyright (c) 2010-2013, Even Rouault <even dot rouault at spatialys.com>
# Copyright (c) 2021, Idan Miara <idan@miara.com>
#
# SPDX-License-Identifier: MIT
# ******************************************************************************

r"""
TMS坐标原点在左下角
Google Maps坐标原点在左上角
"""

import contextlib
import glob
import json
import logging
import math
import optparse
import os
import shutil
import stat
import sys
import tempfile
import threading
from functools import partial
from typing import Any, List, NoReturn, Optional, Tuple
from uuid import uuid4
from xml.etree import ElementTree

from osgeo import gdal, osr

from python.constant import MAXZOOMLEVEL
from python.globalgeodetic import GlobalGeodetic
from python.globalmercator import GlobalMercator
from python.tiledetail import TileDetail
from python.tilematrixset import TileMatrixSet, UnsupportedTileMatrixSet

# from osgeo_utils.auxiliary.util import enable_gdal_exceptions

Options = Any

try:
    import numpy
    from PIL import Image

    import osgeo.gdal_array as gdalarray

    numpy_available = True
except ImportError:
    # 'antialias' resampling is not available
    numpy_available = False

__version__ = gdal.__version__

resampling_list = (
    "average",
    "near",
    "bilinear",
    "cubic",
    "cubicspline",
    "lanczos",
    "antialias",
    "mode",
    "max",
    "min",
    "med",
    "q1",
    "q3",
)

logger = logging.getLogger("gdal2tiles")


def makedirs(path):
    """Wrapper for os.makedirs() that can work with /vsi files too"""
    if path.startswith("/vsi"):
        if gdal.MkdirRecursive(path, 0o755) != 0:
            raise Exception(f"Cannot create {path}")
    else:
        os.makedirs(path, exist_ok=True)


def isfile(path):
    """Wrapper for os.path.isfile() that can work with /vsi files too"""
    if path.startswith("/vsi"):
        stat_res = gdal.VSIStatL(path)
        if stat is None:
            return False
        return stat.S_ISREG(stat_res.mode)
    else:
        return os.path.isfile(path)


class VSIFile:
    """Expose a simplistic file-like API for a /vsi file"""

    def __init__(self, filename, f):
        self.filename = filename
        self.f = f

    def write(self, content):
        if gdal.VSIFWriteL(content, 1, len(content), self.f) != len(content):
            raise Exception("Error while writing into %s" % self.filename)


@contextlib.contextmanager
def my_open(filename, mode):
    """Wrapper for open() built-in method that can work with /vsi files too"""
    if filename.startswith("/vsi"):
        f = gdal.VSIFOpenL(filename, mode)
        if f is None:
            raise Exception(f"Cannot open {filename} in {mode}")
        try:
            yield VSIFile(filename, f)
        finally:
            if gdal.VSIFCloseL(f) != 0:
                raise Exception(f"Cannot close {filename}")
    else:
        yield open(filename, mode)





tmsMap = {}


def get_profile_list():
    profile_list = ["mercator", "geodetic", "raster"]

    # Read additional tile matrix sets from GDAL data directory
    filename = gdal.FindFile("gdal", "tms_MapML_APSTILE.json")
    if filename:
        dirname = os.path.dirname(filename)
        for tmsfilename in glob.glob(os.path.join(dirname, "tms_*.json")):
            data = open(tmsfilename, "rb").read()
            try:
                j = json.loads(data.decode("utf-8"))
            except Exception:
                j = None
            if j is None:
                logger.error("Cannot parse " + tmsfilename)
                continue
            try:
                tms = TileMatrixSet.parse(j)
            except UnsupportedTileMatrixSet as e:
                gdal.Debug("gdal2tiles", "Cannot parse " + tmsfilename + ": " + str(e))
                continue
            except Exception:
                logger.error("Cannot parse " + tmsfilename)
                continue
            tmsMap[tms.identifier] = tms
            profile_list.append(tms.identifier)

    return profile_list


threadLocal = threading.local()

# =============================================================================
# =============================================================================
# =============================================================================

__doc__globalmaptiles = """
globalmaptiles.py

Global Map Tiles as defined in Tile Map Service (TMS) Profiles
==============================================================

Functions necessary for generation of global tiles used on the web.
It contains classes implementing coordinate conversions for:

  - GlobalMercator (based on EPSG:3857)
       for Google Maps, Yahoo Maps, Bing Maps compatible tiles
  - GlobalGeodetic (based on EPSG:4326)
       for OpenLayers Base Map and Google Earth compatible tiles

More info at:

http://wiki.osgeo.org/wiki/Tile_Map_Service_Specification
http://wiki.osgeo.org/wiki/WMS_Tiling_Client_Recommendation
http://msdn.microsoft.com/en-us/library/bb259689.aspx
http://code.google.com/apis/maps/documentation/overlays.html#Google_Maps_Coordinates

Created by Klokan Petr Pridal on 2008-07-03.
Google Summer of Code 2008, project GDAL2Tiles for OSGEO.

In case you use this class in your product, translate it to another language
or find it useful for your project please let me know.
My email: klokan at klokan dot cz.
I would like to know where it was used.

Class is available under the open-source GDAL license (www.gdal.org).
"""

class GDALError(Exception):
    pass


def exit_with_error(message: str, details: str = "") -> NoReturn:
    # Message printing and exit code kept from the way it worked using the OptionParser (in case
    # someone parses the error output)
    sys.stderr.write("Usage: gdal2tiles [options] input_file [output]\n\n")
    sys.stderr.write("gdal2tiles: error: %s\n" % message)
    if details:
        sys.stderr.write("\n\n%s\n" % details)

    sys.exit(2)


def set_cache_max(cache_in_bytes: int) -> None:
    # We set the maximum using `SetCacheMax` and `GDAL_CACHEMAX` to support both fork and spawn as multiprocessing start methods.
    # https://github.com/OSGeo/gdal/pull/2112
    os.environ["GDAL_CACHEMAX"] = "%d" % int(cache_in_bytes / 1024 / 1024)
    gdal.SetCacheMax(cache_in_bytes)


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

# 预处理，输出坐标系
def setup_output_srs(
    input_srs: Optional[osr.SpatialReference], options: Options
) -> Optional[osr.SpatialReference]:
    """
    Setup the desired SRS (based on options)
    """
    output_srs = osr.SpatialReference()

    if options.profile == "mercator":
        output_srs.ImportFromEPSG(3857)
    elif options.profile == "geodetic":
        output_srs.ImportFromEPSG(4326)
    elif options.profile == "raster":
        output_srs = input_srs
    else:
        output_srs = tmsMap[options.profile].srs.Clone()

    if output_srs:
        output_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    return output_srs

# 预处理，判断数据集是否包含六参数
def has_georeference(dataset: gdal.Dataset) -> bool:
    return (
        dataset.GetGeoTransform() != (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        or dataset.GetGCPCount() != 0
    )

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


def add_gdal_warp_options_to_string(vrt_string, warp_options):
    if not warp_options:
        return vrt_string

    vrt_root = ElementTree.fromstring(vrt_string)
    options = vrt_root.find("GDALWarpOptions")

    if options is None:
        return vrt_string

    for key, value in warp_options.items():
        tb = ElementTree.TreeBuilder()
        tb.start("Option", {"name": key})
        tb.data(value)
        tb.end("Option")
        elem = tb.close()
        options.insert(0, elem)

    return ElementTree.tostring(vrt_root).decode()


def update_no_data_values(
    warped_vrt_dataset: gdal.Dataset,
    nodata_values: List[float],
    options: Optional[Options] = None,
) -> gdal.Dataset:
    """
    Takes an array of NODATA values and forces them on the WarpedVRT file dataset passed
    """
    # TODO: gbataille - Seems that I forgot tests there
    assert nodata_values != []

    vrt_string = warped_vrt_dataset.GetMetadata("xml:VRT")[0]

    vrt_string = add_gdal_warp_options_to_string(
        vrt_string, {"INIT_DEST": "NO_DATA", "UNIFIED_SRC_NODATA": "YES"}
    )

    # TODO: gbataille - check the need for this replacement. Seems to work without
    #         # replace BandMapping tag for NODATA bands....
    #         for i in range(len(nodata_values)):
    #             s = s.replace(
    #                 '<BandMapping src="%i" dst="%i"/>' % ((i+1), (i+1)),
    #                 """
    # <BandMapping src="%i" dst="%i">
    # <SrcNoDataReal>%i</SrcNoDataReal>
    # <SrcNoDataImag>0</SrcNoDataImag>
    # <DstNoDataReal>%i</DstNoDataReal>
    # <DstNoDataImag>0</DstNoDataImag>
    # </BandMapping>
    #                 """ % ((i+1), (i+1), nodata_values[i], nodata_values[i]))

    corrected_dataset = gdal.Open(vrt_string)

    # set NODATA_VALUE metadata
    corrected_dataset.SetMetadataItem(
        "NODATA_VALUES", " ".join([str(i) for i in nodata_values])
    )

    if options and options.verbose:
        logger.debug("Modified warping result saved into 'tiles1.vrt'")

        with open("tiles1.vrt", "w") as f:
            f.write(corrected_dataset.GetMetadata("xml:VRT")[0])

    return corrected_dataset


def add_alpha_band_to_string_vrt(vrt_string: str) -> str:
    # TODO: gbataille - Old code speak of this being equivalent to gdalwarp -dstalpha
    # To be checked

    vrt_root = ElementTree.fromstring(vrt_string)

    index = 0
    nb_bands = 0
    for subelem in list(vrt_root):
        if subelem.tag == "VRTRasterBand":
            nb_bands += 1
            color_node = subelem.find("./ColorInterp")
            if color_node is not None and color_node.text == "Alpha":
                raise Exception("Alpha band already present")
        else:
            if nb_bands:
                # This means that we are one element after the Band definitions
                break

        index += 1

    tb = ElementTree.TreeBuilder()
    tb.start(
        "VRTRasterBand",
        {
            "dataType": "Byte",
            "band": str(nb_bands + 1),
            "subClass": "VRTWarpedRasterBand",
        },
    )
    tb.start("ColorInterp", {})
    tb.data("Alpha")
    tb.end("ColorInterp")
    tb.end("VRTRasterBand")
    elem = tb.close()

    vrt_root.insert(index, elem)

    warp_options = vrt_root.find(".//GDALWarpOptions")
    tb = ElementTree.TreeBuilder()
    tb.start("DstAlphaBand", {})
    tb.data(str(nb_bands + 1))
    tb.end("DstAlphaBand")
    elem = tb.close()
    warp_options.append(elem)

    # TODO: gbataille - this is a GDALWarpOptions. Why put it in a specific place?
    tb = ElementTree.TreeBuilder()
    tb.start("Option", {"name": "INIT_DEST"})
    tb.data("0")
    tb.end("Option")
    elem = tb.close()
    warp_options.append(elem)

    return ElementTree.tostring(vrt_root).decode()


def update_alpha_value_for_non_alpha_inputs(
    warped_vrt_dataset: gdal.Dataset, options: Optional[Options] = None
) -> gdal.Dataset:
    """
    Handles dataset with 1 or 3 bands, i.e. without alpha channel, in the case the nodata value has
    not been forced by options
    """
    if warped_vrt_dataset.RasterCount in [1, 3]:

        vrt_string = warped_vrt_dataset.GetMetadata("xml:VRT")[0]

        vrt_string = add_alpha_band_to_string_vrt(vrt_string)

        warped_vrt_dataset = gdal.Open(vrt_string)

        if options and options.verbose:
            logger.debug("Modified -dstalpha warping result saved into 'tiles1.vrt'")

            with open("tiles1.vrt", "w") as f:
                f.write(warped_vrt_dataset.GetMetadata("xml:VRT")[0])

    return warped_vrt_dataset

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


def _get_creation_options(options):
    copts = []
    if options.tiledriver == "WEBP":
        if options.webp_lossless:
            copts = ["LOSSLESS=True"]
        else:
            copts = ["QUALITY=" + str(options.webp_quality)]
    elif options.tiledriver == "JPEG":
        copts = ["QUALITY=" + str(options.jpeg_quality)]
    return copts

# 最高层级切片
def create_base_tile(tile_job_info: "TileJobInfo", tile_detail: "TileDetail") -> None:

    dataBandsCount = tile_job_info.nb_data_bands
    output = tile_job_info.output_file_path
    tileext = tile_job_info.tile_extension
    tile_size = tile_job_info.tile_size
    options = tile_job_info.options

    tilebands = dataBandsCount + 1

    cached_ds = getattr(threadLocal, "cached_ds", None)
    if cached_ds and cached_ds.GetDescription() == tile_job_info.src_file:
        ds = cached_ds
    else:
        ds = gdal.Open(tile_job_info.src_file, gdal.GA_ReadOnly)
        threadLocal.cached_ds = ds

    # MEM: gdal提供的基于内存的栅格图像格式，其大小由用户内存大小决定
    # MEM常用于作为临时文件保存中间结果，减少磁盘IO
    mem_drv = gdal.GetDriverByName("MEM")
    out_drv = gdal.GetDriverByName(tile_job_info.tile_driver)
    alphaband = ds.GetRasterBand(1).GetMaskBand()

    tx = tile_detail.tx
    ty = tile_detail.ty
    tz = tile_detail.tz
    rx = tile_detail.rx
    ry = tile_detail.ry
    rxsize = tile_detail.rxsize
    rysize = tile_detail.rysize
    wx = tile_detail.wx
    wy = tile_detail.wy
    wxsize = tile_detail.wxsize
    wysize = tile_detail.wysize
    querysize = tile_detail.querysize

    # Tile dataset in memory
    tilefilename = os.path.join(output, str(tz), str(tx), "%s.%s" % (ty, tileext))
    # 创建MEM
    dstile = mem_drv.Create("", tile_size, tile_size, tilebands)
    dstile.GetRasterBand(tilebands).SetColorInterpretation(gdal.GCI_AlphaBand)

    data = alpha = None

    if options.verbose:
        logger.debug(
            f"\tReadRaster Extent: ({rx}, {ry}, {rxsize}, {rysize}), ({wx}, {wy}, {wxsize}, {wysize})"
        )

    # Query is in 'nearest neighbour' but can be bigger in then the tile_size
    # We scale down the query to the tile_size by supplied algorithm.

    if rxsize != 0 and rysize != 0 and wxsize != 0 and wysize != 0:
        alpha = alphaband.ReadRaster(rx, ry, rxsize, rysize, wxsize, wysize)

        # Detect totally transparent tile and skip its creation
        if tile_job_info.exclude_transparent and len(alpha) == alpha.count(
            "\x00".encode("ascii")
        ):
            return

        # 从数据集中读取栅格
        data = ds.ReadRaster(
            rx,
            ry,
            rxsize,
            rysize,
            wxsize,
            wysize,
            band_list=list(range(1, dataBandsCount + 1)),
        )

    # The tile in memory is a transparent file by default. Write pixel values into it if
    # any
    if data:
        if tile_size == querysize:
            # Use the ReadRaster result directly in tiles ('nearest neighbour' query)
            dstile.WriteRaster(
                wx,
                wy,
                wxsize,
                wysize,
                data,
                band_list=list(range(1, dataBandsCount + 1)),
            )
            dstile.WriteRaster(wx, wy, wxsize, wysize, alpha, band_list=[tilebands])

            # Note: For source drivers based on WaveLet compression (JPEG2000, ECW,
            # MrSID) the ReadRaster function returns high-quality raster (not ugly
            # nearest neighbour)
            # TODO: Use directly 'near' for WaveLet files
        else:
            # Big ReadRaster query in memory scaled to the tile_size - all but 'near'
            # algo
            dsquery = mem_drv.Create("", querysize, querysize, tilebands)
            dsquery.GetRasterBand(tilebands).SetColorInterpretation(gdal.GCI_AlphaBand)

            # TODO: fill the null value in case a tile without alpha is produced (now
            # only png tiles are supported)
            dsquery.WriteRaster(
                wx,
                wy,
                wxsize,
                wysize,
                data,
                band_list=list(range(1, dataBandsCount + 1)),
            )
            dsquery.WriteRaster(wx, wy, wxsize, wysize, alpha, band_list=[tilebands])

            scale_query_to_tile(dsquery, dstile, options, tilefilename=tilefilename)
            del dsquery

    del data

    if options.resampling != "antialias":
        # Write a copy of tile to png/jpg
        out_drv.CreateCopy(
            tilefilename,
            dstile
            if tile_job_info.tile_driver != "JPEG"
            else remove_alpha_band(dstile),
            strict=0,
            options=_get_creation_options(options),
        )

        # Remove useless side car file
        aux_xml = tilefilename + ".aux.xml"
        if gdal.VSIStatL(aux_xml) is not None:
            gdal.Unlink(aux_xml)

    del dstile



def remove_alpha_band(src_ds):
    if (
        src_ds.GetRasterBand(src_ds.RasterCount).GetColorInterpretation()
        != gdal.GCI_AlphaBand
    ):
        return src_ds

    new_band_count = src_ds.RasterCount - 1

    dst_ds = gdal.GetDriverByName("MEM").Create(
        "",
        src_ds.RasterXSize,
        src_ds.RasterYSize,
        new_band_count,
        src_ds.GetRasterBand(1).DataType,
    )

    gt = src_ds.GetGeoTransform(can_return_null=True)
    if gt:
        dst_ds.SetGeoTransform(gt)
    srs = src_ds.GetSpatialRef()
    if srs:
        dst_ds.SetSpatialRef(srs)

    for i in range(1, new_band_count + 1):
        src_band = src_ds.GetRasterBand(i)
        dst_band = dst_ds.GetRasterBand(i)
        dst_band.WriteArray(src_band.ReadAsArray())

    return dst_ds

# 切片，根据顶层瓦片，构建下层瓦片
def create_overview_tile(
    base_tz: int,
    base_tiles: List[Tuple[int, int]],
    output_folder: str,
    tile_job_info: "TileJobInfo",
    options: Options,
):

    overview_tz = base_tz - 1
    overview_tx = base_tiles[0][0] >> 1
    overview_ty = base_tiles[0][1] >> 1
    overview_ty_real = GDAL2Tiles.getYTile(overview_ty, overview_tz, options)

    tilefilename = os.path.join(
        output_folder,
        str(overview_tz),
        str(overview_tx),
        "%s.%s" % (overview_ty_real, tile_job_info.tile_extension),
    )
    if options.verbose:
        logger.debug(tilefilename)
    if options.resume and isfile(tilefilename):
        if options.verbose:
            logger.debug("Tile generation skipped because of --resume")
        return

    mem_driver = gdal.GetDriverByName("MEM")
    tile_driver = tile_job_info.tile_driver
    out_driver = gdal.GetDriverByName(tile_driver)

    tilebands = tile_job_info.nb_data_bands + 1

    dsquery = mem_driver.Create(
        "", 2 * tile_job_info.tile_size, 2 * tile_job_info.tile_size, tilebands
    )
    dsquery.GetRasterBand(tilebands).SetColorInterpretation(gdal.GCI_AlphaBand)
    # TODO: fill the null value
    dstile = mem_driver.Create(
        "", tile_job_info.tile_size, tile_job_info.tile_size, tilebands
    )
    dstile.GetRasterBand(tilebands).SetColorInterpretation(gdal.GCI_AlphaBand)

    usable_base_tiles = []

    for base_tile in base_tiles:
        base_tx = base_tile[0]
        base_ty = base_tile[1]
        base_ty_real = GDAL2Tiles.getYTile(base_ty, base_tz, options)

        base_tile_path = os.path.join(
            output_folder,
            str(base_tz),
            str(base_tx),
            "%s.%s" % (base_ty_real, tile_job_info.tile_extension),
        )
        if not isfile(base_tile_path):
            continue

        dsquerytile = gdal.Open(base_tile_path, gdal.GA_ReadOnly)

        if base_tx % 2 == 0:
            tileposx = 0
        else:
            tileposx = tile_job_info.tile_size

        if options.xyz and options.profile == "raster":
            if base_ty % 2 == 0:
                tileposy = 0
            else:
                tileposy = tile_job_info.tile_size
        else:
            if base_ty % 2 == 0:
                tileposy = tile_job_info.tile_size
            else:
                tileposy = 0

        if (
            tile_job_info.tile_driver == "JPEG"
            and dsquerytile.RasterCount == 3
            and tilebands == 2
        ):
            # Input is RGB with R=G=B. Add An alpha band
            tmp_ds = mem_driver.Create(
                "", dsquerytile.RasterXSize, dsquerytile.RasterYSize, 2
            )
            tmp_ds.GetRasterBand(1).WriteRaster(
                0,
                0,
                tile_job_info.tile_size,
                tile_job_info.tile_size,
                dsquerytile.GetRasterBand(1).ReadRaster(),
            )
            mask = bytearray(
                [255] * (tile_job_info.tile_size * tile_job_info.tile_size)
            )
            tmp_ds.GetRasterBand(2).WriteRaster(
                0,
                0,
                tile_job_info.tile_size,
                tile_job_info.tile_size,
                mask,
            )
            tmp_ds.GetRasterBand(2).SetColorInterpretation(gdal.GCI_AlphaBand)
            dsquerytile = tmp_ds
        elif dsquerytile.RasterCount == tilebands - 1:
            # assume that the alpha band is missing and add it
            tmp_ds = mem_driver.CreateCopy("", dsquerytile, 0)
            tmp_ds.AddBand()
            mask = bytearray(
                [255] * (tile_job_info.tile_size * tile_job_info.tile_size)
            )
            tmp_ds.WriteRaster(
                0,
                0,
                tile_job_info.tile_size,
                tile_job_info.tile_size,
                mask,
                band_list=[tilebands],
            )
            dsquerytile = tmp_ds
        elif dsquerytile.RasterCount != tilebands:
            raise Exception(
                "Unexpected number of bands in base tile. Got %d, expected %d"
                % (dsquerytile.RasterCount, tilebands)
            )

        base_data = dsquerytile.ReadRaster(
            0, 0, tile_job_info.tile_size, tile_job_info.tile_size
        )

        dsquery.WriteRaster(
            tileposx,
            tileposy,
            tile_job_info.tile_size,
            tile_job_info.tile_size,
            base_data,
            band_list=list(range(1, tilebands + 1)),
        )

        usable_base_tiles.append(base_tile)

    if not usable_base_tiles:
        return

    scale_query_to_tile(dsquery, dstile, options, tilefilename=tilefilename)
    # Write a copy of tile to png/jpg
    if options.resampling != "antialias":
        # Write a copy of tile to png/jpg
        out_driver.CreateCopy(
            tilefilename,
            dstile
            if tile_job_info.tile_driver != "JPEG"
            else remove_alpha_band(dstile),
            strict=0,
            options=_get_creation_options(options),
        )
        # Remove useless side car file
        aux_xml = tilefilename + ".aux.xml"
        if gdal.VSIStatL(aux_xml) is not None:
            gdal.Unlink(aux_xml)

    if options.verbose:
        logger.debug(
            f"\tbuild from zoom {base_tz}, tiles: %s"
            % ",".join(["(%d, %d)" % (t[0], t[1]) for t in base_tiles])
        )

# 预处理，合并快视图基础瓦片 TODO: ?
def group_overview_base_tiles(
    base_tz: int, output_folder: str, tile_job_info: "TileJobInfo"
) -> List[List[Tuple[int, int]]]:
    """Group base tiles that belong to the same overview tile"""

    overview_to_bases = {}
    tminx, tminy, tmaxx, tmaxy = tile_job_info.tminmax[base_tz]
    for ty in range(tmaxy, tminy - 1, -1):
        overview_ty = ty >> 1
        for tx in range(tminx, tmaxx + 1):
            overview_tx = tx >> 1
            base_tile = (tx, ty)
            overview_tile = (overview_tx, overview_ty)

            if overview_tile not in overview_to_bases:
                overview_to_bases[overview_tile] = []

            overview_to_bases[overview_tile].append(base_tile)

    # Create directories for the tiles
    overview_tz = base_tz - 1
    for tx in range(tminx, tmaxx + 1):
        overview_tx = tx >> 1
        tiledirname = os.path.join(output_folder, str(overview_tz), str(overview_tx))
        makedirs(tiledirname)

    return list(overview_to_bases.values())

# 预处理，计算非顶层瓦片数量
def count_overview_tiles(tile_job_info: "TileJobInfo") -> int:
    tile_number = 0
    for tz in range(tile_job_info.tmaxz - 1, tile_job_info.tminz - 1, -1):
        tminx, tminy, tmaxx, tmaxy = tile_job_info.tminmax[tz]
        tile_number += (1 + abs(tmaxx - tminx)) * (1 + abs(tmaxy - tminy))

    return tile_number

# 预处理，初始化参数
def optparse_init() -> optparse.OptionParser:
    """Prepare the option parser for input (argv)"""

    usage = "Usage: %prog [options] input_file [output]"
    p = optparse.OptionParser(usage, version="%prog " + __version__)

    profile_list = get_profile_list()

    # 切片坐标系
    p.add_option(
        "-p",
        "--profile",
        dest="profile",
        type="choice",
        choices=profile_list,
        help=(
            "Tile cutting profile (%s) - default 'mercator' "
            "(Google Maps compatible)" % ",".join(profile_list)
        ),
    )

    # 重采样方法
    p.add_option(
        "-r",
        "--resampling",
        dest="resampling",
        type="choice",
        choices=resampling_list,
        help="Resampling method (%s) - default 'average'" % ",".join(resampling_list),
    )

    # 输入数据坐标系
    p.add_option(
        "-s",
        "--s_srs",
        dest="s_srs",
        metavar="SRS",
        help="The spatial reference system used for the source input data",
    )

    # 切片等级
    p.add_option(
        "-z",
        "--zoom",
        dest="zoom",
        help="Zoom levels to render (format:'2-5', '10-' or '10').",
    )

    # 恢复模式，断点续切
    p.add_option(
        "-e",
        "--resume",
        dest="resume",
        action="store_true",
        help="Resume mode. Generate only missing files.",
    )

    # 数据集中被视为透明的的nodata值
    p.add_option(
        "-a",
        "--srcnodata",
        dest="srcnodata",
        metavar="NODATA",
        help="Value in the input dataset considered as transparent",
    )

    # 当切片坐标系使用地理坐标系，需要指定分辨率
    p.add_option(
        "-d",
        "--tmscompatible",
        dest="tmscompatible",
        action="store_true",
        help=(
            "When using the geodetic profile, specifies the base resolution "
            "as 0.703125 or 2 tiles at zoom level 0."
        ),
    )

    # 切片为XYZ
    p.add_option(
        "--xyz",
        action="store_true",
        dest="xyz",
        help="Use XYZ tile numbering (OSM Slippy Map tiles) instead of TMS",
    )

    # 打印日志
    p.add_option(
        "-v",
        "--verbose",
        action="store_true",
        dest="verbose",
        help="Print status messages to stdout",
    )

    # 排除瓦片集中的透明瓦片
    p.add_option(
        "-x",
        "--exclude",
        action="store_true",
        dest="exclude_transparent",
        help="Exclude transparent tiles from result tileset",
    )

    # 关闭日志
    p.add_option(
        "-q",
        "--quiet",
        action="store_true",
        dest="quiet",
        help="Disable messages and status to stdout",
    )

    # 切片进程数
    p.add_option(
        "--processes",
        dest="nb_processes",
        type="int",
        help="Number of processes to use for tiling",
    )

    # 启用 mpiexec 时，用户应将 GDAL_CACHEMAX 合理设置为每个进程的大小
    p.add_option(
        "--mpi",
        action="store_true",
        dest="mpi",
        help="Assume launched by mpiexec and ignore --processes. "
        "User should set GDAL_CACHEMAX to size per process.",
    )

    # 切片大小(像素)
    p.add_option(
        "--tilesize",
        dest="tilesize",
        metavar="PIXELS",
        default=256,
        type="int",
        help="Width and height in pixel of a tile",
    )

    # 切片格式
    p.add_option(
        "--tiledriver",
        dest="tiledriver",
        choices=["PNG", "WEBP", "JPEG"],
        default="PNG",
        type="choice",
        help="which tile driver to use for the tiles",
    )

    # 采样方式使用平均重采样时，在重采样期间必须忽略作为贡献源像素的值元组
    p.add_option(
        "--excluded-values",
        dest="excluded_values",
        type=str,
        help="Tuples of values (e.g. <R>,<G>,<B> or (<R1>,<G1>,<B1>),(<R2>,<G2>,<B2>)) that must be ignored as contributing source pixels during resampling. Only taken into account for average resampling",
    )

    # 当设置 --excluded-values 参数时，设置阈值
    p.add_option(
        "--excluded-values-pct-threshold",
        dest="excluded_values_pct_threshold",
        type=float,
        default=50,
        help="Minimum percentage of source pixels that must be set at one of the --excluded-values to cause the excluded value, that is in majority among source pixels, to be used as the target pixel value. Default value is 50 (%)",
    )

    # 采样方式使用平均重采样时，要使目标像素值透明的阈值
    p.add_option(
        "--nodata-values-pct-threshold",
        dest="nodata_values_pct_threshold",
        type=float,
        default=100,
        help="Minimum percentage of source pixels that must be at nodata (or alpha=0 or any other way to express transparent pixel) to cause the target pixel value to be transparent. Default value is 100 (%). Only taken into account for average resampling",
    )

    p.set_defaults(
        verbose=True,
        profile="mercator",
        url="",
        webviewer="all",
        copyright="",
        resampling="average",
        resume=True,
        processes=1,
    )

    return p

# 预处理，初始化参数
def process_args(argv: List[str], called_from_main=False) -> Tuple[str, str, Options]:
    parser = optparse_init()
    options, args = parser.parse_args(args=argv)

    # Args should be either an input file OR an input file and an output folder
    if not args:
        exit_with_error(
            "You need to specify at least an input file as argument to the script"
        )
    if len(args) > 2:
        exit_with_error(
            "Processing of several input files is not supported.",
            "Please first use a tool like gdal_vrtmerge.py or gdal_merge.py on the "
            "files: gdal_vrtmerge.py -o merged.vrt %s" % " ".join(args),
        )

    input_file = args[0]
    try:
        input_file_exists = gdal.Open(input_file) is not None
    except Exception:
        input_file_exists = False
    if not input_file_exists:
        exit_with_error(
            "The provided input file %s does not exist or is not a recognized GDAL dataset"
            % input_file
        )

    if len(args) == 2:
        output_folder = args[1]
    else:
        # Directory with input filename without extension in actual directory
        output_folder = os.path.splitext(os.path.basename(input_file))[0]

    if options.webviewer == "mapml":
        options.xyz = True
        if options.profile == "geodetic":
            options.tmscompatible = True

    if called_from_main:
        if options.verbose:
            logging.basicConfig(level=logging.DEBUG, format="%(message)s")
        elif not options.quiet:
            logging.basicConfig(level=logging.INFO, format="%(message)s")

    options = options_post_processing(options, input_file, output_folder)

    return input_file, output_folder, options

# 预处理，输入参数
def options_post_processing(
    options: Options, input_file: str, output_folder: str
) -> Options:
    # User specified zoom levels
    tminz = None
    tmaxz = None
    if hasattr(options, "zoom") and options.zoom and isinstance(options.zoom, str):
        minmax = options.zoom.split("-", 1)
        zoom_min = minmax[0]
        tminz = int(zoom_min)

        if len(minmax) == 2:
            # Min-max zoom value
            zoom_max = minmax[1]
            if zoom_max:
                # User-specified (non-automatically calculated)
                tmaxz = int(zoom_max)
                if tmaxz < tminz:
                    raise Exception(
                        "max zoom (%d) less than min zoom (%d)" % (tmaxz, tminz)
                    )
        else:
            # Single zoom value (min = max)
            tmaxz = tminz
    options.zoom = [tminz, tmaxz]

    if options.url and not options.url.endswith("/"):
        options.url += "/"
    if options.url:
        out_path = output_folder
        if out_path.endswith("/"):
            out_path = out_path[:-1]
        options.url += os.path.basename(out_path) + "/"

    # Supported options
    if options.resampling == "antialias" and not numpy_available:
        exit_with_error(
            "'antialias' resampling algorithm is not available.",
            "Install PIL (Python Imaging Library) and numpy.",
        )

    if options.tiledriver == "WEBP":
        if gdal.GetDriverByName(options.tiledriver) is None:
            exit_with_error("WEBP driver is not available")

        if not options.webp_lossless:
            if options.webp_quality <= 0 or options.webp_quality > 100:
                exit_with_error("webp_quality should be in the range [1-100]")
            options.webp_quality = int(options.webp_quality)
    elif options.tiledriver == "JPEG":
        if gdal.GetDriverByName(options.tiledriver) is None:
            exit_with_error("JPEG driver is not available")

        if options.jpeg_quality <= 0 or options.jpeg_quality > 100:
            exit_with_error("jpeg_quality should be in the range [1-100]")
        options.jpeg_quality = int(options.jpeg_quality)

    # Output the results
    if options.verbose:
        logger.debug("Options: %s" % str(options))
        logger.debug(f"Input: {input_file}")
        logger.debug(f"Output: {output_folder}")
        logger.debug("Cache: %d MB" % (gdal.GetCacheMax() / 1024 / 1024))

    return options


class TileJobInfo:
    """
    Plain object to hold tile job configuration for a dataset
    """

    src_file = ""
    nb_data_bands = 0
    output_file_path = ""
    tile_extension = ""
    tile_size = 0
    tile_driver = None
    tminmax = []
    tminz = 0
    tmaxz = 0
    in_srs_wkt = 0
    out_geo_trans = []
    ominy = 0
    is_epsg_4326 = False
    options = None
    exclude_transparent = False

    def __init__(self, **kwargs):
        for key in kwargs:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])

    def __unicode__(self):
        return "TileJobInfo %s\n" % (self.src_file)

    def __str__(self):
        return "TileJobInfo %s\n" % (self.src_file)

    def __repr__(self):
        return "TileJobInfo %s\n" % (self.src_file)


class Gdal2TilesError(Exception):
    pass


class GDAL2Tiles:
    def __init__(self, input_file: str, output_folder: str, options: Options) -> None:
        """Constructor function - initialization"""
        self.out_drv = None
        self.mem_drv = None
        self.warped_input_dataset = None
        self.out_srs = None
        self.nativezoom = None
        self.tminmax = None
        self.tsize = None
        self.mercator = None
        self.geodetic = None
        self.dataBandsCount = None
        self.out_gt = None
        self.tileswne = None
        self.swne = None
        self.ominx = None
        self.omaxx = None
        self.omaxy = None
        self.ominy = None

        self.input_file = None
        self.output_folder = None

        self.isepsg4326 = None
        self.in_srs = None
        self.in_srs_wkt = None

        # Tile format
        self.tile_size = 256
        if options.tilesize:
            self.tile_size = options.tilesize

        self.tiledriver = options.tiledriver
        if options.tiledriver == "PNG":
            self.tileext = "png"
        elif options.tiledriver == "WEBP":
            self.tileext = "webp"
        else:
            self.tileext = "jpg"
        if options.mpi:
            makedirs(output_folder)
            self.tmp_dir = tempfile.mkdtemp(dir=output_folder)
        else:
            self.tmp_dir = tempfile.mkdtemp()
        self.tmp_vrt_filename = os.path.join(self.tmp_dir, str(uuid4()) + ".vrt")

        # Should we read bigger window of the input raster and scale it down?
        # Note: Modified later by open_input()
        # Not for 'near' resampling
        # Not for Wavelet based drivers (JPEG2000, ECW, MrSID)
        # Not for 'raster' profile
        self.scaledquery = True
        # How big should be query window be for scaling down
        # Later on reset according the chosen resampling algorithm
        self.querysize = 4 * self.tile_size

        # Should we use Read on the input file for generating overview tiles?
        # Note: Modified later by open_input()
        # Otherwise the overview tiles are generated from existing underlying tiles
        self.overviewquery = False

        self.input_file = input_file
        self.output_folder = output_folder
        self.options = options

        if self.options.resampling == "near":
            self.querysize = self.tile_size

        elif self.options.resampling == "bilinear":
            self.querysize = self.tile_size * 2

        self.tminz, self.tmaxz = self.options.zoom


    # 打开影像文件
    def open_input(self) -> None:
        """Initialization of the input raster, reprojection if necessary"""
        gdal.AllRegister()

        self.out_drv = gdal.GetDriverByName(self.tiledriver)
        self.mem_drv = gdal.GetDriverByName("MEM")

        if not self.out_drv:
            raise Exception(
                "The '%s' driver was not found, is it available in this GDAL build?"
                % self.tiledriver
            )
        if not self.mem_drv:
            raise Exception(
                "The 'MEM' driver was not found, is it available in this GDAL build?"
            )

        # 打开数据集
        if self.input_file:
            input_dataset: gdal.Dataset = gdal.Open(self.input_file, gdal.GA_ReadOnly)
        else:
            raise Exception("No input file was specified")

        if self.options.verbose:
            logger.debug(
                "Input file: (%dP x %dL - %d bands)"
                % (
                    input_dataset.RasterXSize,
                    input_dataset.RasterYSize,
                    input_dataset.RasterCount,
                ),
            )

        if not input_dataset:
            # Note: GDAL prints the ERROR message too
            exit_with_error(
                "It is not possible to open the input file '%s'." % self.input_file
            )

        # Read metadata from the input file
        if input_dataset.RasterCount == 0:
            exit_with_error("Input file '%s' has no raster band" % self.input_file)

        if input_dataset.GetRasterBand(1).GetRasterColorTable():
            exit_with_error(
                "Please convert this file to RGB/RGBA and run gdal2tiles on the result.",
                "From paletted file you can create RGBA file (temp.vrt) by:\n"
                "gdal_translate -of vrt -expand rgba %s temp.vrt\n"
                "then run:\n"
                "gdal2tiles temp.vrt" % self.input_file,
            )

        if input_dataset.GetRasterBand(1).DataType != gdal.GDT_Byte:
            exit_with_error(
                "Please convert this file to 8-bit and run gdal2tiles on the result.",
                "To scale pixel values you can use:\n"
                "gdal_translate -of VRT -ot Byte -scale %s temp.vrt\n"
                "then run:\n"
                "gdal2tiles temp.vrt" % self.input_file,
            )

        in_nodata = setup_no_data_values(input_dataset, self.options)

        if self.options.verbose:
            logger.debug(
                "Preprocessed file:(%dP x %dL - %d bands)"
                % (
                    input_dataset.RasterXSize,
                    input_dataset.RasterYSize,
                    input_dataset.RasterCount,
                ),
            )

        self.in_srs, self.in_srs_wkt = setup_input_srs(input_dataset, self.options)

        self.out_srs = setup_output_srs(self.in_srs, self.options)

        # If input and output reference systems are different, we reproject the input dataset into
        # the output reference system for easier manipulation

        self.warped_input_dataset = None

        if self.options.profile != "raster":

            if not self.in_srs:
                exit_with_error(
                    "Input file has unknown SRS.",
                    "Use --s_srs EPSG:xyz (or similar) to provide source reference system.",
                )

            if not has_georeference(input_dataset):
                exit_with_error(
                    "There is no georeference - neither affine transformation (worldfile) "
                    "nor GCPs. You can generate only 'raster' profile tiles.",
                    "Either gdal2tiles with parameter -p 'raster' or use another GIS "
                    "software for georeference e.g. gdal_transform -gcp / -a_ullr / -a_srs",
                )

            if (self.in_srs.ExportToProj4() != self.out_srs.ExportToProj4()) or (
                input_dataset.GetGCPCount() != 0
            ):
                self.warped_input_dataset = reproject_dataset(
                    input_dataset, self.in_srs, self.out_srs
                )

                if in_nodata:
                    self.warped_input_dataset = update_no_data_values(
                        self.warped_input_dataset, in_nodata, options=self.options
                    )
                else:
                    self.warped_input_dataset = update_alpha_value_for_non_alpha_inputs(
                        self.warped_input_dataset, options=self.options
                    )

            if self.warped_input_dataset and self.options.verbose:
                logger.debug(
                    "Projected file: tiles.vrt (%dP x %dL - %d bands)"
                    % (
                        self.warped_input_dataset.RasterXSize,
                        self.warped_input_dataset.RasterYSize,
                        self.warped_input_dataset.RasterCount,
                    ),
                )

        if not self.warped_input_dataset:
            self.warped_input_dataset = input_dataset

        # VRT:gdal基于XML格式的虚拟文件格式
        gdal.GetDriverByName("VRT").CreateCopy(
            self.tmp_vrt_filename, self.warped_input_dataset
        )

        self.dataBandsCount = nb_data_bands(self.warped_input_dataset)

        # KML test
        self.isepsg4326 = False
        srs4326 = osr.SpatialReference()
        srs4326.ImportFromEPSG(4326)
        srs4326.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if self.out_srs and srs4326.ExportToProj4() == self.out_srs.ExportToProj4():
            self.isepsg4326 = True

        # Read the georeference
        self.out_gt = self.warped_input_dataset.GetGeoTransform()

        # Test the size of the pixel

        # Report error in case rotation/skew is in geotransform (possible only in 'raster' profile)
        if (self.out_gt[2], self.out_gt[4]) != (0, 0):
            exit_with_error(
                "Georeference of the raster contains rotation or skew. "
                "Such raster is not supported. Please use gdalwarp first."
            )

        # Here we expect: pixel is square, no rotation on the raster

        # Output Bounds - coordinates in the output SRS
        self.ominx = self.out_gt[0]
        self.omaxx = (
            self.out_gt[0] + self.warped_input_dataset.RasterXSize * self.out_gt[1]
        )
        self.omaxy = self.out_gt[3]
        self.ominy = (
            self.out_gt[3] - self.warped_input_dataset.RasterYSize * self.out_gt[1]
        )
        # Note: maybe round(x, 14) to avoid the gdal_translate behavior, when 0 becomes -1e-15

        if self.options.verbose:
            logger.debug(
                "Bounds (output srs): %f, %f, %f, %f"
                % (round(self.ominx, 13), self.ominy, self.omaxx, self.omaxy)
            )

        # 根据不同的层级，计算瓦片的范围
        if self.options.profile == "mercator":

            self.mercator = GlobalMercator(tile_size=self.tile_size)

            # Function which generates SWNE in LatLong for given tile
            self.tileswne = self.mercator.TileLatLonBounds

            # Generate table with min max tile coordinates for all zoomlevels
            self.tminmax = list(range(0, MAXZOOMLEVEL))
            for tz in range(0, MAXZOOMLEVEL):
                tminx, tminy = self.mercator.MetersToTile(self.ominx, self.ominy, tz)
                tmaxx, tmaxy = self.mercator.MetersToTile(self.omaxx, self.omaxy, tz)
                # crop tiles extending world limits (+-180,+-90)
                tminx, tminy = max(0, tminx), max(0, tminy)
                tmaxx, tmaxy = min(2**tz - 1, tmaxx), min(2**tz - 1, tmaxy)
                self.tminmax[tz] = (tminx, tminy, tmaxx, tmaxy)

            # TODO: Maps crossing 180E (Alaska?)

            # Get the minimal zoom level (map covers area equivalent to one tile)
            if self.tminz is None:
                self.tminz = self.mercator.ZoomForPixelSize(
                    self.out_gt[1]
                    * max(
                        self.warped_input_dataset.RasterXSize,
                        self.warped_input_dataset.RasterYSize,
                    )
                    / float(self.tile_size)
                )

            # Get the maximal zoom level
            # (closest possible zoom level up on the resolution of raster)
            if self.tmaxz is None:
                self.tmaxz = self.mercator.ZoomForPixelSize(self.out_gt[1])
                self.tmaxz = max(self.tminz, self.tmaxz)

            self.tminz = min(self.tminz, self.tmaxz)

            if self.options.verbose:
                logger.debug(
                    "Bounds (latlong): %s, %s",
                    str(self.mercator.MetersToLatLon(self.ominx, self.ominy)),
                    str(self.mercator.MetersToLatLon(self.omaxx, self.omaxy)),
                )
                logger.debug("MinZoomLevel: %d" % self.tminz)
                logger.debug(
                    "MaxZoomLevel: %d (%f)"
                    % (self.tmaxz, self.mercator.Resolution(self.tmaxz))
                )

        elif self.options.profile == "geodetic":

            self.geodetic = GlobalGeodetic(
                self.options.tmscompatible, tile_size=self.tile_size
            )

            # Function which generates SWNE in LatLong for given tile
            self.tileswne = self.geodetic.TileLatLonBounds

            # Generate table with min max tile coordinates for all zoomlevels
            self.tminmax = list(range(0, MAXZOOMLEVEL))
            for tz in range(0, MAXZOOMLEVEL):
                tminx, tminy = self.geodetic.LonLatToTile(self.ominx, self.ominy, tz)
                tmaxx, tmaxy = self.geodetic.LonLatToTile(self.omaxx, self.omaxy, tz)
                # crop tiles extending world limits (+-180,+-90)
                tminx, tminy = max(0, tminx), max(0, tminy)
                tmaxx, tmaxy = min(2 ** (tz + 1) - 1, tmaxx), min(2**tz - 1, tmaxy)
                self.tminmax[tz] = (tminx, tminy, tmaxx, tmaxy)

            # TODO: Maps crossing 180E (Alaska?)

            # Get the maximal zoom level
            # (closest possible zoom level up on the resolution of raster)
            if self.tminz is None:
                self.tminz = self.geodetic.ZoomForPixelSize(
                    self.out_gt[1]
                    * max(
                        self.warped_input_dataset.RasterXSize,
                        self.warped_input_dataset.RasterYSize,
                    )
                    / float(self.tile_size)
                )

            # Get the maximal zoom level
            # (closest possible zoom level up on the resolution of raster)
            if self.tmaxz is None:
                self.tmaxz = self.geodetic.ZoomForPixelSize(self.out_gt[1])
                self.tmaxz = max(self.tminz, self.tmaxz)

            self.tminz = min(self.tminz, self.tmaxz)

            if self.options.verbose:
                logger.debug(
                    "Bounds (latlong): %f, %f, %f, %f"
                    % (self.ominx, self.ominy, self.omaxx, self.omaxy)
                )

        elif self.options.profile == "raster":

            def log2(x):
                return math.log10(x) / math.log10(2)

            self.nativezoom = max(
                0,
                int(
                    max(
                        math.ceil(
                            log2(
                                self.warped_input_dataset.RasterXSize
                                / float(self.tile_size)
                            )
                        ),
                        math.ceil(
                            log2(
                                self.warped_input_dataset.RasterYSize
                                / float(self.tile_size)
                            )
                        ),
                    )
                ),
            )

            if self.options.verbose:
                logger.debug("Native zoom of the raster: %d" % self.nativezoom)

            # Get the minimal zoom level (whole raster in one tile)
            if self.tminz is None:
                self.tminz = 0

            # Get the maximal zoom level (native resolution of the raster)
            if self.tmaxz is None:
                self.tmaxz = self.nativezoom
                self.tmaxz = max(self.tminz, self.tmaxz)
            elif self.tmaxz > self.nativezoom:
                # If the user requests at a higher precision than the native
                # one, generate an oversample temporary VRT file, and tile from
                # it
                oversample_factor = 1 << (self.tmaxz - self.nativezoom)
                if self.options.resampling in ("average", "antialias"):
                    resampleAlg = "average"
                elif self.options.resampling in (
                    "near",
                    "bilinear",
                    "cubic",
                    "cubicspline",
                    "lanczos",
                    "mode",
                ):
                    resampleAlg = self.options.resampling
                else:
                    resampleAlg = "bilinear"  # fallback
                gdal.Translate(
                    self.tmp_vrt_filename,
                    input_dataset,
                    width=self.warped_input_dataset.RasterXSize * oversample_factor,
                    height=self.warped_input_dataset.RasterYSize * oversample_factor,
                    resampleAlg=resampleAlg,
                )
                self.warped_input_dataset = gdal.Open(self.tmp_vrt_filename)
                self.out_gt = self.warped_input_dataset.GetGeoTransform()
                self.nativezoom = self.tmaxz

            # Generate table with min max tile coordinates for all zoomlevels
            self.tminmax = list(range(0, self.tmaxz + 1))
            self.tsize = list(range(0, self.tmaxz + 1))
            for tz in range(0, self.tmaxz + 1):
                tsize = 2.0 ** (self.nativezoom - tz) * self.tile_size
                tminx, tminy = 0, 0
                tmaxx = (
                    int(math.ceil(self.warped_input_dataset.RasterXSize / tsize)) - 1
                )
                tmaxy = (
                    int(math.ceil(self.warped_input_dataset.RasterYSize / tsize)) - 1
                )
                self.tsize[tz] = math.ceil(tsize)
                self.tminmax[tz] = (tminx, tminy, tmaxx, tmaxy)

            # Function which generates SWNE in LatLong for given tile
            self.tileswne = lambda x, y, z: (0, 0, 0, 0)  # noqa

        else:

            tms = tmsMap[self.options.profile]

            # Function which generates SWNE in LatLong for given tile
            self.tileswne = None  # not implemented

            # Generate table with min max tile coordinates for all zoomlevels
            self.tminmax = list(range(0, tms.level_count + 1))
            for tz in range(0, tms.level_count + 1):
                tminx, tminy = tms.GeorefCoordToTileCoord(
                    self.ominx, self.ominy, tz, self.tile_size
                )
                tmaxx, tmaxy = tms.GeorefCoordToTileCoord(
                    self.omaxx, self.omaxy, tz, self.tile_size
                )
                tminx, tminy = max(0, tminx), max(0, tminy)
                tmaxx, tmaxy = min(tms.matrix_width * 2**tz - 1, tmaxx), min(
                    tms.matrix_height * 2**tz - 1, tmaxy
                )
                self.tminmax[tz] = (tminx, tminy, tmaxx, tmaxy)

            # Get the minimal zoom level (map covers area equivalent to one tile)
            if self.tminz is None:
                self.tminz = tms.ZoomForPixelSize(
                    self.out_gt[1]
                    * max(
                        self.warped_input_dataset.RasterXSize,
                        self.warped_input_dataset.RasterYSize,
                    )
                    / float(self.tile_size),
                    self.tile_size,
                )

            # Get the maximal zoom level
            # (closest possible zoom level up on the resolution of raster)
            if self.tmaxz is None:
                self.tmaxz = tms.ZoomForPixelSize(self.out_gt[1], self.tile_size)
                self.tmaxz = max(self.tminz, self.tmaxz)

            self.tminz = min(self.tminz, self.tmaxz)

            if self.options.verbose:
                logger.debug(
                    "Bounds (georef): %f, %f, %f, %f"
                    % (self.ominx, self.ominy, self.omaxx, self.omaxy)
                )
                logger.debug("MinZoomLevel: %d" % self.tminz)
                logger.debug("MaxZoomLevel: %d" % self.tmaxz)
    # 构建tms元数据xml
    def generate_metadata(self) -> None:

        makedirs(self.output_folder)

        if self.options.profile == "mercator":

            south, west = self.mercator.MetersToLatLon(self.ominx, self.ominy)
            north, east = self.mercator.MetersToLatLon(self.omaxx, self.omaxy)
            south, west = max(-85.05112878, south), max(-180.0, west)
            north, east = min(85.05112878, north), min(180.0, east)
            self.swne = (south, west, north, east)

        elif self.options.profile == "geodetic":

            west, south = self.ominx, self.ominy
            east, north = self.omaxx, self.omaxy
            south, west = max(-90.0, south), max(-180.0, west)
            north, east = min(90.0, north), min(180.0, east)
            self.swne = (south, west, north, east)

        elif self.options.profile == "raster":

            west, south = self.ominx, self.ominy
            east, north = self.omaxx, self.omaxy

            self.swne = (south, west, north, east)

        else:
            self.swne = None

        # Generate tilemapresource.xml.
        if (
            not self.options.xyz
            and self.swne is not None
            and (
                not self.options.resume
                or not isfile(os.path.join(self.output_folder, "tilemapresource.xml"))
            )
        ):
            with my_open(
                os.path.join(self.output_folder, "tilemapresource.xml"), "wb"
            ) as f:
                f.write(self.generate_tilemapresource().encode("utf-8"))

        # Generate mapml file
        if (
            self.options.webviewer in ("all", "mapml")
            and self.options.xyz
            and self.options.profile != "raster"
            and (self.options.profile != "geodetic" or self.options.tmscompatible)
            and (
                not self.options.resume
                or not isfile(os.path.join(self.output_folder, "mapml.mapml"))
            )
        ):
            with my_open(os.path.join(self.output_folder, "mapml.mapml"), "wb") as f:
                f.write(self.generate_mapml().encode("utf-8"))

    # 直接从输入的栅格数据中，切出在快视图中的最底部层级瓦片(即最高层级)
    def generate_base_tiles(self) -> Tuple[TileJobInfo, List[TileDetail]]:

        if not self.options.quiet:
            logger.info("Generating Base Tiles:")

        if self.options.verbose:
            logger.debug("")
            logger.debug("Tiles generated from the max zoom level:")
            logger.debug("----------------------------------------")
            logger.debug("")

        # Set the bounds
        tminx, tminy, tmaxx, tmaxy = self.tminmax[self.tmaxz]

        ds = self.warped_input_dataset
        tilebands = self.dataBandsCount + 1
        querysize = self.querysize

        if self.options.verbose:
            logger.debug("dataBandsCount: %d" % self.dataBandsCount)
            logger.debug("tilebands: %d" % tilebands)

        tcount = (1 + abs(tmaxx - tminx)) * (1 + abs(tmaxy - tminy))
        ti = 0

        tile_details = []

        tz = self.tmaxz

        # Create directories for the tiles
        for tx in range(tminx, tmaxx + 1):
            tiledirname = os.path.join(self.output_folder, str(tz), str(tx))
            makedirs(tiledirname)

        for ty in range(tmaxy, tminy - 1, -1):
            for tx in range(tminx, tmaxx + 1):

                ti += 1
                ytile = GDAL2Tiles.getYTile(ty, tz, self.options)
                tilefilename = os.path.join(
                    self.output_folder,
                    str(tz),
                    str(tx),
                    "%s.%s" % (ytile, self.tileext),
                )
                if self.options.verbose:
                    logger.debug("%d / %d, %s" % (ti, tcount, tilefilename))

                if self.options.resume and isfile(tilefilename):
                    if self.options.verbose:
                        logger.debug("Tile generation skipped because of --resume")
                    continue

                if self.options.profile == "mercator":
                    # Tile bounds in EPSG:3857
                    b = self.mercator.TileBounds(tx, ty, tz)
                elif self.options.profile == "geodetic":
                    b = self.geodetic.TileBounds(tx, ty, tz)
                elif self.options.profile != "raster":
                    b = tmsMap[self.options.profile].TileBounds(
                        tx, ty, tz, self.tile_size
                    )

                # Don't scale up by nearest neighbour, better change the querysize
                # to the native resolution (and return smaller query tile) for scaling

                if self.options.profile != "raster":
                    rb, wb = self.geo_query(ds, b[0], b[3], b[2], b[1])

                    # Pixel size in the raster covering query geo extent
                    nativesize = wb[0] + wb[2]
                    if self.options.verbose:
                        logger.debug(
                            f"\tNative Extent (querysize {nativesize}): {rb}, {wb}"
                        )

                    # Tile bounds in raster coordinates for ReadRaster query
                    rb, wb = self.geo_query(
                        ds, b[0], b[3], b[2], b[1], querysize=querysize
                    )

                    rx, ry, rxsize, rysize = rb
                    wx, wy, wxsize, wysize = wb

                else:  # 'raster' profile:

                    tsize = int(
                        self.tsize[tz]
                    )  # tile_size in raster coordinates for actual zoom
                    xsize = (
                        self.warped_input_dataset.RasterXSize
                    )  # size of the raster in pixels
                    ysize = self.warped_input_dataset.RasterYSize
                    querysize = self.tile_size

                    rx = tx * tsize
                    rxsize = 0
                    if tx == tmaxx:
                        rxsize = xsize % tsize
                    if rxsize == 0:
                        rxsize = tsize

                    ry = ty * tsize
                    rysize = 0
                    if ty == tmaxy:
                        rysize = ysize % tsize
                    if rysize == 0:
                        rysize = tsize

                    wx, wy = 0, 0
                    wxsize = int(rxsize / float(tsize) * self.tile_size)
                    wysize = int(rysize / float(tsize) * self.tile_size)

                    if not self.options.xyz:
                        ry = ysize - (ty * tsize) - rysize
                        if wysize != self.tile_size:
                            wy = self.tile_size - wysize

                if rxsize == 0 or rysize == 0 or wxsize == 0 or wysize == 0:
                    if self.options.verbose:
                        logger.debug("\tExcluding tile with no pixel coverage")
                    continue

                # Read the source raster if anything is going inside the tile as per the computed
                # geo_query
                # todo: ?
                tile_details.append(
                    TileDetail(
                        tx=tx,
                        ty=ytile,
                        tz=tz,
                        rx=rx,
                        ry=ry,
                        rxsize=rxsize,
                        rysize=rysize,
                        wx=wx,
                        wy=wy,
                        wxsize=wxsize,
                        wysize=wysize,
                        querysize=querysize,
                    )
                )

        conf = TileJobInfo(
            src_file=self.tmp_vrt_filename,
            nb_data_bands=self.dataBandsCount,
            output_file_path=self.output_folder,
            tile_extension=self.tileext,
            tile_driver=self.tiledriver,
            tile_size=self.tile_size,
            tminmax=self.tminmax,
            tminz=self.tminz,
            tmaxz=self.tmaxz,
            in_srs_wkt=self.in_srs_wkt,
            out_geo_trans=self.out_gt,
            ominy=self.ominy,
            is_epsg_4326=self.isepsg4326,
            options=self.options,
            exclude_transparent=self.options.exclude_transparent,
        )

        return conf, tile_details

    def geo_query(self, ds, ulx, uly, lrx, lry, querysize=0):
        """
        For given dataset and query in cartographic coordinates returns parameters for ReadRaster()
        in raster coordinates and x/y shifts (for border tiles). If the querysize is not given, the
        extent is returned in the native resolution of dataset ds.

        raises Gdal2TilesError if the dataset does not contain anything inside this geo_query
        """
        geotran = ds.GetGeoTransform()
        rx = int((ulx - geotran[0]) / geotran[1] + 0.001)
        ry = int((uly - geotran[3]) / geotran[5] + 0.001)
        rxsize = max(1, int((lrx - ulx) / geotran[1] + 0.5))
        rysize = max(1, int((lry - uly) / geotran[5] + 0.5))

        if not querysize:
            wxsize, wysize = rxsize, rysize
        else:
            wxsize, wysize = querysize, querysize

        # Coordinates should not go out of the bounds of the raster
        wx = 0
        if rx < 0:
            rxshift = abs(rx)
            wx = int(wxsize * (float(rxshift) / rxsize))
            wxsize = wxsize - wx
            rxsize = rxsize - int(rxsize * (float(rxshift) / rxsize))
            rx = 0
        if rx + rxsize > ds.RasterXSize:
            wxsize = int(wxsize * (float(ds.RasterXSize - rx) / rxsize))
            rxsize = ds.RasterXSize - rx

        wy = 0
        if ry < 0:
            ryshift = abs(ry)
            wy = int(wysize * (float(ryshift) / rysize))
            wysize = wysize - wy
            rysize = rysize - int(rysize * (float(ryshift) / rysize))
            ry = 0
        if ry + rysize > ds.RasterYSize:
            wysize = int(wysize * (float(ds.RasterYSize - ry) / rysize))
            rysize = ds.RasterYSize - ry

        return (rx, ry, rxsize, rysize), (wx, wy, wxsize, wysize)

    def generate_tilemapresource(self) -> str:
        """
        Template for tilemapresource.xml. Returns filled string. Expected variables:
          title, north, south, east, west, isepsg4326, projection, publishurl,
          zoompixels, tile_size, tileformat, profile
        """

        args = {}
        args["south"], args["west"], args["north"], args["east"] = self.swne
        args["tile_size"] = self.tile_size
        args["tileformat"] = self.tileext
        args["publishurl"] = self.options.url
        args["profile"] = self.options.profile

        if self.options.profile == "mercator":
            args["srs"] = "EPSG:3857"
        elif self.options.profile == "geodetic":
            args["srs"] = "EPSG:4326"
        elif self.options.s_srs:
            args["srs"] = self.options.s_srs
        elif self.out_srs:
            args["srs"] = self.out_srs.ExportToWkt()
        else:
            args["srs"] = ""

        s = (
            """<?xml version="1.0" encoding="utf-8"?>
    <TileMap version="1.0.0" tilemapservice="http://tms.osgeo.org/1.0.0">
      <Title>title</Title>
      <Abstract></Abstract>
      <SRS>%(srs)s</SRS>
      <BoundingBox minx="%(west).14f" miny="%(south).14f" maxx="%(east).14f" maxy="%(north).14f"/>
      <Origin x="%(west).14f" y="%(south).14f"/>
      <TileFormat width="%(tile_size)d" height="%(tile_size)d" mime-type="image/%(tileformat)s" extension="%(tileformat)s"/>
      <TileSets profile="%(profile)s">
"""
            % args
        )  # noqa
        for z in range(self.tminz, self.tmaxz + 1):
            if self.options.profile == "raster":
                s += (
                    """        <TileSet href="%s%d" units-per-pixel="%.14f" order="%d"/>\n"""
                    % (
                        args["publishurl"],
                        z,
                        (2 ** (self.nativezoom - z) * self.out_gt[1]),
                        z,
                    )
                )
            elif self.options.profile == "mercator":
                s += (
                    """        <TileSet href="%s%d" units-per-pixel="%.14f" order="%d"/>\n"""
                    % (args["publishurl"], z, 156543.0339 / 2**z, z)
                )
            elif self.options.profile == "geodetic":
                s += (
                    """        <TileSet href="%s%d" units-per-pixel="%.14f" order="%d"/>\n"""
                    % (args["publishurl"], z, 0.703125 / 2**z, z)
                )
        s += """      </TileSets>
    </TileMap>
    """
        return s

    @staticmethod
    def getYTile(ty, tz, options):
        """
        Calculates the y-tile number based on whether XYZ or TMS (default) system is used
        :param ty: The y-tile number
        :param tz: The z-tile number
        :return: The transformed y-tile number
        """
        if options.xyz and options.profile != "raster":
            if options.profile in ("mercator", "geodetic"):
                return (2**tz - 1) - ty  # Convert from TMS to XYZ numbering system

            tms = tmsMap[options.profile]
            return (
                tms.matrix_height * 2**tz - 1
            ) - ty  # Convert from TMS to XYZ numbering system

        return ty

# 构建切片任务列表
def worker_tile_details(
    input_file: str, output_folder: str, options: Options
) -> Tuple[TileJobInfo, List[TileDetail]]:
    gdal2tiles = GDAL2Tiles(input_file, output_folder, options)
    gdal2tiles.open_input()
    gdal2tiles.generate_metadata()
    tile_job_info, tile_details = gdal2tiles.generate_base_tiles()
    return tile_job_info, tile_details


class ProgressBar:
    def __init__(self, total_items: int, progress_cbk=gdal.TermProgress_nocb) -> None:
        self.total_items = total_items
        self.nb_items_done = 0
        self.progress_cbk = progress_cbk

    def start(self) -> None:
        self.progress_cbk(0, "", None)

    def log_progress(self, nb_items: int = 1) -> None:
        self.nb_items_done += nb_items
        progress = float(self.nb_items_done) / self.total_items
        self.progress_cbk(progress, "", None)

# 单线程切片
def single_threaded_tiling(
    input_file: str, output_folder: str, options: Options
) -> None:
    """
    Keep a single threaded version that stays clear of multiprocessing, for platforms that would not
    support it
    """
    if options.verbose:
        logger.debug("Begin tiles details calc")
    conf, tile_details = worker_tile_details(input_file, output_folder, options)

    if options.verbose:
        logger.debug("Tiles details calc complete.")

    if not options.verbose and not options.quiet:
        base_progress_bar = ProgressBar(len(tile_details))
        base_progress_bar.start()

    # 遍历切片任务列表，串行执行切片任务
    for tile_detail in tile_details:
        create_base_tile(conf, tile_detail)

        if not options.verbose and not options.quiet:
            base_progress_bar.log_progress()

    if getattr(threadLocal, "cached_ds", None):
        del threadLocal.cached_ds

    if not options.quiet:
        count = count_overview_tiles(conf)
        if count:
            logger.info("Generating Overview Tiles:")

            if not options.verbose:
                overview_progress_bar = ProgressBar(count)
                overview_progress_bar.start()

    for base_tz in range(conf.tmaxz, conf.tminz, -1):
        base_tile_groups = group_overview_base_tiles(base_tz, output_folder, conf)
        for base_tiles in base_tile_groups:
            create_overview_tile(base_tz, base_tiles, output_folder, conf, options)
            if not options.verbose and not options.quiet:
                overview_progress_bar.log_progress()

    shutil.rmtree(os.path.dirname(conf.src_file))

# 多线程切片
# @enable_gdal_exceptions
def multi_threaded_tiling(
    input_file: str, output_folder: str, options: Options, pool
) -> None:
    nb_processes = options.nb_processes or 1

    if options.verbose:
        logger.debug("Begin tiles details calc")

    conf, tile_details = worker_tile_details(input_file, output_folder, options)

    if options.verbose:
        logger.debug("Tiles details calc complete.")

    if not options.verbose and not options.quiet:
        base_progress_bar = ProgressBar(len(tile_details))
        base_progress_bar.start()

    # TODO: gbataille - check the confs for which each element is an array... one useless level?
    # TODO: gbataille - assign an ID to each job for print in verbose mode "ReadRaster Extent ..."
    chunksize = max(1, min(128, len(tile_details) // nb_processes))
    for _ in pool.imap_unordered(
        partial(create_base_tile, conf), tile_details, chunksize=chunksize
    ):
        if not options.verbose and not options.quiet:
            base_progress_bar.log_progress()

    if not options.quiet:
        count = count_overview_tiles(conf)
        if count:
            logger.info("Generating Overview Tiles:")

            if not options.verbose:
                overview_progress_bar = ProgressBar(count)
                overview_progress_bar.start()

    for base_tz in range(conf.tmaxz, conf.tminz, -1):
        base_tile_groups = group_overview_base_tiles(base_tz, output_folder, conf)
        chunksize = max(1, min(128, len(base_tile_groups) // nb_processes))
        for _ in pool.imap_unordered(
            partial(
                create_overview_tile,
                base_tz,
                output_folder=output_folder,
                tile_job_info=conf,
                options=options,
            ),
            base_tile_groups,
            chunksize=chunksize,
        ):
            if not options.verbose and not options.quiet:
                overview_progress_bar.log_progress()

    shutil.rmtree(os.path.dirname(conf.src_file))


class DividedCache:
    def __init__(self, nb_processes):
        self.nb_processes = nb_processes

    def __enter__(self):
        self.gdal_cache_max = gdal.GetCacheMax()
        # Make sure that all processes do not consume more than `gdal.GetCacheMax()`
        gdal_cache_max_per_process = max(
            1024 * 1024, math.floor(self.gdal_cache_max / self.nb_processes)
        )
        set_cache_max(gdal_cache_max_per_process)

    def __exit__(self, type, value, tb):
        # Set the maximum cache back to the original value
        set_cache_max(self.gdal_cache_max)


def main(argv: List[str] = sys.argv, called_from_main=False) -> int:
    # TODO: gbataille - use mkdtemp to work in a temp directory
    # TODO: gbataille - debug intermediate tiles.vrt not produced anymore?
    # TODO: gbataille - Refactor generate overview tiles to not depend on self variables

    # For multiprocessing, we need to propagate the configuration options to
    # the environment, so that forked processes can inherit them.
    for i in range(len(argv)):
        if argv[i] == "--config" and i + 2 < len(argv):
            os.environ[argv[i + 1]] = argv[i + 2]

    if "--mpi" in argv:
        from mpi4py import MPI
        from mpi4py.futures import MPICommExecutor

        with MPICommExecutor(MPI.COMM_WORLD, root=0) as pool:
            if pool is None:
                return 0
            # add interface of multiprocessing.Pool to MPICommExecutor
            pool.imap_unordered = partial(pool.map, unordered=True)
            return submain(
                argv, pool, MPI.COMM_WORLD.Get_size(), called_from_main=called_from_main
            )
    else:
        return submain(argv, called_from_main=called_from_main)


# @enable_gdal_exceptions
def submain(argv: List[str], pool=None, pool_size=0, called_from_main=False) -> int:

    argv = gdal.GeneralCmdLineProcessor(argv)
    if argv is None:
        return 0
    input_file, output_folder, options = process_args(
        argv[1:], called_from_main=called_from_main
    )
    if pool_size:
        options.nb_processes = pool_size
    nb_processes = options.nb_processes or 1

    if pool is not None:  # MPI
        multi_threaded_tiling(input_file, output_folder, options, pool)
    elif nb_processes == 1:
        single_threaded_tiling(input_file, output_folder, options)
    else:
        # Trick inspired from https://stackoverflow.com/questions/45720153/python-multiprocessing-error-attributeerror-module-main-has-no-attribute
        # and https://bugs.python.org/issue42949
        import __main__

        if not hasattr(__main__, "__spec__"):
            __main__.__spec__ = None
        from multiprocessing import Pool

        with DividedCache(nb_processes), Pool(processes=nb_processes) as pool:
            multi_threaded_tiling(input_file, output_folder, options, pool)

    return 0


# vim: set tabstop=4 shiftwidth=4 expandtab:

# Running main() must be protected that way due to use of multiprocessing on Windows:
# https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods
if __name__ == "__main__":
    sys.exit(main(sys.argv, called_from_main=True))
