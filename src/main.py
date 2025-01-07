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

import glob
import json
import logging
import optparse
import os
import shutil
import sys
import threading
from functools import partial
from typing import Any, List, Tuple

from osgeo import gdal

from src.cache.cache import DividedCache
from src.common.fileutil import makedirs, isfile
from src.core.gdal2tiles import GDAL2Tiles
from src.core.preprocess import count_overview_tiles
from src.core.tiledetail import TileDetail
from src.core.tilejobinfo import TileJobInfo
from src.helper.processbar import ProgressBar
from src.log.log import exit_with_error

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

logger = logging.getLogger("main")


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


class Gdal2TilesError(Exception):
    pass


# 构建切片任务列表
def worker_tile_details(
        input_file: str, output_folder: str, options: Options
) -> Tuple[TileJobInfo, List[TileDetail]]:
    gdal2tiles = GDAL2Tiles(input_file, output_folder, options)
    gdal2tiles.open_input()
    gdal2tiles.generate_metadata()
    tile_job_info, tile_details = gdal2tiles.generate_base_tiles()
    return tile_job_info, tile_details


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
