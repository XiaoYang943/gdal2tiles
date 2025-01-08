import logging
import math
import os
import tempfile
from typing import Tuple, List, Any
from uuid import uuid4

from osgeo import gdal, osr

from src.common.constant import MAXZOOMLEVEL
from src.common.fileutil import makedirs, isfile, my_open
from src.core.preprocess import setup_no_data_values, setup_input_srs, has_georeference, reproject_dataset, \
    nb_data_bands, \
    update_alpha_value_for_non_alpha_inputs, update_no_data_values, setup_output_srs
from src.core.tiledetail import TileDetail
from src.core.tilejobinfo import TileJobInfo
from src.log.log import exit_with_error
from src.profile.globalgeodetic import GlobalGeodetic
from src.profile.globalmercator import GlobalMercator

Options = Any
logger = logging.getLogger("gdal2tiles")
tms = {}


class GDAL2Tiles:
    def __init__(self, input_file: str, output_folder: str, options: Options) -> None:
        """Constructor function - initialization"""
        # 切片结果数据格式驱动
        self.out_drv = None

        # mem驱动
        self.mem_drv = None

        # 输入的数据集
        self.warped_input_dataset = None

        # 输出坐标系
        self.out_srs = None

        # 具有所有缩放级别的最小最大瓦片坐标的表
        self.tminmax = None

        # 构造的墨卡托投影
        self.mercator = None

        # 构造的地理坐标系
        self.geodetic = None

        # 数据集的波段数量
        self.dataBandsCount = None

        # 数据集的仿射变换六参数
        self.out_gt = None
        # 由瓦片坐标计算瓦片bound
        self.tileswne = None
        # 数据集的envelop
        self.swne = None
        # 仿射变换六参数-dataSetLeftTopX
        self.ominx = None
        # rightTopX
        self.omaxx = None
        # 仿射变换六参数-dataSetLeftTopY
        self.omaxy = None
        # rightBottomY
        self.ominy = None

        # 是否是4326
        self.isepsg4326 = None

        # 数据集的srs
        self.in_srs = None
        # 数据集的srs的wkt
        self.in_srs_wkt = None

        # 瓦片大小
        self.tile_size = 256
        if options.tilesize:
            self.tile_size = options.tilesize

        # 输出瓦片格式
        self.tiledriver = options.tiledriver
        if options.tiledriver == "PNG":
            self.tileext = "png"
        elif options.tiledriver == "WEBP":
            self.tileext = "webp"
        else:
            self.tileext = "jpg"

        # 是否启用gdal的多进程
        if options.mpi:
            makedirs(output_folder)
            self.tmp_dir = tempfile.mkdtemp(dir=output_folder)
        else:
            self.tmp_dir = tempfile.mkdtemp()

        # 临时vrt文件路径
        self.tmp_vrt_filename = os.path.join(self.tmp_dir, str(uuid4()) + ".vrt")

        # How big should be query window be for scaling down
        # Later on reset according the chosen resampling algorithm
        # 查询大小，4倍应该是为了处理瓦片接边处的像素值
        # 4倍可能太大了，降低了效率，https://github.com/OSGeo/gdal/issues/10803
        # 和瓦片大小保持一致，应该也可以
        # query = tileSize * 1 反而效果更好，瓦片接边处没有明显的线，而4*有
        self.querysize = 4 * self.tile_size

        # 数据集路径
        self.input_file = input_file

        # 输出文件夹路径
        self.output_folder = output_folder

        self.options = options

        # 重采样方法
        if self.options.resampling == "near":
            self.querysize = self.tile_size

        elif self.options.resampling == "bilinear":
            self.querysize = self.tile_size * 2

        # 切片级别
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
                tmaxx, tmaxy = min(2 ** tz - 1, tmaxx), min(2 ** tz - 1, tmaxy)
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
                tmaxx, tmaxy = min(2 ** (tz + 1) - 1, tmaxx), min(2 ** tz - 1, tmaxy)
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
                if self.options.resampling in ("average"):
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
                tmaxx, tmaxy = min(tms.matrix_width * 2 ** tz - 1, tmaxx), min(
                    tms.matrix_height * 2 ** tz - 1, tmaxy
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
            logger.debug("")
            logger.info("开始最高层级切片")

        # 最高级别瓦片坐标bound
        tminx, tminy, tmaxx, tmaxy = self.tminmax[self.tmaxz]

        ds = self.warped_input_dataset
        tilebands = self.dataBandsCount + 1  # TODO 波段数量为什么加1
        querysize = self.querysize  # TODO querysize代表什么

        if self.options.verbose:
            logger.debug("数据集波段数量: %d" % self.dataBandsCount)
            logger.debug("瓦片波段数量: %d" % tilebands)

        tcount = (1 + abs(tmaxx - tminx)) * (1 + abs(tmaxy - tminy))
        logger.debug("顶层瓦片总数: %d" % tcount)

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
                    logger.debug("顶层瓦片切片进度：%d / %d, 瓦片输出路径：%s" % (ti, tcount, tilefilename))

                if self.options.resume and isfile(tilefilename):
                    if self.options.verbose:
                        logger.debug("Tile generation skipped because of --resume")
                    continue

                # 根据瓦片坐标计算瓦片边界
                if self.options.profile == "mercator":
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
                    # b(minx, miny, maxx, maxy)
                    rb, wb = self.geo_query(ds, b[0], b[3], b[2], b[1])

                    # Pixel size in the raster covering query geo extent
                    nativesize = wb[0] + wb[2]
                    if self.options.verbose:
                        # wb是切片有效像元(数据集边界被正方形切割，不完整)的左上角和右下角坐标 TODO 什么坐标？不是像素坐标
                        logger.debug(
                            "Native Extent (querysize %s): %s, %s" % (nativesize, rb, wb)
                        )

                    # Tile bounds in raster coordinates for ReadRaster query
                    rb, wb = self.geo_query(
                        ds, b[0], b[3], b[2], b[1], querysize=querysize
                    )

                    # TODO 为什么又执行一边geo_query
                    # Native Extent (querysize 929): (1985, 2446, 929, 929), (0, 0, 929, 929)
                    # new Native Extent (querysize 929): (1985, 2446, 929, 929), (0, 0, 1024, 1024)
                    logger.debug(
                        "new Native Extent (querysize %s): %s, %s" % (nativesize, rb, wb)
                    )

                    tileLeftTopPixelX, tileLeftTopPixelY, tilePixelWidth, tilePixelHeight = rb
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

                    tileLeftTopPixelX = tx * tsize
                    tilePixelWidth = 0
                    if tx == tmaxx:
                        tilePixelWidth = xsize % tsize
                    if tilePixelWidth == 0:
                        tilePixelWidth = tsize

                    tileLeftTopPixelY = ty * tsize
                    tilePixelHeight = 0
                    if ty == tmaxy:
                        tilePixelHeight = ysize % tsize
                    if tilePixelHeight == 0:
                        tilePixelHeight = tsize

                    wx, wy = 0, 0
                    wxsize = int(tilePixelWidth / float(tsize) * self.tile_size)
                    wysize = int(tilePixelHeight / float(tsize) * self.tile_size)

                    if not self.options.xyz:
                        tileLeftTopPixelY = ysize - (ty * tsize) - tilePixelHeight
                        if wysize != self.tile_size:
                            wy = self.tile_size - wysize

                if tilePixelWidth == 0 or tilePixelHeight == 0 or wxsize == 0 or wysize == 0:
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
                        rx=tileLeftTopPixelX,
                        ry=tileLeftTopPixelY,
                        rxsize=tilePixelWidth,
                        rysize=tilePixelHeight,
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

    def geo_query(self, ds, tileMinX, tileMaxY, tileMaxX, tileMinY, querysize=0):
        geotran = ds.GetGeoTransform()
        dataSetPixelWidth = geotran[1]
        dataSetPixelHeight = geotran[5]
        dataSetLeftTopX = geotran[0]
        dataSetLeftTopY = geotran[3]

        # TODO 为什么要加0.001、0.5? 是否是为了解决python精度的问题？这样是否会造成切片偏移？
        tileLeftTopPixelX = int((tileMinX - dataSetLeftTopX) / dataSetPixelWidth + 0.001)  # 瓦片左上角的像素坐标x(基于数据集左上角)
        tileLeftTopPixelY = int((tileMaxY - dataSetLeftTopY) / dataSetPixelHeight + 0.001)  # 瓦片左上角的像素坐标y(基于数据集左上角)
        tilePixelWidth = max(1, int((tileMaxX - tileMinX) / dataSetPixelWidth + 0.5))  # 瓦片像素宽度
        tilePixelHeight = max(1, int((tileMinY - tileMaxY) / dataSetPixelHeight + 0.5))  # 瓦片像素高度

        # querysize 默认是瓦片宽度*4，即256*4=1024。而不同的重采样方法 querysize 大小不同
        if not querysize:
            wxsize, wysize = tilePixelWidth, tilePixelHeight
        else:
            wxsize, wysize = querysize, querysize

        # Coordinates should not go out of the bounds of the raster
        wx = 0
        if tileLeftTopPixelX < 0:
            rxshift = abs(tileLeftTopPixelX)
            wx = int(wxsize * (float(rxshift) / tilePixelWidth))
            wxsize = wxsize - wx
            tilePixelWidth = tilePixelWidth - int(tilePixelWidth * (float(rxshift) / tilePixelWidth))
            tileLeftTopPixelX = 0
        if tileLeftTopPixelX + tilePixelWidth > ds.RasterXSize:
            wxsize = int(wxsize * (float(ds.RasterXSize - tileLeftTopPixelX) / tilePixelWidth))
            tilePixelWidth = ds.RasterXSize - tileLeftTopPixelX

        wy = 0
        if tileLeftTopPixelY < 0:
            ryshift = abs(tileLeftTopPixelY)
            wy = int(wysize * (float(ryshift) / tilePixelHeight))
            wysize = wysize - wy
            tilePixelHeight = tilePixelHeight - int(tilePixelHeight * (float(ryshift) / tilePixelHeight))
            tileLeftTopPixelY = 0
        if tileLeftTopPixelY + tilePixelHeight > ds.RasterYSize:
            wysize = int(wysize * (float(ds.RasterYSize - tileLeftTopPixelY) / tilePixelHeight))
            tilePixelHeight = ds.RasterYSize - tileLeftTopPixelY

        return (tileLeftTopPixelX, tileLeftTopPixelY, tilePixelWidth, tilePixelHeight), (wx, wy, wxsize, wysize)

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
                        % (args["publishurl"], z, 156543.0339 / 2 ** z, z)
                )
            elif self.options.profile == "geodetic":
                s += (
                        """        <TileSet href="%s%d" units-per-pixel="%.14f" order="%d"/>\n"""
                        % (args["publishurl"], z, 0.703125 / 2 ** z, z)
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
                return (2 ** tz - 1) - ty  # Convert from TMS to XYZ numbering system

            tms = tmsMap[options.profile]
            return (
                    tms.matrix_height * 2 ** tz - 1
            ) - ty  # Convert from TMS to XYZ numbering system

        return ty
