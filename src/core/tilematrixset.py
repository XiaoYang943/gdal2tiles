import math

from osgeo import osr


class TileMatrixSet:
    def __init__(self) -> None:
        self.identifier = None
        self.srs = None
        self.topleft_x = None
        self.topleft_y = None
        self.matrix_width = None  # at zoom 0
        self.matrix_height = None  # at zoom 0
        self.tile_size = None
        self.resolution = None  # at zoom 0
        self.level_count = None

    def GeorefCoordToTileCoord(self, x, y, z, overriden_tile_size):
        res = self.resolution * self.tile_size / overriden_tile_size / (2 ** z)
        tx = int((x - self.topleft_x) / (res * overriden_tile_size))
        # In default mode, we use a bottom-y origin
        ty = int(
            (
                    y
                    - (
                            self.topleft_y
                            - self.matrix_height * self.tile_size * self.resolution
                    )
            )
            / (res * overriden_tile_size)
        )
        return tx, ty

    def ZoomForPixelSize(self, pixelSize, overriden_tile_size):
        "Maximal scaledown zoom of the pyramid closest to the pixelSize."

        for i in range(self.level_count):
            res = self.resolution * self.tile_size / overriden_tile_size / (2 ** i)
            if pixelSize > res:
                return max(0, i - 1)  # We don't want to scale up
        return self.level_count - 1

    def PixelsToMeters(self, px, py, zoom, overriden_tile_size):
        "Converts pixel coordinates in given zoom level of pyramid to EPSG:3857"

        res = self.resolution * self.tile_size / overriden_tile_size / (2 ** zoom)
        mx = px * res + self.topleft_x
        my = py * res + (
                self.topleft_y - self.matrix_height * self.tile_size * self.resolution
        )
        return mx, my

    def TileBounds(self, tx, ty, zoom, overriden_tile_size):
        "Returns bounds of the given tile in georef coordinates"

        minx, miny = self.PixelsToMeters(
            tx * overriden_tile_size,
            ty * overriden_tile_size,
            zoom,
            overriden_tile_size,
        )
        maxx, maxy = self.PixelsToMeters(
            (tx + 1) * overriden_tile_size,
            (ty + 1) * overriden_tile_size,
            zoom,
            overriden_tile_size,
        )
        return (minx, miny, maxx, maxy)

    @staticmethod
    def parse(j: dict) -> "TileMatrixSet":
        assert "identifier" in j
        assert "supportedCRS" in j
        assert "tileMatrix" in j
        assert isinstance(j["tileMatrix"], list)
        srs = osr.SpatialReference()
        assert srs.SetFromUserInput(str(j["supportedCRS"])) == 0
        swapaxis = srs.EPSGTreatsAsLatLong() or srs.EPSGTreatsAsNorthingEasting()
        metersPerUnit = 1.0
        if srs.IsProjected():
            metersPerUnit = srs.GetLinearUnits()
        elif srs.IsGeographic():
            metersPerUnit = srs.GetSemiMajor() * math.pi / 180
        tms = TileMatrixSet()
        tms.srs = srs
        tms.identifier = str(j["identifier"])
        for i, tileMatrix in enumerate(j["tileMatrix"]):
            assert "topLeftCorner" in tileMatrix
            assert isinstance(tileMatrix["topLeftCorner"], list)
            topLeftCorner = tileMatrix["topLeftCorner"]
            assert len(topLeftCorner) == 2
            assert "scaleDenominator" in tileMatrix
            assert "tileWidth" in tileMatrix
            assert "tileHeight" in tileMatrix

            topleft_x = topLeftCorner[0]
            topleft_y = topLeftCorner[1]
            tileWidth = tileMatrix["tileWidth"]
            tileHeight = tileMatrix["tileHeight"]
            if tileWidth != tileHeight:
                raise UnsupportedTileMatrixSet("Only square tiles supported")
            # Convention in OGC TileMatrixSet definition. See gcore/tilematrixset.cpp
            resolution = tileMatrix["scaleDenominator"] * 0.28e-3 / metersPerUnit
            if swapaxis:
                topleft_x, topleft_y = topleft_y, topleft_x
            if i == 0:
                tms.topleft_x = topleft_x
                tms.topleft_y = topleft_y
                tms.resolution = resolution
                tms.tile_size = tileWidth

                assert "matrixWidth" in tileMatrix
                assert "matrixHeight" in tileMatrix
                tms.matrix_width = tileMatrix["matrixWidth"]
                tms.matrix_height = tileMatrix["matrixHeight"]
            else:
                if topleft_x != tms.topleft_x or topleft_y != tms.topleft_y:
                    raise UnsupportedTileMatrixSet("All levels should have same origin")
                if abs(tms.resolution / (1 << i) - resolution) > 1e-8 * resolution:
                    raise UnsupportedTileMatrixSet(
                        "Only resolutions varying as power-of-two supported"
                    )
                if tileWidth != tms.tile_size:
                    raise UnsupportedTileMatrixSet(
                        "All levels should have same tile size"
                    )
        tms.level_count = len(j["tileMatrix"])
        return tms


class UnsupportedTileMatrixSet(Exception):
    pass
