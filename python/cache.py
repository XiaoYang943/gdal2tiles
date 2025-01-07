import math
import os

from osgeo import gdal


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


def set_cache_max(cache_in_bytes: int) -> None:
    # We set the maximum using `SetCacheMax` and `GDAL_CACHEMAX` to support both fork and spawn as multiprocessing start methods.
    # https://github.com/OSGeo/gdal/pull/2112
    os.environ["GDAL_CACHEMAX"] = "%d" % int(cache_in_bytes / 1024 / 1024)
    gdal.SetCacheMax(cache_in_bytes)
