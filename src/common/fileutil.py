import contextlib
import os
import stat

from osgeo import gdal


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


class VSIFile:
    """Expose a simplistic file-like API for a /vsi file"""

    def __init__(self, filename, f):
        self.filename = filename
        self.f = f

    def write(self, content):
        if gdal.VSIFWriteL(content, 1, len(content), self.f) != len(content):
            raise Exception("Error while writing into %s" % self.filename)
