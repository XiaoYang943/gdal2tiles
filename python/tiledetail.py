class TileDetail:
    tx = 0
    ty = 0
    tz = 0
    rx = 0
    ry = 0
    rxsize = 0
    rysize = 0
    wx = 0
    wy = 0
    wxsize = 0
    wysize = 0
    querysize = 0

    def __init__(self, **kwargs):
        for key in kwargs:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])

    def __str__(self):
        return "TileDetail %s\n%s\n%s\n" % (self.tx, self.ty, self.tz)

    def __repr__(self):
        return "TileDetail %s\n%s\n%s\n" % (self.tx, self.ty, self.tz)
