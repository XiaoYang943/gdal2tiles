class TileDetail:
    # 瓦片坐标X
    tx = 0
    # 瓦片坐标Y
    ty = 0
    # 瓦片坐标Z
    tz = 0
    # 瓦片左上角像素X
    rx = 0
    # 瓦片左上角像素Y
    ry = 0
    # 瓦片像素宽度
    rxsize = 0
    # 瓦片像素高度
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
