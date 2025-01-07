from osgeo import gdal


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
