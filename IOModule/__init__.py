import os


class ImageIO:
    def __init__(self, logger, folder, start_frame=0, **kwargs):
        """Image I/O Operation Base Class

        :param logger:
        :param folder:
        :param start_frame:
        :param kwargs:
        """
        self.logger = logger
        if folder is None or os.path.isfile(folder):
            raise OSError(f"Invalid Image Sequence Folder: {folder}")
        self.folder = folder  # + "/tmp"  # weird situation, cannot write to target dir, father dir instead
        os.makedirs(self.folder, exist_ok=True)
        self.start_frame = start_frame
        self.frame_cnt = 0
        self.img_list = list()

    def close(self):
        raise NotImplementedError()