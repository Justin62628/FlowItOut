import os
import threading
import time
from queue import Queue

import cv2
import numpy as np

from IOModule import ImageIO
from Utils.StaticParameters import SupportFormat, RGB_TYPE
from Utils.utils import Tools, AnytimeFpsIndexer


class ImageRead(ImageIO):
    def __init__(self, logger, folder, start_frame=0, input_fps=24.0, output_fps=60.0, resize=(0, 0), **kwargs):
        """Image Reader from folder

        :param start_frame: start reading point relative to output_fps

        example: input_fps = 24, output_fps = 60, start_frame = 60
                 The Reader will start reading images in the input folder from POINT 24

        """
        super().__init__(logger, folder, start_frame)
        self.resize = resize
        self.resize_flag = all(self.resize)
        self.inputfps = input_fps
        self.outputfps = output_fps
        self.is_current_dup = True
        self.ratio = self.outputfps / self.inputfps

        img_list = os.listdir(self.folder)
        img_list.sort()
        for p in img_list:
            fn, ext = os.path.splitext(p)
            if ext.lower() in SupportFormat.img_inputs:
                if self.frame_cnt * self.ratio < start_frame:  # 折算成输出图片数
                    self.frame_cnt += 1  # update frame_cnt
                    continue  # do not read frame until reach start_frame img
                self.img_list.append(os.path.join(self.folder, p))
        self.logger.debug(f"Load {len(self.img_list)} frames at {self.frame_cnt}")

    def get_frames_cnt(self):
        """Interface for OLS to read frames cnt from Image Folder

        the length of input is determined in __init__, affecting all_frames_cnt

        """
        return round(len(self.img_list) * self.ratio)

    def _read_frame_from_path(self, path):
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)[:, :, ::-1].copy()
        if self.resize_flag:
            img = cv2.resize(img, (self.resize[0], self.resize[1]), interpolation=cv2.INTER_AREA)
        return img

    def _read_frame(self):
        for p in self.img_list:
            img = self._read_frame_from_path(p)
            yield img

    def nextFrame(self):
        indexer = AnytimeFpsIndexer(self.inputfps, self.outputfps)
        ogen = self._read_frame()
        frame = Tools.gen_next(ogen)
        if frame is None or len(frame) == 0:
            return
        while True:
            if indexer.isCurrentDup():
                # Duplicate
                self.is_current_dup = True
            else:
                self.is_current_dup = False
                frame = Tools.gen_next(ogen)
                if frame is None or len(frame) == 0:
                    break
            yield frame

    def isCurrentDup(self):
        return self.is_current_dup

    def close(self):
        return


class ImageWrite(ImageIO):
    def __init__(self, logger, folder, start_frame=0, resize=(0, 0), output_ext='.png', thread_cnt=4,
                 is_tool=False, **kwargs):
        super().__init__(logger, folder, start_frame)
        self.resize = resize
        self.resize_flag = all(self.resize)
        self.output_ext = output_ext

        self.thread_cnt = thread_cnt
        self.thread_pool = list()
        self.write_queue = Queue()
        self.frame_cnt = start_frame
        if not is_tool:
            self.logger.debug(f"Start Writing {self.output_ext} at No. {self.frame_cnt}")
            for t in range(self.thread_cnt):
                _t = threading.Thread(target=self._write_buffer, name=f"IMG.IO Write Buffer No.{t + 1}")
                self.thread_pool.append(_t)
            for _t in self.thread_pool:
                _t.start()

    @staticmethod
    def get_write_start_frame(folder: str):
        """Get Start Frame from folder when start_frame is at its default value

        :return:
        """
        img_list = list()
        for f in os.listdir(folder):  # output folder
            fn, ext = os.path.splitext(f)
            if ext in SupportFormat.img_inputs:
                img_list.append(fn)
        if not len(img_list):
            return 0
        return len(img_list)

    def _write_buffer(self):
        while True:
            img_data = self.write_queue.get()
            if img_data[1] is None:
                self.logger.debug(f"{threading.current_thread().name}: get None, break")
                break
            self._write_frame_to_path(img_data[1], img_data[0])

    def _write_frame_to_path(self, img, path):
        if self.resize_flag:
            if img.shape[1] != self.resize[0] or img.shape[0] != self.resize[1]:
                img = cv2.resize(img, (self.resize[0], self.resize[1]), interpolation=cv2.INTER_AREA)
        if img.dtype != RGB_TYPE.DTYPE:
            img = img.astype(RGB_TYPE.DTYPE)
        cv2.imencode(self.output_ext, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tofile(path)

    def writeFrame(self, img):
        img_path = os.path.join(self.folder, f"{self.frame_cnt:0>8d}{self.output_ext}")
        img_path = img_path.replace("\\", "/")
        if img is None:
            for t in range(self.thread_cnt):
                self.write_queue.put((img_path, None))
            return
        self.write_queue.put((img_path, img))
        self.frame_cnt += 1
        return

    def close(self):
        for t in range(self.thread_cnt):
            self.write_queue.put(("", None))
        for _t in self.thread_pool:
            while _t.is_alive():
                time.sleep(0.2)
        return
