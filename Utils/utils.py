# coding: utf-8
import datetime
import hashlib
import logging
import math
import os
import re
import shutil
import signal
import string
import subprocess
import sys
import traceback
from collections import deque
from configparser import ConfigParser, NoOptionError, NoSectionError
from queue import Queue

import cv2
import numpy as np
import psutil
from sklearn import linear_model

from Utils.StaticParameters import RGB_TYPE, IS_RELEASE, IS_CLI
from skvideo.utils import startupinfo


class DefaultConfigParser(ConfigParser):
    """
    自定义参数提取
    """

    def get(self, section, option, fallback=None, raw=False):
        try:
            d = self._unify_values(section, None)
        except NoSectionError:
            if fallback is None:
                raise
            else:
                return fallback
        option = self.optionxform(option)
        try:
            value = d[option]
        except KeyError:
            if fallback is None:
                raise NoOptionError(option, section)
            else:
                return fallback

        if type(value) == str and not len(str(value)):
            return fallback

        if type(value) == str and value in ["false", "true"]:
            if value == "false":
                return False
            return True

        return value

class CliFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(module)s - %(lineno)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class Tools:
    resize_param = (300, 300)
    crop_param = (0, 0, 0, 0)

    def __init__(self):
        pass

    @staticmethod
    def fillQuotation(_str):
        if _str[0] != '"':
            return f'"{_str}"'
        else:
            return _str

    @staticmethod
    def get_logger(name, log_path, debug=False):
        logger = logging.getLogger(name)
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        if IS_CLI:
            logger_formatter = CliFormatter()
        else:
            logger_formatter = logging.Formatter(f'%(asctime)s - %(module)s - %(lineno)s - %(levelname)s - %(message)s')
            if IS_RELEASE:
                logger_formatter = logging.Formatter(f'%(asctime)s - %(module)s - %(levelname)s - %(message)s')

        log_path = os.path.join(log_path, "log")  # private dir for logs
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        logger_path = os.path.join(log_path,
                                   f"{name}-{datetime.datetime.now().date()}.log")

        txt_handler = logging.FileHandler(logger_path, encoding='utf-8')

        txt_handler.setFormatter(logger_formatter)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(logger_formatter)

        logger.addHandler(console_handler)
        logger.addHandler(txt_handler)
        return logger

    @staticmethod
    def make_dirs(dir_lists, rm=False):
        for d in dir_lists:
            if rm and os.path.exists(d):
                shutil.rmtree(d)
                continue
            if not os.path.exists(d):
                os.mkdir(d)
        pass

    @staticmethod
    def gen_next(gen: iter):
        try:
            return next(gen)
        except StopIteration:
            return None

    @staticmethod
    def dict2Args(d: dict):
        args = []
        for key in d.keys():
            args.append(key)
            if len(d[key]):
                args.append(d[key])
        return args

    @staticmethod
    def clean_parsed_config(args: dict) -> dict:
        for a in args:
            if args[a] in ["false", "true"]:
                if args[a] == "false":
                    args[a] = False
                else:
                    args[a] = True
                continue
            try:
                tmp = float(args[a])
                try:
                    if not tmp - int(args[a]):
                        tmp = int(args[a])
                except ValueError:
                    pass
                args[a] = tmp
                continue
            except ValueError:
                pass
            if not len(args[a]):
                print(f"INFO: Find Empty Arguments at '{a}'", file=sys.stderr)
                args[a] = ""
        return args
        pass

    @staticmethod
    def check_pure_img(img1):
        try:
            if np.var(img1[::4, ::4, 0]) < 10:
                return True
            return False
        except:
            return False

    @staticmethod
    def check_non_ascii(s: str):
        ascii_set = set(string.printable)
        _s = ''.join(filter(lambda x: x in ascii_set, s))
        if s != _s:
            return True
        else:
            return False

    @staticmethod
    def get_u1_from_u2_img(img: np.ndarray):
        if img.dtype in (np.uint16, np.dtype('>u2'), np.dtype('<u2')):
            img = img.view(np.uint8)[:, :, ::2]  # default to uint8
        return img

    @staticmethod
    def get_norm_img(img1, resize=True):
        img1 = Tools.get_u1_from_u2_img(img1)
        if img1.shape[0] > 1000:
            img1 = img1[::4, ::4, 0]
        else:
            img1 = img1[::2, ::2, 0]
        if resize and img1.shape[0] > Tools.resize_param[0]:
            img1 = cv2.resize(img1, Tools.resize_param)
        img1 = cv2.equalizeHist(img1)  # 进行直方图均衡化
        return img1

    @staticmethod
    def get_norm_img_diff(img1, img2, resize=True, is_flow=False) -> float:
        """
        Normalize Difference
        :param resize:
        :param img1: cv2
        :param img2: cv2
        :param is_flow: bool
        :return: float
        """

        def fd(_i0, _i1):
            """
            Calculate Flow Distance
            :param _i0: np.ndarray
            :param _i1: np.ndarray
            :return:
            """
            prev_gray = cv2.cvtColor(_i0, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(_i1, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, flow=None,
                                                pyr_scale=0.5, levels=1, winsize=64, iterations=20,
                                                poly_n=5, poly_sigma=1.1, flags=0)
            x = flow[:, :, 0]
            y = flow[:, :, 1]
            return np.linalg.norm(x) + np.linalg.norm(y)

        if (img1[::4, ::4, 0] == img2[::4, ::4, 0]).all():
            return 0

        if is_flow:
            img1 = Tools.get_u1_from_u2_img(img1)
            img2 = Tools.get_u1_from_u2_img(img2)
            i0 = cv2.resize(img1, (64, 64))
            i1 = cv2.resize(img2, (64, 64))
            diff = fd(i0, i1)
        else:
            img1 = Tools.get_norm_img(img1, resize)
            img2 = Tools.get_norm_img(img2, resize)
            # h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
            diff = cv2.absdiff(img1, img2).mean()

        return diff

    @staticmethod
    def get_norm_img_flow(img1, img2, resize=True, flow_thres=1) -> (int, np.array):
        """
        Normalize Difference
        :param flow_thres: 光流移动像素长
        :param resize:
        :param img1: cv2
        :param img2: cv2
        :return:  (int, np.array)
        """
        prevgray = Tools.get_norm_img(img1, resize)
        gray = Tools.get_norm_img(img2, resize)
        # h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        # prevgray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # 使用Gunnar Farneback算法计算密集光流
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # 绘制线
        step = 10
        h, w = gray.shape[:2]
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines)
        line = []
        flow_cnt = 0

        for l in lines:
            if math.sqrt(math.pow(l[0][0] - l[1][0], 2) + math.pow(l[0][1] - l[1][1], 2)) > flow_thres:
                flow_cnt += 1
                line.append(l)

        cv2.polylines(prevgray, line, 0, (0, 255, 255))
        comp_stack = np.hstack((prevgray, gray))
        return flow_cnt, comp_stack

    @staticmethod
    def get_filename(path):
        if not os.path.isfile(path):
            return os.path.basename(path)
        return os.path.splitext(os.path.basename(path))[0]

    @staticmethod
    def get_mixed_scenes(img0, img1, n):
        """
        return n-1 images
        :param img0:
        :param img1:
        :param n:
        :return:
        """
        step = 1 / n
        beta = 0
        output = list()

        def normalize_img(img):
            if img.dtype in (np.dtype('>u2'), np.dtype('<u2')):
                img = img.astype(np.uint16)
            return img

        img0 = normalize_img(img0)
        img1 = normalize_img(img1)
        for _ in range(n - 1):
            beta += step
            alpha = 1 - beta
            mix = cv2.addWeighted(img0[:, :, ::-1], alpha, img1[:, :, ::-1], beta, 0)[:, :, ::-1].copy()
            output.append(mix)
        return output

    @staticmethod
    def get_fps(path: str):
        """
        Get Fps from path
        :param path:
        :return: fps float
        """
        if not os.path.isfile(path):
            return 0
        try:
            if not os.path.isfile(path):
                input_fps = 0
            else:
                input_stream = cv2.VideoCapture(path)
                input_fps = input_stream.get(cv2.CAP_PROP_FPS)
            return input_fps
        except Exception:
            return 0

    @staticmethod
    def get_existed_chunks(project_dir: str):
        chunk_paths = []
        for chunk_p in os.listdir(project_dir):
            if re.match("chunk-\d+-\d+-\d+\.\w+", chunk_p):
                chunk_paths.append(chunk_p)

        if not len(chunk_paths):
            return chunk_paths, -1, -1

        chunk_paths.sort()
        last_chunk = chunk_paths[-1]
        chunk_cnt, last_frame = re.findall('chunk-(\d+)-\d+-(\d+).*?', last_chunk)[0]
        return chunk_paths, int(chunk_cnt), int(last_frame)

    @staticmethod
    def get_custom_cli_params(_command: str):
        command_params = _command.split('||')
        command_dict = dict()
        param = ""
        for command in command_params:
            command = command.strip().replace("\\'", "'").replace('\\"', '"').strip('\\')
            if command.startswith("-"):
                if param != "":
                    command_dict.update({param: ""})
                param = command
            else:
                command_dict.update({param: command})
                param = ""
        if param != "":  # final note
            command_dict.update({param: ""})
        return command_dict

    @staticmethod
    def popen(args: str, is_stdout=False, is_stderr=False):
        p = subprocess.Popen(args, startupinfo=startupinfo,
                             stdout=subprocess.PIPE if is_stdout else None,
                             stderr=subprocess.PIPE if is_stderr else None,
                             encoding='utf-8')
        return p

    @staticmethod
    def md5(d: str):
        m = hashlib.md5(d.encode(encoding='utf-8'))
        return m.hexdigest()

    @staticmethod
    def get_pids():
        """
        get key-value of pids
        :return: dict {pid: pid-name}
        """
        pid_dict = {}
        pids = psutil.pids()
        for pid in pids:
            try:
                p = psutil.Process(pid)
                pid_dict[pid] = p.name()
            except psutil.NoSuchProcess:
                pass
            # print("pid-%d,pname-%s" %(pid,p.name()))
        return pid_dict

    @staticmethod
    def kill_svfi_related(pid: int, is_rude=False):
        """

        :param is_rude:
        :param pid: PID of One Line Shot Args.exe
        :return:
        """

        try:
            p = Tools.popen(f'wmic process where parentprocessid={pid} get processid', is_stdout=True)
            related_pids = p.stdout.readlines()
            p.stdout.close()
            related_pids = [i.strip() for i in related_pids if i.strip().isdigit()]
            if not len(related_pids):
                return
            taskkill_cmd = "taskkill "
            for p in related_pids:
                taskkill_cmd += f"/pid {p} "
            taskkill_cmd += "/f"
            try:
                p = Tools.popen(taskkill_cmd)
                p.wait(timeout=15)
            except:
                pass
        except FileNotFoundError:
            if is_rude:
                pids = Tools.get_pids()
                for pid, pname in pids.items():
                    if pname in ['ffmpeg.exe', 'ffprobe.exe', 'one_line_shot_args.exe', 'QSVEncC64.exe', 'NVEncC64.exe',
                                 'SvtHevcEncApp.exe', 'SvtVp9EncApp.exe', 'SvtAv1EncApp.exe']:
                        try:
                            os.kill(pid, signal.SIGABRT)
                        except Exception as e:
                            traceback.print_exc()
                        print(f"Warning: Kill Process before exit: {pname}", file=sys.stderr)
                return
            pass

    @staticmethod
    def get_plural(i: int):
        if i > 0:
            if i % 2 != 0:
                return i + 1
        return i


class TransitionDetection_ST:
    def __init__(self, project_dir, scene_queue_length, scdet_threshold=50, no_scdet=False,
                 use_fixed_scdet=False, fixed_max_scdet=50, scdet_output=False):
        """

        :param project_dir: 项目所在文件夹
        :param scene_queue_length:
        :param scdet_threshold:
        :param no_scdet: 无转场检测
        :param use_fixed_scdet: 使用固定转场识别
        :param fixed_max_scdet: 固定转场识别模式下的阈值
        :param scdet_output:
        """
        self.scdet_output = scdet_output
        self.scdet_threshold = scdet_threshold
        self.use_fixed_scdet = use_fixed_scdet
        if self.use_fixed_scdet:
            self.scdet_threshold = fixed_max_scdet
        self.scdet_cnt = 0
        self.scene_stack_len = scene_queue_length
        self.absdiff_queue = deque(maxlen=self.scene_stack_len)  # absdiff队列
        self.black_scene_queue = deque(maxlen=self.scene_stack_len)  # 黑场开场特判队列
        self.scene_checked_queue = deque(maxlen=self.scene_stack_len // 2)  # 已判断的转场absdiff特判队列
        self.utils = Tools
        self.dead_thres = 80
        self.born_thres = 2
        self.img1 = None
        self.img2 = None
        self.scdet_cnt = 0
        self.scene_dir = os.path.join(project_dir, "scene")
        if not os.path.exists(self.scene_dir):
            os.mkdir(self.scene_dir)
        self.scene_stack = Queue(maxsize=scene_queue_length)
        self.no_scdet = no_scdet
        self.scedet_info = {"scene": 0, "normal": 0, "dup": 0, "recent_scene": -1}

    def _check_coef(self):
        reg = linear_model.LinearRegression()
        reg.fit(np.array(range(len(self.absdiff_queue))).reshape(-1, 1), np.array(self.absdiff_queue).reshape(-1, 1))
        return reg.coef_, reg.intercept_

    def _check_var(self):
        coef, intercept = self._check_coef()
        coef_array = coef * np.array(range(len(self.absdiff_queue))).reshape(-1, 1) + intercept
        diff_array = np.array(self.absdiff_queue)
        sub_array = diff_array - coef_array
        return sub_array.var() ** 0.65

    def _judge_mean(self, diff):
        var_before = self._check_var()
        self.absdiff_queue.append(diff)
        var_after = self._check_var()
        if var_after - var_before > self.scdet_threshold and diff > self.born_thres:
            # Detect new scene
            self.scdet_cnt += 1
            self.save_scene(
                f"diff: {diff:.3f}, var_a: {var_before:.3f}, var_b: {var_after:.3f}, cnt: {self.scdet_cnt}")
            self.absdiff_queue.clear()
            self.scene_checked_queue.append(diff)
            return True
        else:
            if diff > self.dead_thres:
                self.absdiff_queue.clear()
                self.scdet_cnt += 1
                self.save_scene(f"diff: {diff:.3f}, Dead Scene, cnt: {self.scdet_cnt}")
                self.scene_checked_queue.append(diff)
                return True
            return False

    def end_view(self):
        self.scene_stack.put(None)
        while True:
            scene_data = self.scene_stack.get()
            if scene_data is None:
                return
            title = scene_data[0]
            scene = scene_data[1]
            self.save_scene(title)

    def save_scene(self, title):
        if not self.scdet_output:
            return
        try:
            comp_stack = np.hstack((self.img1, self.img2))
            comp_stack = cv2.resize(comp_stack, (960, int(960 * comp_stack.shape[0] / comp_stack.shape[1])),
                                    interpolation=cv2.INTER_AREA)
            cv2.putText(comp_stack,
                        title,
                        (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (int(RGB_TYPE.SIZE), 0, 0))
            if "pure" in title.lower():
                path = f"{self.scdet_cnt:08d}_pure.png"
            elif "band" in title.lower():
                path = f"{self.scdet_cnt:08d}_band.png"
            else:
                path = f"{self.scdet_cnt:08d}.png"
            path = os.path.join(self.scene_dir, path)
            if os.path.exists(path):
                os.remove(path)
            cv2.imencode('.png', cv2.cvtColor(comp_stack, cv2.COLOR_RGB2BGR))[1].tofile(path)
            return
            # TODO Preview Add Scene Preview
            cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.imshow(title, cv2.cvtColor(comp_stack, cv2.COLOR_BGR2RGB))
            cv2.moveWindow(title, 500, 500)
            cv2.resizeWindow(title, 1920, 540)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception:
            traceback.print_exc()

    def check_scene(self, _img1, _img2, add_diff=False, no_diff=False, use_diff=-1, **kwargs) -> bool:
        """
        Check if current scene is scene
        :param use_diff:
        :param _img2:
        :param _img1:
        :param add_diff:
        :param no_diff: check after "add_diff" mode
        :return: 是转场则返回真
        """

        if self.no_scdet:
            return False

        img1 = _img1.copy()
        img2 = _img2.copy()
        self.img1 = img1
        self.img2 = img2

        if use_diff != -1:
            diff = use_diff
        else:
            diff = self.utils.get_norm_img_diff(self.img1, self.img2)

        if self.use_fixed_scdet:
            if diff < self.scdet_threshold:
                return False
            else:
                self.scdet_cnt += 1
                self.save_scene(f"diff: {diff:.3f}, Fix Scdet, cnt: {self.scdet_cnt}")
                return True

        # 检测开头黑场
        if diff < 0.001:
            # 000000
            if self.utils.check_pure_img(img1):
                self.black_scene_queue.append(0)
            return False
        elif len(self.black_scene_queue) and np.mean(self.black_scene_queue) == 0:
            # 检测到00000001
            self.black_scene_queue.clear()
            self.scdet_cnt += 1
            self.save_scene(f"diff: {diff:.3f}, Pure Scene, cnt: {self.scdet_cnt}")
            # self.save_flow()
            return True

        # Check really hard scene at the beginning
        if diff > self.dead_thres:
            self.absdiff_queue.clear()
            self.scdet_cnt += 1
            self.save_scene(f"diff: {diff:.3f}, Dead Scene, cnt: {self.scdet_cnt}")
            self.scene_checked_queue.append(diff)
            return True

        if len(self.absdiff_queue) < self.scene_stack_len or add_diff:
            if diff not in self.absdiff_queue:
                self.absdiff_queue.append(diff)
            return False

        # Duplicate Frames Special Judge
        if no_diff and len(self.absdiff_queue):
            self.absdiff_queue.pop()
            if not len(self.absdiff_queue):
                return False

        # Judge
        return self._judge_mean(diff)

    def update_scene_status(self, recent_scene, scene_type: str):
        # 更新转场检测状态
        self.scedet_info[scene_type] += 1
        if scene_type == "scene":
            self.scedet_info["recent_scene"] = recent_scene

    def get_scene_status(self):
        return self.scedet_info


class TransitionDetection:
    def __init__(self, scene_queue_length, scdet_threshold=50, project_dir="", no_scdet=False,
                 use_fixed_scdet=False, fixed_max_scdet=50, remove_dup_mode=0, scdet_output=False, scdet_flow=0,
                 **kwargs):
        """
        转场检测类
        :param scdet_flow: 输入光流模式：0：2D 1：3D
        :param scene_queue_length: 转场判定队列长度
        :param fixed_scdet:
        :param scdet_threshold: （标准输入）转场阈值
        :param output: 输出
        :param no_scdet: 不进行转场识别
        :param use_fixed_scdet: 使用固定转场阈值
        :param fixed_max_scdet: 使用的最大转场阈值
        :param kwargs:
        """
        self.view = False
        self.utils = Tools
        self.scdet_cnt = 0
        self.scdet_threshold = scdet_threshold
        self.scene_dir = os.path.join(project_dir, "scene")  # 存储转场图片的文件夹路径
        if not os.path.exists(self.scene_dir):
            os.mkdir(self.scene_dir)

        self.dead_thres = 80  # 写死最高的absdiff
        self.born_thres = 3  # 写死判定为非转场的最低阈值

        self.scene_queue_len = scene_queue_length
        if remove_dup_mode in [1, 2]:
            # 去除重复帧一拍二或N
            self.scene_queue_len = 8  # 写死

        self.flow_queue = deque(maxlen=self.scene_queue_len)  # flow_cnt队列
        self.black_scene_queue = deque(maxlen=self.scene_queue_len)  # 黑场景特判队列
        self.absdiff_queue = deque(maxlen=self.scene_queue_len)  # absdiff队列
        self.scene_stack = Queue(maxsize=self.scene_queue_len)  # 转场识别队列

        self.no_scdet = no_scdet
        self.use_fixed_scdet = use_fixed_scdet
        self.scedet_info = {"scene": 0, "normal": 0, "dup": 0, "recent_scene": -1}
        # 帧种类，scene为转场，normal为正常帧，dup为重复帧，即两帧之间的计数关系

        self.img1 = None
        self.img2 = None
        self.flow_img = None
        self.before_img = None
        if self.use_fixed_scdet:
            self.dead_thres = fixed_max_scdet

        self.scene_output = scdet_output
        if scdet_flow == 0:
            self.scdet_flow = 3
        else:
            self.scdet_flow = 1

        self.now_absdiff = -1
        self.now_vardiff = -1
        self.now_flow_cnt = -1

    def _check_coef(self):
        reg = linear_model.LinearRegression()
        reg.fit(np.array(range(len(self.flow_queue))).reshape(-1, 1), np.array(self.flow_queue).reshape(-1, 1))
        return reg.coef_, reg.intercept_

    def _check_var(self):
        """
        计算“转场”方差
        :return:
        """
        coef, intercept = self._check_coef()
        coef_array = coef * np.array(range(len(self.flow_queue))).reshape(-1, 1) + intercept
        diff_array = np.array(self.flow_queue)
        sub_array = np.abs(diff_array - coef_array)
        return sub_array.var() ** 0.65

    def _judge_mean(self, flow_cnt, diff, flow):
        # absdiff_mean = 0
        # if len(self.absdiff_queue) > 1:
        #     self.absdiff_queue.pop()
        #     absdiff_mean = np.mean(self.absdiff_queue)

        var_before = self._check_var()
        self.flow_queue.append(flow_cnt)
        var_after = self._check_var()
        self.now_absdiff = diff
        self.now_vardiff = var_after - var_before
        self.now_flow_cnt = flow_cnt
        if var_after - var_before > self.scdet_threshold and diff > self.born_thres and flow_cnt > np.mean(
                self.flow_queue):
            # Detect new scene
            self.see_flow(
                f"flow_cnt: {flow_cnt:.3f}, diff: {diff:.3f}, before: {var_before:.3f}, after: {var_after:.3f}, "
                f"cnt: {self.scdet_cnt + 1}", flow)
            self.flow_queue.clear()
            self.scdet_cnt += 1
            self.save_flow()
            return True
        else:
            if diff > self.dead_thres:
                # 不漏掉死差转场
                self.flow_queue.clear()
                self.see_result(f"diff: {diff:.3f}, False Alarm, cnt: {self.scdet_cnt + 1}")
                self.scdet_cnt += 1
                self.save_flow()
                return True
            # see_result(f"compare: False, diff: {diff}, bm: {before_measure}")
            self.absdiff_queue.append(diff)
            return False

    def end_view(self):
        self.scene_stack.put(None)
        while True:
            scene_data = self.scene_stack.get()
            if scene_data is None:
                return
            title = scene_data[0]
            scene = scene_data[1]
            self.see_result(title)

    def see_result(self, title):
        # 捕捉转场帧预览
        if not self.view:
            return
        comp_stack = np.hstack((self.img1, self.img2))
        cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow(title, cv2.cvtColor(comp_stack, cv2.COLOR_BGR2RGB))
        cv2.moveWindow(title, 0, 0)
        cv2.resizeWindow(title, 960, 270)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_flow(self):
        if not self.scene_output:
            return
        try:
            cv2.putText(self.flow_img,
                        f"diff: {self.now_absdiff:.2f}, vardiff: {self.now_vardiff:.2f}, flow: {self.now_flow_cnt:.2f}",
                        (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0))
            cv2.imencode('.png', cv2.cvtColor(self.flow_img, cv2.COLOR_RGB2BGR))[1].tofile(
                os.path.join(self.scene_dir, f"{self.scdet_cnt:08d}.png"))
        except Exception:
            traceback.print_exc()
        pass

    def see_flow(self, title, img):
        # 捕捉转场帧光流
        if not self.view:
            return
        cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow(title, img)
        cv2.moveWindow(title, 0, 0)
        cv2.resizeWindow(title, 960, 270)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def check_scene(self, _img1, _img2, add_diff=False, use_diff=-1.0) -> bool:
        """
                检查当前img1是否是转场
                :param use_diff: 使用已计算出的absdiff
                :param _img2:
                :param _img1:
                :param add_diff: 仅添加absdiff到计算队列中
                :return: 是转场则返回真
                """
        img1 = _img1.copy()
        img2 = _img2.copy()

        if self.no_scdet:
            return False

        if use_diff != -1:
            diff = use_diff
        else:
            diff = self.utils.get_norm_img_diff(img1, img2)

        if self.use_fixed_scdet:
            if diff < self.dead_thres:
                return False
            else:
                self.scdet_cnt += 1
                return True

        self.img1 = img1
        self.img2 = img2

        # 检测开头转场
        if diff < 0.001:
            # 000000
            if self.utils.check_pure_img(img1):
                self.black_scene_queue.append(0)
            return False
        elif np.mean(self.black_scene_queue) == 0:
            # 检测到00000001
            self.black_scene_queue.clear()
            self.scdet_cnt += 1
            self.see_result(f"absdiff: {diff:.3f}, Pure Scene Alarm, cnt: {self.scdet_cnt}")
            self.flow_img = img1
            self.save_flow()
            return True

        flow_cnt, flow = self.utils.get_norm_img_flow(img1, img2, flow_thres=self.scdet_flow)

        self.absdiff_queue.append(diff)
        self.flow_img = flow

        if len(self.flow_queue) < self.scene_queue_len or add_diff or self.utils.check_pure_img(img1):
            # 检测到纯色图片，那么下一帧大概率可以被识别为转场
            if flow_cnt > 0:
                self.flow_queue.append(flow_cnt)
            return False

        if flow_cnt == 0:
            return False

        # Judge
        return self._judge_mean(flow_cnt, diff, flow)

    def update_scene_status(self, recent_scene, scene_type: str):
        # 更新转场检测状态
        self.scedet_info[scene_type] += 1
        if scene_type == "scene":
            self.scedet_info["recent_scene"] = recent_scene

    def get_scene_status(self):
        return self.scedet_info


class AnytimeFpsIndexer:
    def __init__(self, input_fps: float, output_fps: float):
        self.inputfps = input_fps
        self.outputfps = output_fps
        self.ratio = self.inputfps / self.outputfps
        self.iNow = 0
        self.oNow = 0

    def isCurrentDup(self):
        iNext = self.iNow + 1
        if abs(self.oNow - self.iNow) <= abs(self.oNow - iNext):
            isDup = True
        else:
            self.iNow += 1
            isDup = False
        self.oNow += self.ratio
        return isDup

    def getNow(self):
        return self.iNow


if __name__ == "__main__":
    # u = Tools()
    # cp = DefaultConfigParser(allow_no_value=True)
    # cp.read(r"D:\60-fps-Project\arXiv2020-RIFE-main\release\SVFI.Ft.RIFE_GUI.release.v6.2.2.A\RIFE_GUI.ini",
    #         encoding='utf-8')
    # print(cp.get("General", "UseCUDAButton=true", 6))
    # print(u.clean_parsed_config(dict(cp.items("General"))))
    # dm = DoviMaker(r"D:\60-fps-Project\input_or_ref\Test\output\dolby vision-blocks_71fps_[S-0.5]_[offical_3.8]_963577.mp4", Tools.get_logger('', ''),
    #                r"D:\60-fps-Project\input_or_ref\Test\output\dolby vision-blocks_ec4c18_963577",
    #                ArgumentManager(
    #                    {'ffmpeg': r'D:\60-fps-Project\ffmpeg',
    #                     'input': r"E:\Library\Downloads\Video\dolby vision-blocks.mp4"}),
    #                int(72 / 24),
    #                )
    # dm.run()
    u = Tools()
    # print(u.get_custom_cli_params("-t -d x=\" t\":p=6 -p g='p ':z=1 -qf 3 --dd-e 233"))
    print(u.get_custom_cli_params("-x265-params loseless=1 -preset:v placebo"))
    pass
