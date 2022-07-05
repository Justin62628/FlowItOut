import glob
import json
import logging
import os
import re
import shlex
import shutil
import sys
import traceback

import cv2
import numpy as np

from Utils.StaticParameters import LUTS_TYPE, HDR_STATE, SupportFormat, TB_LIMIT, EncodePresetAssemply, appDir
from Utils.utils import Tools, AnytimeFpsIndexer
from skvideo import check_output


class Hdr10PlusProcessor:
    def __init__(self, logger: logging.Logger, project_dir: str, render_gap: int,
                 interp_times: int, hdr10_metadata: dict, **kwargs):
        """Parse HDR10+ Metadata for every chunk to be rendered

        TODO: Support Non-integral Multiple times frame rates

        :param logger:
        :param project_dir:
        :param render_gap: args.render_gap
        :param interp_times: exp
        :param kwargs:
        """
        self.logger = logger
        self.project_dir = project_dir
        self.interp_times = interp_times
        self.render_gap = render_gap
        self.hdr10_metadata_dict = hdr10_metadata
        self.hdr10plus_metadata_4interp = []
        self._initialize()

    def _initialize(self):
        if not len(self.hdr10_metadata_dict):
            return
        hdr10plus_metadata = self.hdr10_metadata_dict.copy()
        hdr10plus_metadata = hdr10plus_metadata['SceneInfo']
        hdr10plus_metadata.sort(key=lambda x: int(x['SceneFrameIndex']))
        current_index = -1
        for m in hdr10plus_metadata:
            for j in range(int(self.interp_times)):
                current_index += 1
                _m = m.copy()
                _m['SceneFrameIndex'] = current_index
                self.hdr10plus_metadata_4interp.append(_m)
        return

    def get_hdr10plus_metadata_path_at_point(self, start_frame: int):
        """Dump HDR10+ Metadata json at start frame

        :return: path of metadata json to use immediately
        """
        if not len(self.hdr10plus_metadata_4interp) or start_frame < 0 or start_frame > len(
                self.hdr10plus_metadata_4interp):
            return ""
        if start_frame + self.render_gap < len(self.hdr10plus_metadata_4interp):
            hdr10plus_metadata = self.hdr10plus_metadata_4interp[start_frame:start_frame + self.render_gap]
        else:
            hdr10plus_metadata = self.hdr10plus_metadata_4interp[start_frame:]
        hdr10plus_metadata_path = os.path.join(self.project_dir,
                                               f'hdr10plus_metadata_{start_frame}_{start_frame + self.render_gap}.json')
        json.dump(hdr10plus_metadata, open(hdr10plus_metadata_path, 'w'))
        return hdr10plus_metadata_path.replace('/', '\\')


class DoviProcessor:
    def __init__(self, concat_input: str, original_input: str, project_dir: str, input_fps: float, output_fps: float,
                 logger: logging,
                 is_p5x=False, p5x_control_word=2, is_gen84=False, **kwargs):
        """DOVI Metadata Parser

        :param concat_input: output without DOVI metadata, concatenated from chunks
        :param original_input: the input for SVFI to process
        :param project_dir:
        :param logger:
        :param kwargs:
        """
        self.concat_input = concat_input
        self.original_input = original_input
        self.project_dir = project_dir
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.is_p5x = is_p5x
        self.p5x_control_word = p5x_control_word
        self.logger = logger
        self.ffmpeg = "ffmpeg"
        self.ffprobe = "ffprobe"
        self.dovi_tool = "dovi_tool"
        self.dovi_muxer = "mp4muxer"
        self.video_info, self.audio_info = {}, {}
        self.dovi_profile = 8
        self.is_gen84 = is_gen84

        # Get Startup Info
        self.get_input_info()

        # Fit Generate 84 settings
        if self.is_gen84:
            self.video_info["codec_name"] = "hevc"
            if len(self.audio_info):
                self.audio_info["codec_name"] = "ac3"

        # Initiate Path Parameters
        self.concat_video_stream = os.path.join(self.project_dir, f"concat_video.{self.video_info['codec_name'] if not self.is_gen84 else 'hevc'}")
        self.dv_audio_stream = ""
        self.dv_before_rpu = os.path.join(self.project_dir, f"dv_before_rpu.rpu")
        self.rpu_edit_json = os.path.join(self.project_dir, 'rpu_duplicate_edit.json')
        self.dv_after_rpu = os.path.join(self.project_dir, f"dv_after_rpu.rpu")
        self.dv_after_rpu_2 = os.path.join(self.project_dir, f"dv_after_rpu2.rpu")
        self.dv_injected_video_stream = os.path.join(self.project_dir,
                                                     f"dv_injected_video.{self.video_info['codec_name']}")
        self.dv_concat_output_path = f'{os.path.splitext(self.concat_input)[0]}_dovi.mp4'

    def get_input_info(self):
        """Parse original input video and audio stream info, only support mp4
        """
        try:
            self.video_info, self.audio_info = VideoInfoProcessor.ffprobe_stream(self.ffprobe, self.original_input)
        except Exception as e:
            self.logger.error(f"Parse Video Info Failed")
            raise e
        if os.path.splitext(self.original_input)[1].lower() != ".mp4":  # doesn't support other format except mp4
            # TODO DV in mkv support
            self.audio_info.clear()
        self.logger.info(f"DV Processing [0] - Input Information Extracted")
        pass

    def run(self):
        """Main Point of Processor

        :return:
        """
        try:
            self.demux()
            if not self.is_gen84:
                self.extract_rpu()
                self.modify_rpu()
            else:
                self.generate_rpu()
            self.inject_rpu()
            result = self.mux()
            return result
        except Exception:
            self.logger.error("Dovi Conversion Failed \n" + traceback.format_exc())
            raise Exception

    def demux(self):
        self.logger.info(f"DV Processing [1] - Demuxing Video and Audio Tracks")
        audio_map = {'eac3': 'ac3'}
        command_line = (
            f"{self.ffmpeg} -i {Tools.fillQuotation(self.concat_input)} -c:v copy -an "
            f"-f {self.video_info['codec_name']} "
            f"{Tools.fillQuotation(self.concat_video_stream)} -y")
        check_output(command_line)
        if len(self.audio_info):
            audio_ext = self.audio_info['codec_name']
            if self.audio_info['codec_name'] in audio_map:
                audio_ext = audio_map[self.audio_info['codec_name']]
            self.dv_audio_stream = Tools.fillQuotation(os.path.join(self.project_dir, f"dv_audio.{audio_ext}"))
            command_line = (
                f"{self.ffmpeg} -i {Tools.fillQuotation(self.original_input)} "
                f"-c:a {'copy' if not self.is_gen84 else 'eac3'} -vn {Tools.fillQuotation(self.dv_audio_stream)} -y")
            check_output(command_line)

    def generate_rpu(self):
        """Generate Profile 8.4 for One-Click-HDR Dolby Vision 8.4

        Using static json
        :return:
        """
        self.logger.info(f"DV Processing [2, 3] - Generate Profile 8.4 RPU")
        dovi_len = round(self.get_input_len() * self.output_fps / self.input_fps)
        dv_json_path = os.path.join(appDir, "3x3d.json")
        dv_json = {"cm_version": "V40", "profile": "8.4", "length": 1e10}
        dv_json.update({"length": dovi_len})
        with open(dv_json_path, "w", encoding="utf-8") as w:
            json.dump(dv_json, w)

        command_line = f"{self.dovi_tool} generate " \
                       f"-j {Tools.fillQuotation(dv_json_path)} " \
                       f"-o {Tools.fillQuotation(self.dv_after_rpu)}"
        check_output(command_line, shell=True)

    def extract_rpu(self):
        self.logger.info(f"DV Processing [2] - Start RPU Extracting")
        command_line = (
            f"{self.ffmpeg} -loglevel panic -i {Tools.fillQuotation(self.original_input)} -c:v copy "
            f'-vbsf {self.video_info["codec_name"]}_mp4toannexb -f {self.video_info["codec_name"]} - | '
            f'{self.dovi_tool} extract-rpu -o {Tools.fillQuotation(self.dv_before_rpu)} -')
        check_output(command_line, shell=True)
        pass

    def modify_rpu(self):
        self.logger.info(f"DV Processing [3] - Modifying RPU from {self.input_fps:.3f} to {self.output_fps:.3f}")
        command_line = (
            f"{self.dovi_tool} info -i {Tools.fillQuotation(self.dv_before_rpu)} -f 0")
        rpu_info = check_output(command_line)
        try:
            rpu_info = rpu_info.decode().replace("Parsing RPU file...\n", "")
            rpu_info = json.loads(rpu_info)
            self.dovi_profile = rpu_info["dovi_profile"]
        except Exception as e:
            self.logger.warning(f"Parse RPU Info Failed: {rpu_info}")
            raise e
        dovi_len = self.get_input_len()

        duplicate_list, remove_list = self.generate_edit_dict(dovi_len)

        before_remove_rpu_path = self.dv_after_rpu
        if len(remove_list):
            before_remove_rpu_path = self.dv_after_rpu_2

        edit_dict = {'duplicate': duplicate_list}
        with open(self.rpu_edit_json, 'w') as w:
            json.dump(edit_dict, w)
        command_line = (
            f"{self.dovi_tool} editor -i {Tools.fillQuotation(self.dv_before_rpu)} "
            f"-j {Tools.fillQuotation(self.rpu_edit_json)} "
            f"-o {Tools.fillQuotation(before_remove_rpu_path)}")
        check_output(command_line)
        if len(remove_list):
            edit_dict = {'remove': remove_list}
            with open(self.rpu_edit_json, 'w') as w:
                json.dump(edit_dict, w)
            command_line = (
                f"{self.dovi_tool} editor -i {Tools.fillQuotation(before_remove_rpu_path)} "
                f"-j {Tools.fillQuotation(self.rpu_edit_json)} "
                f"-o {Tools.fillQuotation(self.dv_after_rpu)}")
            check_output(command_line)

    def inject_rpu(self):
        self.logger.info(f"DV Processing [4] - Injecting RPU layer")
        command_line = (
            f"{self.dovi_tool} inject-rpu -i {Tools.fillQuotation(self.concat_video_stream)} "
            f"--rpu-in {Tools.fillQuotation(self.dv_after_rpu)} "
            f"-o {Tools.fillQuotation(self.dv_injected_video_stream)}")
        check_output(command_line)

    def mux(self):
        self.logger.info(f"DV Processing [5] - Start Muxing")
        audio_path = ''
        if len(self.audio_info):
            audio_path = f"-i {Tools.fillQuotation(self.dv_audio_stream)}"
        command_line = f"{self.dovi_muxer} -i {Tools.fillQuotation(self.dv_injected_video_stream)} {audio_path} " \
                       f"-o {Tools.fillQuotation(self.dv_concat_output_path)} " \
                       f"--dv-profile {self.dovi_profile} " \
                       f"--mpeg4-comp-brand mp42,iso6,isom,msdh,dby1 --overwrite " \
                       f"--dv-bl-compatible-id {'4' if self.is_gen84 else '1'}"
        check_output(command_line)
        self.logger.info(f"DV Processing FINISHED")
        return True

    def generate_edit_dict(self, frame_cnt: int):
        duplicate_list = []
        remove_list = []
        if not self.is_p5x:
            indexer = AnytimeFpsIndexer(self.input_fps, self.output_fps)
            while True:
                frame = indexer.getNow()
                if frame >= frame_cnt:
                    break
                dup = 0
                while indexer.isCurrentDup():
                    dup += 1  # update dup and frame(actually)
                if frame == 0:
                    dup = 2  # force first to be dup
                if dup:
                    duplicate_list.append({'source': frame, 'offset': frame, 'length': dup})
        else:
            for frame in range(frame_cnt):
                duplicate_list.append({'source': frame, 'offset': frame, 'length': self.p5x_control_word})
            for frame in range(round(frame_cnt * (self.p5x_control_word + 0.5))):
                if frame and frame % (self.p5x_control_word + 1) == 0:  # A BB [B] CC C, [B] != 0
                    remove_list.append(f"{frame}")
            pass
        return duplicate_list, remove_list

    def get_input_len(self):
        if 'nb_frames' in self.video_info:
            dovi_len = int(self.video_info['nb_frames'])
        elif 'r_frame_rate' in self.video_info and 'duration' in self.video_info:
            frame_rate = self.video_info['r_frame_rate'].split('/')
            frame_rate = int(frame_rate[0]) / int(frame_rate[1])
            dovi_len = round(frame_rate * float(self.video_info['duration']))
        else:
            dovi_len = 0
            try:
                input_stream = cv2.VideoCapture(self.original_input)
                duration = round(input_stream.get(cv2.CAP_PROP_POS_MSEC) / 1000)
                fps = input_stream.get(cv2.CAP_PROP_FPS)
                dovi_len = round(duration * fps)
            except:
                pass
            if not dovi_len:
                raise ZeroDivisionError("Unable to find Length of video")
        return dovi_len


class VideoInfoProcessor:
    """Parse Metadatas of Input File or Folder for SVFI"""
    std_PQ_cdata = {'color_range': 'tv', 'color_transfer': 'smpte2084',
                    'color_space': 'bt2020nc', 'color_primaries': 'bt2020'}
    std_HLG_cdata = {'color_range': 'tv', 'color_transfer': 'arib-std-b67',
                     'color_space': 'bt2020nc', 'color_primaries': 'bt2020'}

    def __init__(self, input_file: str, logger: logging.Logger, project_dir: str,
                 hdr_cube_mode=LUTS_TYPE.NONE,
                 is_pipe_in=False, pipe_in_width=0, pipe_in_height=0, pipe_in_fps=0, **kwargs):
        """

        :param input_file:
        :param logger:
        :param project_dir:
        :param kwargs:
        """
        self.input_file = input_file
        self.is_pipe_in = is_pipe_in
        self.logger = logger
        self.is_img_input = not os.path.isfile(self.input_file) and not is_pipe_in
        self.ffmpeg = "ffmpeg"
        self.ffprobe = "ffprobe"
        self.hdr10_parser = "hdr10plus_parser"
        self.hdr_mode = HDR_STATE.NOT_CHECKED
        self.project_dir = project_dir
        self.color_data_tag = [('color_range', ''),
                               ('color_space', ''),
                               ('color_transfer', ''),
                               ('color_primaries', '')]
        self.fps = 0  # float
        self.fps_frac = (0, 1)
        self.frame_size = (0, 0)  # width, height, float
        self.first_img_frame_size = (0, 0)
        self.frames_cnt = 0  # int
        self.duration = 0
        self.video_info = {'color_range': '', 'color_transfer': '',
                           'color_space': '', 'color_primaries': ''}
        self.master_display_info = {"R": (0, 0), "G": (0, 0), "B": (0, 0), "WP": (0, 0), "LL": (0, 0)}
        self.hdr_cube_mode = hdr_cube_mode
        self.audio_info = dict()
        self.hdr10plus_metadata_path = None

        # Under Pipe Mode:
        self.pipe_in_width = pipe_in_width
        self.pipe_in_height = pipe_in_height
        self.pipe_in_fps = pipe_in_fps

        self.update_info()

    def check_single_image_input(self):
        """
        This is quite amending actually
        By replacing input image into a folder
        :return:
        """
        ext_split = os.path.splitext(self.input_file)
        if ext_split[1].lower() not in SupportFormat.img_inputs:
            return
        self.is_img_input = True
        os.makedirs(ext_split[0], exist_ok=True)
        shutil.copy(self.input_file, os.path.join(ext_split[0], os.path.split(ext_split[0])[1] + ext_split[1]))
        self.input_file = ext_split[0]

    def update_hdr_master_display(self, md_info: dict):
        """

        :param md_info:
        {
                    "side_data_type": "Mastering display metadata",
                    "red_x": "34000/50000",
                    "red_y": "16000/50000",
                    "green_x": "13250/50000",
                    "green_y": "34500/50000",
                    "blue_x": "7500/50000",
                    "blue_y": "3000/50000",
                    "white_point_x": "15635/50000",
                    "white_point_y": "16450/50000",
                    "min_luminance": "40/10000",
                    "max_luminance": "11000000/10000"
                }
        :return:
        """

        def getSplit(s: str):
            return s.split("/")[0]

        self.master_display_info.update({"R": (getSplit(md_info["red_x"]), getSplit(md_info["red_y"])),
                                         "G": (getSplit(md_info["green_x"]), getSplit(md_info["green_y"])),
                                         "B": (getSplit(md_info["blue_x"]), getSplit(md_info["blue_y"])),
                                         "WP": (getSplit(md_info["white_point_x"]), getSplit(md_info["white_point_y"])),
                                         "LL": (getSplit(md_info["max_luminance"]), getSplit(md_info["min_luminance"])),
                                         "CLL": (md_info["max_content"], md_info["max_average"])
                                         })
        EncodePresetAssemply.update_hdr_master_display(self.master_display_info)
        pass

    def is_have_hdr_master_display(self):
        """Check single frame of Input to determine HDR metadata"""
        check_command = (f'{self.ffprobe} -hide_banner -loglevel warning -select_streams v -print_format json '
                         f'-show_frames -read_intervals "%+#1" '
                         f'{Tools.fillQuotation(self.input_file)}')
        result = check_output(shlex.split(check_command))
        try:
            result = json.loads(result)["frames"][0]
            if "side_data_list" in result:
                result = result["side_data_list"]
                md_info = dict()
                for r in result:
                    if r["side_data_type"] in ["Mastering display metadata", "Content light level metadata"]:
                        md_info.update(r)
                # result.sort(key=lambda x: len(x), reverse=True)
                self.update_hdr_master_display(md_info)
                return True
            else:
                return False
        except Exception as e:
            self.logger.error(f"Parse Master Display Info Failed: {result}")
            return False
        pass

    def update_hdr_mode(self):
        self.hdr_mode = HDR_STATE.NONE  # default to be bt709
        if any([i in str(self.video_info) for i in ['dv_profile', 'DOVI']]):
            self.hdr_mode = HDR_STATE.DOLBY_VISION  # Dolby Vision
            self.logger.warning("Dolby Vision Content Detected")
            return
        if "color_transfer" not in self.video_info:
            self.logger.warning("Not Find Color Transfer Characteristics")
            return

        color_trc = self.video_info["color_transfer"]
        if "smpte2084" in color_trc or "bt2020" in color_trc:
            self.hdr_mode = HDR_STATE.CUSTOM_HDR  # hdr(normal)
            self.logger.warning("HDR Content Detected")
            if self.is_have_hdr_master_display():
                """Could be HDR10+"""
                self.hdr_mode = HDR_STATE.HDR10
                self.hdr10plus_metadata_path = os.path.join(self.project_dir, "hdr10plus_metadata.json")
                check_command = (f'{self.ffmpeg} -loglevel panic -i {Tools.fillQuotation(self.input_file)} -c:v copy '
                                 f'-vbsf hevc_mp4toannexb -f hevc - | '
                                 f'{self.hdr10_parser} -o {Tools.fillQuotation(self.hdr10plus_metadata_path)} -')
                self.logger.info("Try to Detect and Extract HDR10+ Dynamic Metadata")
                try:
                    check_output(shlex.split(check_command), shell=True)
                except Exception:
                    self.logger.warning("Failed to extract HDR10+ data")
                    self.logger.error(traceback.format_exc(limit=TB_LIMIT))
                if len(self.get_input_hdr10p_metadata()):
                    self.logger.warning("HDR10+ Content Detected")
                    self.hdr_mode = HDR_STATE.HDR10_PLUS  # hdr10+

        elif "arib-std-b67" in color_trc:
            self.hdr_mode = HDR_STATE.HLG  # HLG
            self.logger.warning("HLG Content Detected")

        # Update One-Click-HDR
        if self.hdr_cube_mode != LUTS_TYPE.NONE:
            if self.hdr_mode == HDR_STATE.NONE:  # BT709
                self.logger.warning(f"One Click HDR Applying: {self.hdr_cube_mode.name}")
                lut_colormatrix = LUTS_TYPE.get_lut_colormatrix(self.hdr_cube_mode)
                if lut_colormatrix is LUTS_TYPE.PQ:
                    self.hdr_mode = HDR_STATE.HDR10  # update input HDR mode
                    self.video_info.update(self.std_PQ_cdata)
                elif lut_colormatrix is LUTS_TYPE.HLG:
                    self.hdr_mode = HDR_STATE.HLG
                    if self.hdr_cube_mode == LUTS_TYPE.DV84:
                        self.hdr_mode = HDR_STATE.DOLBY_VISION
                    self.video_info.update(self.std_HLG_cdata)
                else:
                    pass  # keep original
            else:
                self.logger.warning(f"HDR Content Detected, Neglect Applying One Click HDR")

        pass

    def check_vfr(self):
        """Check whether input is vfr content

        :return:
        """
        check_command = f'{self.ffmpeg} -to 00:00:05 -i {Tools.fillQuotation(self.input_file)} -vf vfrdet -f null -'
        result = list()
        try:
            p = Tools.popen(check_command, is_stderr=True)
            result = p.stderr.read()  # read from stderr (ffmpeg cli output)
            p.stderr.close()
            result = re.findall("VFR:(\d+\.\d+) \((\d+)/(\d+)", str(result))[0]  # select first video stream as input
            vfr_ratio, vfr_frames_cnt, cfr_frames_cnt = float(result[0]), float(result[1]), float(result[2])
            if vfr_ratio > 0.8:  # Over Half frames are vfr frames. # damn
                self.logger.warning(f"Over 80%({vfr_ratio * 100:.2f}%) vfr frames detected at the first 5s of video. "
                                    f"Audio could be out of sync!!!!")
        except Exception as e:
            self.logger.warning(f"Parse VFR Info Failed: {result}")
        pass

    @staticmethod
    def ffprobe_stream(ffprobe: str, input_file: str):
        check_command = (f'{ffprobe} -v error '
                         f'-show_streams -print_format json '
                         f'{Tools.fillQuotation(input_file)}')
        result = check_output(shlex.split(check_command))
        try:
            stream_info = json.loads(result)['streams']  # select first video stream as input
        except Exception as e:
            print(result, file=sys.stderr)
            raise e

        # Select first stream
        video_info, audio_info = dict(), dict()
        for stream in stream_info:
            if stream['codec_type'] == 'video':
                video_info = stream
                break
        for stream in stream_info:
            if stream['codec_type'] == 'audio':
                audio_info = stream
                break
        return video_info, audio_info

    def update_frames_info_ffprobe(self):
        try:
            self.video_info, self.audio_info = self.ffprobe_stream(self.ffprobe, self.input_file)
        except Exception as e:
            self.logger.error(f"Parse Video Info Failed")
            raise e

        for cdt in self.color_data_tag:
            if cdt[0] not in self.video_info:
                self.video_info[cdt[0]] = cdt[1]

        self.update_hdr_mode()

        # update frame size info
        if 'width' in self.video_info and 'height' in self.video_info:
            self.frame_size = (int(self.video_info['width']), int(self.video_info['height']))

        if "r_frame_rate" in self.video_info:
            fps_frac = self.video_info["r_frame_rate"].split('/')
            self.fps = int(fps_frac[0]) / int(fps_frac[1])
            self.logger.info(f"Auto Find FPS in r_frame_rate: {self.fps}")
            self.fps_frac = (int(fps_frac[0]), int(fps_frac[1]))
        if "avg_frame_rate" in self.video_info:
            fps_frac = self.video_info["avg_frame_rate"].split('/')
            fps = int(fps_frac[0]) / int(fps_frac[1])
            if 0 < fps < self.fps:
                self.fps = fps
                self.logger.info(f"Update Input FPS in avg_frame_rate: {self.fps}")
                self.fps_frac = (int(fps_frac[0]), int(fps_frac[1]))
        if self.fps == 0:
            self.logger.warning("Auto Find FPS Failed")
            return False

        if "nb_frames" in self.video_info:
            self.frames_cnt = int(self.video_info["nb_frames"])
            self.logger.info(f"Auto Find frames cnt in nb_frames: {self.frames_cnt}")
        elif "duration" in self.video_info:
            self.duration = float(self.video_info["duration"])
            self.frames_cnt = round(float(self.duration * self.fps))
            self.logger.info(f"Auto Find Frames Cnt by duration deduction: {self.frames_cnt}")
        else:
            self.logger.warning("FFprobe Not Find Frames Cnt")
            return False
        return True

    def update_frames_info_cv2(self):
        if self.is_img_input:
            return
        video_input = cv2.VideoCapture(self.input_file)
        try:
            if not self.fps:
                self.fps = video_input.get(cv2.CAP_PROP_FPS)
            if not self.frames_cnt:
                self.frames_cnt = video_input.get(cv2.CAP_PROP_FRAME_COUNT)
            if not self.duration:
                self.duration = self.frames_cnt / self.fps
            if self.frame_size == (0, 0):
                self.frame_size = (
                    round(video_input.get(cv2.CAP_PROP_FRAME_WIDTH)), round(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        except Exception:
            self.logger.error(traceback.format_exc(limit=TB_LIMIT))

    def update_img_sequence_info(self):
        seq_list = []
        for ext in SupportFormat.img_inputs:
            glob_expression = glob.escape(self.input_file) + f"/*{ext}"
            seq_list.extend(glob.glob(glob_expression))
        if not len(seq_list):
            raise OSError("Input Dir does not contain any valid images(png, jpg, tiff only)")
        self.frames_cnt = len(seq_list)  # original length of sequence, not final output length
        np_read = np.fromfile(os.path.join(self.input_file, seq_list[0]), dtype=np.uint8)
        img = cv2.imdecode(np_read, 1)[:, :, ::-1].copy()
        h, w, _ = img.shape
        self.first_img_frame_size = (w, h)
        # for img input, do not question their size for non-monotonous resolution input
        return

    def update_pipe_args(self):
        self.frame_size = (self.pipe_in_width, self.pipe_in_height)
        self.frames_cnt = 1  # set to 1 frames, 1s of duration
        self.duration = 1
        self.fps = self.pipe_in_fps
        self.hdr_mode = HDR_STATE.NONE

    def update_info(self):
        """Main Point of Processor in Initiation"""
        if self.is_pipe_in:
            self.update_pipe_args()
            return
        self.check_single_image_input()
        if self.is_img_input:
            self.update_img_sequence_info()
            return
        self.update_frames_info_ffprobe()
        self.check_vfr()
        self.update_frames_info_cv2()

    def get_input_color_info(self) -> dict:
        return dict(map(lambda x: (x[0], self.video_info.get(x[0], x[1])), self.color_data_tag))
        pass

    def get_input_hdr10p_metadata(self) -> dict:
        """

        :return: dict
        """
        if self.hdr10plus_metadata_path is not None and os.path.exists(self.hdr10plus_metadata_path):
            try:
                hdr10plus_metadata = json.load(open(self.hdr10plus_metadata_path, 'r'))
                return hdr10plus_metadata
            except json.JSONDecodeError:
                self.logger.error("Unable to Decode HDR10+ Metadata")
        return {}
