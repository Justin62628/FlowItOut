""" Plugin that uses ffmpeg to read and write series of images to
a wide range of video formats.

"""

# Heavily inspired from Almar Klein's imageio code
# Copyright (c) 2015, imageio contributors
# distributed under the terms of the BSD License (included in release).
import ctypes
import functools
import os
import subprocess as sp
import sys

import numpy as np

from Utils.StaticParameters import RGB_TYPE
from .abstract import VideoReaderAbstract, VideoWriterAbstract
from .ffprobe import ffprobe
from .. import _FFMPEG_APPLICATION
from .. import _FFMPEG_PATH
from .. import _FFMPEG_SUPPORTED_DECODERS
from .. import _FFMPEG_SUPPORTED_ENCODERS
from .. import _HAS_FFMPEG
from ..utils import *
# uses FFmpeg to read the given file with parameters
from ..utils import startupinfo


class PipeReader(VideoReaderAbstract):
    """Read rgb from stdin

    no-strict subclass of AbstractReader

    """
    def __init__(self, outputfps=-1, inputfps=-1, iw=0, ih=0, pix_fmt="rgb24", verbosity=0):
        self._proc = sys

        self._filename = "-"
        self.verbosity = verbosity

        self.outputfps = outputfps

        # General information
        self.extension = b".raw"
        self.size = 0
        self.probeInfo = {}

        # smartphone video data is weird
        self.rotationAngle = '0'  # specific FFMPEG

        self.inputfps = inputfps

        self.inputwidth = np.int(iw)
        self.inputheight = np.int(ih)

        self.bpp = -1  # bits per pixel
        self.pix_fmt = ""
        # completely unsure of this:
        self.pix_fmt = pix_fmt

        self.inputdepth = np.int(bpplut[self.pix_fmt][0])
        self.bpp = np.int(bpplut[self.pix_fmt][1])

        self.inputframenum = 0

        self.output_pix_fmt = pix_fmt

        self.outputwidth = self.inputwidth
        self.outputheight = self.inputheight

        self.outputdepth = np.int(bpplut[pix_fmt][0])
        self.outputbpp = np.int(bpplut[pix_fmt][1])
        bitpercomponent = self.outputbpp // self.outputdepth
        if bitpercomponent == 8:
            self.dtype = np.dtype('u1')  # np.uint8
        elif bitpercomponent == 16:
            suffix = pix_fmt[-2:]
            if suffix == 'le':
                self.dtype = np.dtype('<u2')
            elif suffix == 'be':
                self.dtype = np.dtype('>u2')
        else:
            raise ValueError(pix_fmt + 'is not a valid pix_fmt for numpy conversion')

        self._createProcess({}, {}, verbosity)
        self.is_current_dup = False

    def _createProcess(self, inputdict, outputdict, verbosity):
        self._proc = sys
        self._cmd = " "

    def close(self):
        pass

    def _terminate(self, timeout=1.0):
        """ Terminate the sub process.
        """
        pass

    def _read_frame_data(self):
        # Init and check
        framesize = self.outputdepth * self.outputwidth * self.outputheight
        assert self._proc is not None

        try:
            # Read framesize bytes
            pipe_read = self._proc.stdin.buffer.read(framesize * self.dtype.itemsize)
            arr = np.frombuffer(pipe_read, dtype=self.dtype)
            if len(arr) != framesize:
                if self.verbosity > 0:
                    print(self._proc.stdin.buffer.read(600), file=sys.stderr)  # read from stderr to obtain error
                return np.array([])
            # assert len(arr) == framesize
        except Exception as err:
            self._terminate()
            err1 = str(err)
            raise RuntimeError("%s" % (err1,))
        return arr


class FFmpegReader(VideoReaderAbstract):
    """Reads frames using FFmpeg

    Using FFmpeg as a backend, this class
    provides sane initializations meant to
    handle the default case well.

    """

    INFO_AVERAGE_FRAMERATE = "@r_frame_rate"
    INFO_WIDTH = "@width"
    INFO_HEIGHT = "@height"
    INFO_PIX_FMT = "@pix_fmt"
    INFO_DURATION = "@duration"
    INFO_NB_FRAMES = "@nb_frames"
    OUTPUT_METHOD = "image2pipe"

    def __init__(self, *args, **kwargs):
        assert _HAS_FFMPEG, "Cannot find installation of real FFmpeg (which comes with ffprobe)."
        super(FFmpegReader, self).__init__(*args, **kwargs)

    def _createProcess(self, inputdict, outputdict, verbosity):
        if '-vcodec' not in outputdict:
            outputdict['-vcodec'] = "rawvideo"

        iargs = self._dict2Args(inputdict)
        oargs = self._dict2Args(outputdict)

        if verbosity > 0:
            cmd = [_FFMPEG_PATH + "/" + _FFMPEG_APPLICATION, ] + \
                  iargs + ['-i', self._filename] + oargs + ['-']
            try:
                print("FFmpeg Read Command:", " ".join(cmd), file=sys.stderr)
            except UnicodeEncodeError:
                print("FFmpeg Read Command: NON-ASCII character exists in command, not shown", file=sys.stderr)
        else:
            cmd = [_FFMPEG_PATH + "/" + _FFMPEG_APPLICATION, "-nostats", "-loglevel", "0"] + \
                  iargs + ['-i', self._filename] + oargs + ['-']
        self._proc = sp.Popen(cmd, stdin=sp.PIPE,
                              stdout=sp.PIPE, stderr=sp.PIPE, startupinfo=startupinfo)
        self._cmd = " ".join(cmd)
        pass

    def _probCountFrames(self):
        # open process, grabbing number of frames using ffprobe
        probecmd = [_FFMPEG_PATH + "/ffprobe"] + ["-v", "error", "-count_frames", "-select_streams", "v:0",
                                                  "-show_entries", "stream=nb_read_frames", "-of",
                                                  "default=nokey=1:noprint_wrappers=1", self._filename]
        return np.int(check_output(probecmd).decode().split('\n')[0])

    def _probe(self):
        return ffprobe(self._filename)

    def _getSupportedDecoders(self):
        return _FFMPEG_SUPPORTED_DECODERS


class PipeWriter(VideoWriterAbstract):
    """Write rgb to stdout

    no-strict subclass of VideoWriterAbstract

    """

    def __init__(self, fps_frac: tuple, colormatrix="709", verbosity=0):
        """Prepares parameters

        """
        self._proc = None
        self._cmd = None
        try:
            self.DEVNULL = open(os.devnull, 'wb')
        except FileNotFoundError:
            self.DEVNULL = open('null', 'wb')

        filename = "-"
        self.extension = b".raw"

        self._filename = filename

        self.inputdict = {}
        self.outputdict = {}
        self.verbosity = verbosity

        self.vs_matrix = "bt709"

        if "-f" not in self.inputdict:
            self.inputdict["-f"] = "rawvideo"
        self.warmStarted = False

        self.fps_frac = fps_frac

        self._update_vs_matrix(colormatrix)

    def _update_vs_matrix(self, colormatrix):
        if '709' in colormatrix:
            self.vs_matrix = "709"
        elif '470' in colormatrix:
            self.vs_matrix = "470bg"
        elif '170' in colormatrix:
            self.vs_matrix = "170m"
        elif '2020' in colormatrix:
            self.vs_matrix = "2020ncl"

    def _createProcess(self, inputdict, outputdict, verbosity):
        self._proc = sys
        self._cmd = " "

    def close(self):
        """Closes the video and terminates pipe process

        """
        self.DEVNULL.close()

    def writeFrame(self, im):
        import vapoursynth as vs
        from vapoursynth import core

        vid = vshape(im)
        T, M, N, C = vid.shape
        if not self.warmStarted:
            self._warmStart(M, N, C, im.dtype)
            self._proc.stdout.buffer.write(f"YUV4MPEG2 C444p10 W{N} H{M} F{self.fps_frac[0]}:{self.fps_frac[1]} Ip A1:1\n".encode())

        assert self._proc is not None  # Check status

        try:
            if RGB_TYPE.DTYPE == np.uint16:
                input_format = vs.RGB48
            else:
                input_format = vs.RGB24
            clip = core.std.BlankClip(width=N, height=M, format=input_format, length=1)

            def get_c_dtype(frame):
                st = frame.format.sample_type
                bps = frame.format.bytes_per_sample
                if st == vs.INTEGER:
                    if bps == 1:
                        dtype = ctypes.c_uint8
                    elif bps == 2:
                        dtype = ctypes.c_uint16
                    else:
                        raise ValueError('Wrong bps type!')
                else:
                    raise ValueError('Wrong st type!')
                return dtype

            def get_raw_plane_from_c(frame, pi, w=False):
                dtype = get_c_dtype(frame)
                if w:
                    ptr = frame.get_write_ptr(pi).value
                else:
                    ptr = frame.get_read_ptr(pi).value
                arr = np.ctypeslib.as_array((dtype * frame.width * frame.height).from_address(ptr))
                return arr

            def get_vsFrame(n, f, npArray):
                vsFrame = f.copy()
                [np.copyto(get_raw_plane_from_c(vsFrame, pi, w=True), npArray[:, :, pi]) for pi in range(3)]
                return vsFrame

            clip = core.std.ModifyFrame(clip, clip, functools.partial(get_vsFrame, npArray=im))
            clip = core.resize.Bicubic(clip=clip, format=vs.YUV444P10, matrix_s=self.vs_matrix)
            self._proc.stdout.buffer.write(f"FRAME\n".encode())
            for i in range(3):
                self._proc.stdout.buffer.write(get_raw_plane_from_c(clip.get_frame(0), i).tostring())
        except IOError as e:
            # Show the command and stderr from pipe
            msg = '{0:}\n\nPIPE COMMAND:\n{1:}\n\nPIPE STDERR ' \
                  'OUTPUT:\n'.format(e, self._cmd)
            raise IOError(msg)


class FFmpegWriter(VideoWriterAbstract):
    """Writes frames using FFmpeg

    Using FFmpeg as a backend, this class
    provides sane initializations for the default case.
    """

    def __init__(self, *args, **kwargs):
        assert _HAS_FFMPEG, "Cannot find installation of real FFmpeg (which comes with ffprobe)."
        super(FFmpegWriter, self).__init__(*args, **kwargs)

    def _getSupportedEncoders(self):
        return _FFMPEG_SUPPORTED_ENCODERS

    def _createProcess(self, inputdict, outputdict, verbosity):
        iargs = self._dict2Args(inputdict)
        oargs = self._dict2Args(outputdict)

        cmd = [_FFMPEG_PATH + "/" + _FFMPEG_APPLICATION, "-hide_banner", "-y"] + iargs + ["-i", "-"] + \
              oargs + [self._filename]

        self._cmd = " ".join(cmd)

        # Launch process
        if self.verbosity > 0:
            try:
                print("FFmpeg Write Command:", self._cmd, file=sys.stderr)
            except UnicodeEncodeError:
                print("FFmpeg Write Command: NON-ASCII character exists in command, not shown", file=sys.stderr)
            self._proc = sp.Popen(cmd, stdin=sp.PIPE,
                                  stdout=sp.PIPE, stderr=None, startupinfo=startupinfo)
        else:
            self._proc = sp.Popen(cmd, stdin=sp.PIPE,
                                  stdout=self.DEVNULL, stderr=sp.STDOUT, startupinfo=startupinfo)


class EnccWriter(VideoWriterAbstract):
    """Writes frames using NVencc

    Using NVencc as a backend, this class
    provides sane initializations for the default case.
    """

    def __init__(self, *args, **kwargs):
        assert _HAS_FFMPEG, "Cannot find installation of EncC (with ffmpeg)."
        super(EnccWriter, self).__init__(*args, **kwargs)
        self.vs_matrix = "709"

    def _update_vs_matrix(self, colormatrix):
        if '709' in colormatrix:
            self.vs_matrix = "709"
        elif '470' in colormatrix:
            self.vs_matrix = "470bg"
        elif '170' in colormatrix:
            self.vs_matrix = "170m"
        elif '2020' in colormatrix:
            self.vs_matrix = "2020ncl"
        pass

    def _getSupportedEncoders(self):
        return _FFMPEG_SUPPORTED_ENCODERS

    def _createProcess(self, inputdict, outputdict, verbosity):
        if inputdict['encc'] == "NVENCC":
            _ENCC_APPLICATION = "NVEncC64"
        elif inputdict['encc'] == "QSVENCC":
            _ENCC_APPLICATION = "QSVEncC64"
        else:
            _ENCC_APPLICATION = "VCEEncC64"
        _ENCC_APPLICATION += ".exe"
        inputdict.pop('encc')
        n_inputdict = self._dealWithFFmpegArgs(inputdict)
        n_outputdict = self._dealWithFFmpegArgs(outputdict)
        if '-s' in inputdict:
            n_inputdict.update({'--input-res': inputdict['-s']})
            n_inputdict.pop('-s')
        if '--colormatrix' in outputdict:
            self._update_vs_matrix(outputdict['--colormatrix'])
        """
        !!!Attention!!!
        "--input-csp", "yv12" Yes
        "--input-csp yv12" No
        """

        n_iargs = self._dict2Args(n_inputdict)
        n_oargs = self._dict2Args(n_outputdict)

        cmd = [_FFMPEG_PATH + "/" + _ENCC_APPLICATION] + ["--raw", "--input-csp", "yuv444p10le"] + n_iargs + \
              ["-i", "-", ] + n_oargs + ["-o"] + [self._filename]
        self._cmd = " ".join(cmd)

        # Launch process
        if self.verbosity > 0:
            try:
                print("EnCc Write Command:", self._cmd, file=sys.stderr)
            except UnicodeEncodeError:
                print("EnCc Write Command: NON-ASCII character exists in command, not shown", file=sys.stderr)
            self._proc = sp.Popen(cmd, stdin=sp.PIPE,
                                  stdout=sp.PIPE, stderr=None, startupinfo=startupinfo)
        else:
            self._proc = sp.Popen(cmd, stdin=sp.PIPE,
                                  stdout=self.DEVNULL, stderr=sp.STDOUT, startupinfo=startupinfo, shell=True)

    def writeFrame(self, im: np.ndarray):
        """Sends ndarray frames to NVENCC

        """
        import vapoursynth as vs
        from vapoursynth import core

        vid = vshape(im)
        T, M, N, C = vid.shape
        if not self.warmStarted:
            self._warmStart(M, N, C, im.dtype)

        assert self._proc is not None  # Check status

        try:
            if RGB_TYPE.DTYPE == np.uint16:
                input_format = vs.RGB48
            else:
                input_format = vs.RGB24
            clip_placeholder = core.std.BlankClip(width=N, height=M, format=input_format, length=1)

            def get_vsFrame(n, f, npArray):
                vsFrame = f.copy()
                [np.copyto(np.asarray(vsFrame.get_write_array(i)), npArray[:, :, i]) for i in range(3)]
                return vsFrame

            clip = core.std.ModifyFrame(clip_placeholder, clip_placeholder,
                                        functools.partial(get_vsFrame, npArray=im))
            clip = core.resize.Bicubic(clip=clip, format=vs.YUV444P10, matrix_s=self.vs_matrix)
            for i in range(3):
                self._proc.stdin.write(np.asarray(clip.get_frame(0).get_read_array(i)).tostring())
        except IOError as e:
            # Show the command and stderr from pipe
            msg = '{0:}\n\nENCODE COMMAND:\n{1:}\n\nENCODE STDERR ' \
                  'OUTPUT:{2:}\n'.format(e, self._cmd, sp.STDOUT)
            raise IOError(msg)

    def _dealWithFFmpegArgs(self, args: dict):
        input_args = args.copy()
        pop_list = ['-f', '-pix_fmt']
        for p in pop_list:
            if p in input_args:
                input_args.pop(p)
        return input_args
        pass


class SVTWriter(EnccWriter):
    """Writes frames using SVT

    Using SVT as a backend, this class
    provides sane initializations for the default case.
    """

    def __init__(self, *args, **kwargs):
        assert _HAS_FFMPEG, "Cannot find installation of real SVT (which comes with ffmpeg)."
        super(SVTWriter, self).__init__(*args, **kwargs)

    def _createProcess(self, inputdict, outputdict, verbosity):
        if outputdict['encc'] == "hevc":
            _SVT_APPLICATION = "SvtHevcEncApp"
        elif outputdict['encc'] == "vp9":
            _SVT_APPLICATION = "SvtVp9EncApp"
        else:  # av1
            _SVT_APPLICATION = "SvtAv1EncApp"
        _SVT_APPLICATION += ".exe"
        outputdict.pop('encc')
        n_inputdict = self._dealWithFFmpegArgs(inputdict)
        n_outputdict = self._dealWithFFmpegArgs(outputdict)
        if '-s' in inputdict:
            input_resolution = inputdict['-s'].split('x')
            n_inputdict.update({'-w': input_resolution[0], '-h': input_resolution[1]})
            n_inputdict.pop('-s')
        if '-s' in outputdict:
            n_outputdict.pop('-s')
        if '-bit-depth' in outputdict:
            if outputdict['-bit-depth'] == '10':
                self.bit_depth = 10
        """
        !!!Attention!!!
        "--input-csp", "yv12" Yes
        "--input-csp yv12" No
        """

        iargs = self._dict2Args(inputdict)
        oargs = self._dict2Args(outputdict)
        n_iargs = self._dict2Args(n_inputdict)
        n_oargs = self._dict2Args(n_outputdict)

        # _cmd = [_FFMPEG_PATH + "/" + _FFMPEG_APPLICATION] + ["-i", "-"]
        # if outputdict['-bit-depth'] in ['8']:
        #     _cmd += ["-pix_fmt", "yuv420p"]
        # else:
        #     """10bit"""
        #     _cmd += ["-pix_fmt", "yuv420p10le"]
        # _cmd += ["-f", "rawvideo", "-", "|"]

        cmd = [_FFMPEG_PATH + "/" + _SVT_APPLICATION] + ["-i", "stdin"] + n_iargs + n_oargs + ["-b",
                                                                                               self._filename]
        self._cmd = " ".join(cmd)
        # self._cmd = r"D:\60-fps-Project\ffmpeg\ffmpeg.exe -i D:\60-fps-Project\input_or_ref\Test\【4】暗场+黑边裁切+时长片段+字幕轨合并.mkv -pix_fmt yuv420p10le -f rawvideo - | D:\60-fps-Project\ffmpeg\SvtHevcEncApp.exe -i stdin -fps 25 -n 241 -w 3840 -h 2160 -brr 1 -sharp 1 -q 16 -bit-depth 10 -b D:\60-fps-Project\input_or_ref\Test\svt_output.mp4"
        # cmd = [_FFMPEG_PATH + "/" + _SVT_APPLICATION] + ["-i", "stdin"] + n_iargs + n_oargs + ["-b", self._filename]
        # self._cmd = " ".join(cmd)

        # Launch process
        if self.verbosity > 0:
            try:
                print("SVT Write Command:", self._cmd, file=sys.stderr)
            except UnicodeEncodeError:
                print("SVT Write Command: NON-ASCII character exists in command, not shown", file=sys.stderr)
            self._proc = sp.Popen(cmd, stdin=sp.PIPE,
                                  stdout=sp.PIPE, stderr=None, startupinfo=startupinfo)
        else:
            self._proc = sp.Popen(cmd, stdin=sp.PIPE,
                                  stdout=self.DEVNULL, stderr=sp.STDOUT, startupinfo=startupinfo)
