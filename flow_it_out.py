# Parse Args
import argparse
import os
import sys

import numpy as np
import tqdm

from IOModule.video import VideoInfoProcessor
from Utils.StaticParameters import RGB_TYPE
from Utils.utils import Tools
from skvideo.io.ffmpeg import FFmpegReader, FFmpegWriter

sys.path.append('Flow')
sys.path.append('Flow/RAFT')
sys.path.append('Flow/RAFT/core')
# For the sake of importing RAFT

global_args_parser = argparse.ArgumentParser(prog="#### FlowItOut by Jeanna ####",
                                             description='To generate flow by different trending algorithms')
global_basic_parser = global_args_parser.add_argument_group(title="Basic Settings")
global_basic_parser.add_argument('-i', '--input', dest='input', type=str, required=True,
                                 help="Path of input video")
global_basic_parser.add_argument('-s', '--resize', dest='resize', type=str, default="480x270",
                                 help="Resized Resolution for flow, leave '0' for no-resize")

flow_selection_parser = global_basic_parser.add_mutually_exclusive_group()
flow_selection_parser.add_argument('--raft', action='store_true')
flow_selection_parser.add_argument('--others', action='store_false')  # TODO implement other flow algorithms

global_raft_parser = global_args_parser.add_argument_group(title="RAFT Settings",
                                                           description="Set the following parameters for RAFT")
global_raft_parser.add_argument('--model', default="models/raft.pth", help="restore checkpoint")
global_raft_parser.add_argument('--small', action='store_true', help='use small model')
global_raft_parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
global_raft_parser.add_argument('--alternate_corr', action='store_true',
                                help='use efficent correlation implementation, if alternate_corr is not compiled, do not use')

# Clean Args
global_args = global_args_parser.parse_args()
global_args_dict = dict(vars(global_args))  # update -i -o -c and other parameters to config

if __name__ == '__main__':
    if global_args.raft:
        from FlowInference.inference import RaftInference

        inference = RaftInference(global_args)
    else:
        # Default: RAFT
        from FlowInference.inference import RaftInference

        inference = RaftInference(global_args)
    RGB_TYPE.change_8bit(True)
    project_dir = "Projects"
    os.makedirs(project_dir, exist_ok=True)
    logger = Tools.get_logger("FlowItOut", project_dir)
    video_info = VideoInfoProcessor(global_args.input, logger, project_dir)

    is_resize = True
    if global_args.resize == "0":
        is_resize = False

    reader_input_dict = {'-vsync': 'passthrough',  # Do not touch
                         '-hwaccel': 'auto',  # Use Hardware Acceleration for decoding
                         "-to": '10'  # Up to 10s of preview, can be customized
                         }
    reader_output_dict = {'-map': '0:v:0', '-vframes': '10000000000', '-color_range': 'tv', '-color_primaries': 'bt709',
                          '-colorspace': 'bt709', '-color_trc': 'bt709', }  # Do not touch
    if is_resize:
        reader_output_dict.update({'-s': global_args.resize})

    writer_input_dict = {'-vsync': 'cfr', '-r': str(video_info.fps)}  # Do not touch
    writer_output_dict = {'-r': str(video_info.fps),
                          '-color_range': 'tv', '-color_primaries': 'bt709', '-colorspace': 'bt709',
                          '-color_trc': 'bt709',  # Do not touch
                          '-preset:v': 'p7',  # render preset, decided by the encoder
                          '-c:v': 'h264_nvenc',  # encoder
                          '-pix_fmt': 'yuv420p',
                          '-crf': '16',  # Render Quality, 0-51, smaller the better
                          "-b:v": "0"}

    # NOTE THAT FFmpeg should be within System Environments(PATH), or the follow call would fail
    reader = FFmpegReader(global_args.input, inputdict=reader_input_dict, outputdict=reader_output_dict,
                          outputfps=video_info.fps, inputfps=video_info.fps, verbosity=0)
    writer = FFmpegWriter(global_args.input + ".output.mp4", inputdict=writer_input_dict, outputdict=writer_output_dict,
                          verbosity=0)
    reader_gen = reader.nextFrame()

    f0 = next(reader_gen)
    pbar = tqdm.tqdm(total=video_info.frames_cnt, unit="frames")
    for f1 in reader_gen:
        flow = inference.inference(f0, f1)
        flow = flow.astype(np.uint8)
        writer.writeFrame(flow)
        pbar.update(1)
        f0 = f1

    reader.close()
    writer.close()
