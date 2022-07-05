import enum
import os
import platform
import re

import numpy as np

abspath = os.path.abspath(__file__)
appDir = os.path.dirname(os.path.dirname(abspath))

INVALID_CHARACTERS = ["'", '"', '“', '”']
IS_RELEASE = False
TB_LIMIT = 0 if IS_RELEASE else None  # Traceback Limit
PATH_LEN_LIMIT = 230

PURE_SCENE_THRESHOLD = 20

IS_CLI = 'window' not in platform.platform().lower()


class TASKBAR_STATE(enum.Enum):
    TBPF_NOPROGRESS = 0x00000000
    TBPF_INDETERMINATE = 0x00000001
    TBPF_NORMAL = 0x00000002
    TBPF_ERROR = 0x00000004
    TBPF_PAUSED = 0x00000008


class HDR_STATE(enum.Enum):
    AUTO = -2
    NOT_CHECKED = -1
    NONE = 0
    CUSTOM_HDR = 1
    HDR10 = 2
    HDR10_PLUS = 3
    DOLBY_VISION = 4
    HLG = 5


class RT_RATIO(enum.Enum):
    """
    Resolution Transfer Ratio
    """
    AUTO = 0
    WHOLE = 1
    THREE_QUARTERS = 2
    HALF = 3
    QUARTER = 4

    @staticmethod
    def get_auto_transfer_ratio(sr_times: float):
        if sr_times >= 1:
            return RT_RATIO.WHOLE
        elif 0.75 <= sr_times < 1:
            return RT_RATIO.THREE_QUARTERS
        elif 0.5 <= sr_times < 0.75:
            return RT_RATIO.HALF
        else:
            return RT_RATIO.QUARTER

    @staticmethod
    def get_surplus_sr_scale(scale: float, ratio):
        if ratio == RT_RATIO.WHOLE:
            return scale
        elif ratio == RT_RATIO.THREE_QUARTERS:
            return scale * (4 / 3)
        elif ratio == RT_RATIO.HALF:
            return scale * 2
        elif ratio == RT_RATIO.QUARTER:
            return scale * 4
        else:
            return scale

    @staticmethod
    def get_modified_resolution(params: tuple, ratio, is_reverse=False, keep_single=False):
        w, h = params
        mod_ratio = 1
        if ratio == RT_RATIO.WHOLE:
            mod_ratio = 1
        elif ratio == RT_RATIO.THREE_QUARTERS:
            mod_ratio = 0.75
        elif ratio == RT_RATIO.HALF:
            mod_ratio = 0.5
        elif ratio == RT_RATIO.QUARTER:
            mod_ratio = 0.25
        if not is_reverse:
            w, h = int(w * mod_ratio), int(h * mod_ratio)
        else:
            w, h = int(w / mod_ratio), int(h / mod_ratio)
        if not keep_single:
            if w % 2:
                w += 1
            if h % 2:
                h += 1
        return w, h


class SR_TILESIZE_STATE(enum.Enum):
    NONE = 0
    CUSTOM = 1
    VRAM_2G = 2
    VRAM_4G = 3
    VRAM_6G = 4
    VRAM_8G = 5
    VRAM_12G = 6

    @staticmethod
    def get_tilesize(state):
        if state == SR_TILESIZE_STATE.NONE:
            return 0
        if state == SR_TILESIZE_STATE.VRAM_2G:
            return 100
        if state == SR_TILESIZE_STATE.VRAM_4G:
            return 200
        if state == SR_TILESIZE_STATE.VRAM_6G:
            return 1000
        if state == SR_TILESIZE_STATE.VRAM_8G:
            return 1200
        if state == SR_TILESIZE_STATE.VRAM_12G:
            return 2000
        return 100


class SupportFormat:
    img_inputs = ['.png', '.tif', '.tiff', '.jpg', '.jpeg']
    img_outputs = ['.png', '.tiff', '.jpg']
    vid_outputs = ['.mp4', '.mkv', '.mov']


class EncodePresetAssemply:
    encoder = {  # the order should correspond to "slow fast medium" - "3 1 2" out of "0 1 2 3 4", at least 3
        "AUTO": {"AUTO": ["AUTO"]},
        "CPU": {
            "H264,8bit": ["slow", "fast", "medium", "ultrafast", "veryslow", "placebo", ],
            "H264,10bit": ["slow", "fast", "medium", "ultrafast", "veryslow"],
            "H265,8bit": ["slow", "fast", "medium", "ultrafast", "veryslow"],
            "H265,10bit": ["slow", "fast", "medium", "ultrafast", "veryslow"],
            "AV1,8bit": ["slow", "fast", "medium", "ultrafast", "veryslow"],
            "AV1,10bit": ["slow", "fast", "medium", "ultrafast", "veryslow"],
            "ProRes,422": ["hq", "4444", "4444xq"],
            "ProRes,444": ["hq", "4444", "4444xq"],
        },
        "NVENC":
            {"H264,8bit": ["p7", "fast", "hq", "bd", "llhq", "loseless", "slow"],
             "H265,8bit": ["p7", "fast", "hq", "bd", "llhq", "loseless", "slow"],
             "H265,10bit": ["p7", "fast", "hq", "bd", "llhq", "loseless", "slow"], },
        "QSV":
            {"H264,8bit": ["slow", "fast", "medium", "veryslow", ],
             "H265,8bit": ["slow", "fast", "medium", "veryslow", ],
             "H265,10bit": ["slow", "fast", "medium", "veryslow", ], },
        "VCE":
            {"H264,8bit": ["quality", "speed", "balanced"],
             "H265,8bit": ["quality", "speed", "balanced"], },
        "NVENCC":
            {"H264,8bit": ["quality", "performance", "default"],
             "H265,8bit": ["quality", "performance", "default"],
             "H265,10bit": ["quality", "performance", "default"], },
        "VCEENCC":
            {"H264,8bit": ["slow", "fast", "balanced"],
             "H265,8bit": ["slow", "fast", "balanced"],
             "H265,10bit": ["slow", "fast", "balanced"], },
        "QSVENCC":
            {"H264,8bit": ["best", "fast", "balanced", "higher", "high",  "faster", "fastest"],
             "H265,8bit": ["best", "fast", "balanced", "higher", "high",  "faster", "fastest"],
             "H265,10bit": ["best", "fast", "balanced", "higher", "high",  "faster", "fastest"], },
        # "SVT":
        #     {"VP9,8bit": ["slowest", "slow", "fast", "faster"],
        #      "H265,8bit": ["slowest", "slow", "fast", "faster"],
        #      "AV1,8bit": ["slowest", "slow", "fast", "faster"],
        #      },
    }
    ffmpeg_encoders = ["AUTO", "CPU", "NVENC", "QSV", "VCE"]
    encc_encoders = ["NVENCC", "QSVENCC", "VCEENCC"]
    community_encoders = ffmpeg_encoders
    params_libx265s = {
        "fast": "asm=avx512:ref=2:rd=2:ctu=32:min-cu-size=16:limit-refs=3:limit-modes=1:rect=0:amp=0:early-skip=1:fast-intra=1:b-intra=1:rdoq-level=0:tu-intra-depth=1:tu-inter-depth=1:limit-tu=0:max-merge=2:me=1:subme=3:merange=25:weightb=1:strong-intra-smoothing=0:open-gop=0:keyint=250:min-keyint=1:rc-lookahead=15:lookahead-slices=8:b-adapt=1:bframes=4:aq-mode=2:aq-strength=1:qg-size=16:cbqpoffs=-2:crqpoffs=-2:qcomp=0.65:deblock=-1:sao=0:info=0",
        "fast_FD+ZL": "asm=avx512:ref=2:rd=2:ctu=32:min-cu-size=16:limit-refs=3:limit-modes=1:rect=0:amp=0:early-skip=1:fast-intra=1:b-intra=0:rdoq-level=0:tu-intra-depth=1:tu-inter-depth=1:limit-tu=0:max-merge=2:me=1:subme=3:merange=25:weightp=0:strong-intra-smoothing=0:open-gop=0:keyint=50:min-keyint=1:rc-lookahead=25:lookahead-slices=8:b-adapt=0:bframes=0:aq-mode=2:aq-strength=1:qg-size=16:cbqpoffs=-2:crqpoffs=-2:qcomp=0.65:deblock=false:sao=0:info=0",
        "slow": "asm=avx512:pmode=1:ref=4:rd=4:ctu=32:min-cu-size=8:limit-refs=1:limit-modes=1:rect=0:amp=0:early-skip=0:fast-intra=0:b-intra=1:rdoq-level=2:tu-intra-depth=1:tu-inter-depth=1:limit-tu=0:max-merge=4:me=3:subme=5:merange=25:weightb=1:strong-intra-smoothing=0:psy-rd=2:psy-rdoq=1:open-gop=0:keyint=250:min-keyint=1:rc-lookahead=35:lookahead-slices=4:b-adapt=2:bframes=6:aq-mode=2:aq-strength=0.8:qg-size=8:cbqpoffs=-2:crqpoffs=-2:qcomp=0.65:deblock=-1:sao=0:info=0",
        "slow_FD+ZL": "asm=avx512:pmode=1:ref=4:rd=4:ctu=32:min-cu-size=8:limit-refs=1:limit-modes=1:rect=0:amp=0:early-skip=0:fast-intra=0:b-intra=0:rdoq-level=2:tu-intra-depth=1:tu-inter-depth=1:limit-tu=0:max-merge=4:me=3:subme=5:merange=25:weightp=0:strong-intra-smoothing=0:psy-rd=2:psy-rdoq=1:open-gop=0:keyint=50:min-keyint=1:rc-lookahead=25:lookahead-slices=4:b-adapt=0:bframes=0:aq-mode=2:aq-strength=0.8:qg-size=8:cbqpoffs=-2:crqpoffs=-2:qcomp=0.65:deblock=false:sao=0:info=0",
    }

    params_libx264s = {
        "fast": "keyint=250:min-keyint=1:bframes=3:b-adapt=1:open-gop=0:ref=2:rc-lookahead=20:chroma-qp-offset=-1:aq-mode=1:aq-strength=0.9:mbtree=0:qcomp=0.60:weightp=1:me=hex:merange=16:subme=7:psy-rd='1.0:0.0':mixed-refs=0:trellis=1:deblock='-1:-1'",
        "fast_FD+ZL": "keyint=50:min-keyint=1:bframes=0:b-adapt=0:open-gop=0:ref=2:rc-lookahead=25:chroma-qp-offset=-1:aq-mode=1:aq-strength=0.9:mbtree=0:qcomp=0.60:weightp=0:me=hex:merange=16:subme=7:psy-rd='1.0:0.0':mixed-refs=0:trellis=1:deblock=false:cabac=0:weightb=0",
        "slow": "keyint=250:min-keyint=1:bframes=6:b-adapt=2:open-gop=0:ref=8:rc-lookahead=35:chroma-qp-offset=0:aq-mode=1:aq-strength=0.9:mbtree=1:qcomp=0.75:partitions=all:direct=auto:weightp=2:me=umh:merange=24:subme=10:psy-rd='1.0:0.1':mixed-refs=1:trellis=2:deblock='-1:-1'",
        "slow_FD+ZL": "keyint=50:min-keyint=1:bframes=0:b-adapt=0:open-gop=0:ref=8:rc-lookahead=25:chroma-qp-offset=0:aq-mode=1:aq-strength=0.9:mbtree=1:qcomp=0.75:partitions=all:direct=auto:weightp=0:me=umh:merange=24:subme=10:psy-rd='1.0:0.1':mixed-refs=1:trellis=2:deblock=false:cabac=0:weightb=0",
    }

    h265_hdr10_info = "master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50):max-cll=1000,100:hdr10-opt=1:repeat-headers=1"  # no need to transfer color metadatas(useless)
    h264_hdr10_info = "mastering-display='G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50)'"
    master_display_info = 'G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50)'
    max_cll_info = '1000,100'

    @staticmethod
    def update_hdr_master_display(md_info):
        EncodePresetAssemply.master_display_info = f"G({md_info['G'][0]},{md_info['G'][1]})" \
                                                   f"B({md_info['B'][0]},{md_info['B'][1]})" \
                                                   f"R({md_info['R'][0]},{md_info['R'][1]})" \
                                                   f"WP({md_info['WP'][0]},{md_info['WP'][1]})"\
                                                   f"L({md_info['LL'][0]},{md_info['LL'][1]})"
        EncodePresetAssemply.max_cll_info = f"{md_info['CLL'][0]},{md_info['CLL'][1]}"
        EncodePresetAssemply.h264_hdr10_info = re.sub("mastering-display='.*?'",
                                                      f"mastering-display='{EncodePresetAssemply.master_display_info}'",
                                                      EncodePresetAssemply.h264_hdr10_info)
        EncodePresetAssemply.h265_hdr10_info = re.sub("master-display=.*?:max-cll=.*?:",
                                                      f"master-display={EncodePresetAssemply.master_display_info}:"
                                                      f"max-cll={EncodePresetAssemply.max_cll_info}:",
                                                      EncodePresetAssemply.h265_hdr10_info)

    @staticmethod
    def get_encoder_format(encoder: str, filter_str: str):
        if encoder not in EncodePresetAssemply.encoder:
            return ["H264,8bit"]
        formats = EncodePresetAssemply.encoder[encoder].keys()
        formats = list(filter(lambda x: filter_str in x, formats))
        return formats


class RGB_TYPE:
    SIZE = 65535.
    DTYPE = np.uint8 if SIZE == 255. else np.uint16

    @staticmethod
    def change_8bit(d8: bool):
        if d8:
            RGB_TYPE.SIZE = 255.
            RGB_TYPE.DTYPE = np.uint8 if RGB_TYPE.SIZE == 255. else np.uint16


class LUTS_TYPE(enum.Enum):
    NONE = 0
    SaturationPQ = 1
    ColorimetricPQ = 2
    ColorimetricHLG = 3
    DV84 = 4

    PQ = 0xfff
    HLG = 0xffe
    SDR = 0xffd

    @staticmethod
    def get_lut_path(lut_type):
        if lut_type is LUTS_TYPE.NONE:
            return
        elif lut_type is LUTS_TYPE.SaturationPQ:
            return "1x3d.cube"
        elif lut_type is LUTS_TYPE.ColorimetricPQ:
            return "1x3d2.cube"
        elif lut_type is LUTS_TYPE.ColorimetricHLG:
            return "2x3d.cube"
        elif lut_type is LUTS_TYPE.DV84:
            return "3x3d.cube"

    @staticmethod
    def get_lut_colormatrix(lut_type):
        """
        Get colormatrix for lut
        :param lut_type:
        :return:
        """
        if lut_type in [LUTS_TYPE.SaturationPQ, LUTS_TYPE.ColorimetricPQ]:
            return LUTS_TYPE.PQ
        elif lut_type in [LUTS_TYPE.ColorimetricHLG, LUTS_TYPE.DV84]:
            return LUTS_TYPE.HLG
        return LUTS_TYPE.SDR


class ALGO_TYPE(enum.Enum):
    @staticmethod
    def get_model_version(model_path: str):
        raise NotImplementedError()

    @staticmethod
    def get_available_algos(algo_type: str, is_full_path=False):
        algo_dir = os.path.join(appDir, "models", algo_type)
        algos = [os.path.join(algo_dir, d) for d in os.listdir(algo_dir)]
        algos = list(filter(lambda x: not os.path.isfile(x), algos))
        if not is_full_path:
            algos = list(map(lambda x: os.path.basename(x), algos))
        return algos

    @staticmethod
    def get_available_models(algo_type: str, algo: str, is_file=False, is_full_path=False):
        model_dir = os.path.join(appDir, "models", algo_type, algo, "models")
        models = [os.path.join(model_dir, d) for d in os.listdir(model_dir)]
        if is_file:
            models = list(filter(lambda x: os.path.isfile(x), models))
        else:
            models = list(filter(lambda x: not os.path.isfile(x), models))
        if not is_full_path:
            models = list(map(lambda x: os.path.basename(x), models))
        return models


class GLOBAL_PARAMETERS:
    CURRENT_CUDA_ID = 0


class RIFE_TYPE(ALGO_TYPE):
    """
    0000000 = RIFEv2, RIFEv3, RIFEv6, RIFEv7/RIFE 4.0, RIFEvNew from Master Zhe, XVFI, ABME

        Rv2 Rv3 Rv6 Rv4 Rv7 Rv+ XVFI ABME RIFT AUTO
    DS   1   1   1   1   1   1    0   0    0    1
    TTA  1   1   1   1   1   0    0   1    1    1
    MC   0   0   0   0   0   0    0   0    0    1
    EN   1   1   1   1   1   1    0   0    0    1
    OE   0   0   0   0   0   1    0   0    0    1
    """
    RIFEv2       =          0b1000000000000000
    RIFEv3       =          0b0100000000000000
    RIFEv6       =          0b0010000000000000
    RIFEv4       =          0b0001000000000000
    RIFEv7       =          0b0000100000000000
    RIFEvPlus    =          0b0000010000000000
    XVFI         =          0b0000001000000000
    ABME         =          0b0000000100000000
    RIFT         =          0b0000000010000000
    IFRNET       =          0b0000000001000000
    IFUNET       =          0b0000000000100000
    M2M          =          0b0000000000010000
    NCNNv2       =          0b0000000000001000
    NCNNv3       =          0b0000000000000100
    NCNNv4       =          0b0000000000000010
    AUTO         =          0b0000000000000001

    DS =                    0b1111110000000001
    TTA =                   0b1111100110000001
    ENSEMBLE =              0b1111110000000001
    MULTICARD =             0b0000000000000001
    OUTPUTMODE =            0b0000010000000001

    @staticmethod
    def get_model_version(model_path: str):
        """
        Return Model Type of VFI
        :param model_path:
        :return:

        Register New VFI Model
        1. here, add distinguish rule
        1.x update AUTO rules in TaskArgumentManager Initiation
        2. update __check_interp_prerequisite in OLS
        3. update _generate_padding in inference_rife
        4. update initiate_algorithm in inference_rife for two classes respectively
        """
        model_path = model_path.lower()
        # AUTO
        if 'auto' in model_path:
            current_model_index = RIFE_TYPE.AUTO
        # CUDA Model
        elif 'abme_best' in model_path: # unavailable
            current_model_index = RIFE_TYPE.ABME
        elif 'rpa_' in model_path or 'rpr_' in model_path:  # RIFE Plus Anime, prior than anime
            current_model_index = RIFE_TYPE.RIFEvPlus  # RIFEv New from Master Zhe
        elif 'anime_' in model_path:
            if any([i in model_path for i in ['sharp', 'smooth']]):
                current_model_index = RIFE_TYPE.RIFEv2  # RIFEv2
            else:  # RIFEv6, anime_training
                current_model_index = RIFE_TYPE.RIFEv6
        elif 'official_' in model_path:
            if '2.' in model_path:
                current_model_index = RIFE_TYPE.RIFEv2  # RIFEv2.x
            elif '3.' in model_path:
                current_model_index = RIFE_TYPE.RIFEv3
            elif 'v6' in model_path:
                current_model_index = RIFE_TYPE.RIFEv6
            elif '4.' in model_path:
                current_model_index = RIFE_TYPE.RIFEv4
            else:  # RIFEv7
                current_model_index = RIFE_TYPE.RIFEv7
        elif 'xvfi_' in model_path: # unavailable
            current_model_index = RIFE_TYPE.XVFI
        elif 'rift' in model_path:
            current_model_index = RIFE_TYPE.RIFT  # unavailable
        elif 'ifrnet' in model_path:
            current_model_index = RIFE_TYPE.IFRNET
        elif 'ifunet' in model_path:
            current_model_index = RIFE_TYPE.IFUNET
        elif 'm2m' in model_path:
            current_model_index = RIFE_TYPE.M2M
        # NCNN Model:
        elif 'rife-v2' in model_path:
            current_model_index = RIFE_TYPE.NCNNv2  # RIFEv2.x
        elif 'rife-v3' in model_path:
            current_model_index = RIFE_TYPE.NCNNv3  # RIFEv3.x
        elif 'rife-v4' in model_path:
            current_model_index = RIFE_TYPE.NCNNv4  # RIFEv4.x
        # Default
        else:
            current_model_index = RIFE_TYPE.RIFEv2  # default RIFEv2
        return current_model_index

    @staticmethod
    def update_current_gpu_id(gpu_id: int):
        GLOBAL_PARAMETERS.CURRENT_CUDA_ID = gpu_id


RIFE_ANYTIME = [RIFE_TYPE.RIFEv4, RIFE_TYPE.RIFEv7, RIFE_TYPE.RIFEvPlus, RIFE_TYPE.IFUNET,
                RIFE_TYPE.IFRNET, RIFE_TYPE.M2M]
RIFE_NCNN = [RIFE_TYPE.NCNNv2, RIFE_TYPE.NCNNv3, RIFE_TYPE.NCNNv4]
RIFE_ANYTIME.extend(RIFE_NCNN)


class SR_TYPE(ALGO_TYPE):
    """
    Model Type Index Table of Super Resolution used by SVFI
    """
    Anime4K       = 0b1000000000000
    RealESR       = 0b0100000000000
    RealCUGAN     = 0b0010000000000
    NcnnCUGAN     = 0b0001000000000
    WaifuCUDA     = 0b0000100000000
    Waifu2x       = 0b0000010000000
    NcnnRealESR   = 0b0000001000000
    BasicVSRPP    = 0b0000000100000
    RealBasicVSR  = 0b0000000010000
    BasicVSRPPR   = 0b0000000001000
    PureBasicVSR  = 0b0000000000100
    TensorRT      = 0b0000000000010
    AUTO          = 0b0000000000001

    @staticmethod
    def get_model_version(model_path: str):
        """
        Return Model Type of SR
        :param model_path:
        :return:

        Register New SR Model
        1. Register New SR Model here (Map: Name -> Model Location
        1.x update AUTO rules in TaskArgumentManager Initiation
        2. Specify NCNN or CUDA Model.
        3. on_AiSrSelector_currentTextChanged, settings_update_sr_model
        4. Other Specifics: settings_check_args, settings_update_gpu_info etc.
        """
        model_path = model_path.lower()
        current_model_index = SR_TYPE.AUTO

        # AUTO
        if 'auto' in model_path:
            current_model_index = SR_TYPE.AUTO
        # Render Model
        elif 'anime4k' in model_path:
            current_model_index = SR_TYPE.Anime4K
        # CUDA Model
        elif 'realcugan' in model_path:
            current_model_index = SR_TYPE.RealCUGAN
        elif 'ncnnrealesr' in model_path:
            current_model_index = SR_TYPE.NcnnRealESR
        elif 'realesr' in model_path:
            current_model_index = SR_TYPE.RealESR
        elif 'waifucuda' in model_path:
            current_model_index = SR_TYPE.WaifuCUDA
        # NCNN Model
        elif 'ncnncugan' in model_path:
            current_model_index = SR_TYPE.NcnnCUGAN
        elif 'waifu2x' in model_path:
            current_model_index = SR_TYPE.Waifu2x
        elif 'basicvsrplusplusrestore' in model_path:
            current_model_index = SR_TYPE.BasicVSRPPR
        elif 'basicvsrplusplus' in model_path:
            current_model_index = SR_TYPE.BasicVSRPP
        elif 'realbasicvsr' in model_path:
            current_model_index = SR_TYPE.RealBasicVSR
        elif 'purebasicvsr' in model_path:
            current_model_index = SR_TYPE.PureBasicVSR
        elif 'tensorrt' in model_path:
            current_model_index = SR_TYPE.TensorRT
        return current_model_index


SR_NCNN = [SR_TYPE.NcnnCUGAN, SR_TYPE.Waifu2x, SR_TYPE.Anime4K, SR_TYPE.NcnnRealESR]
SR_MULTIPLE_INPUTS = [SR_TYPE.BasicVSRPP, SR_TYPE.RealBasicVSR, SR_TYPE.BasicVSRPPR, SR_TYPE.PureBasicVSR,
                      SR_TYPE.TensorRT]
SR_FILE_MODELS = [SR_TYPE.RealESR, SR_TYPE.WaifuCUDA, SR_TYPE.RealCUGAN, SR_TYPE.Anime4K, SR_TYPE.PureBasicVSR,
                  SR_TYPE.TensorRT]


class P5X_STATE(enum.Enum):
    INACTIVE = -1
    POST = 1
    BEFORE = 0


USER_PRIVACY_GRANT_PATH = os.path.join(appDir, "PrivacyStat.md")
