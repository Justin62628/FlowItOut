import numpy as np
import torch

from Flow.RAFT.core.raft import RAFT
from Flow.RAFT.core.utils.utils import InputPadder
from Flow.gmflow.gmflow import GMFlow
from flow_it_out_base import Inference
import torch.nn.functional as F


DEVICE = 'cuda'


class RaftInference(Inference):
    def __init__(self, args):
        # Initiation
        self.model = torch.nn.DataParallel(RAFT(args))
        self.model.load_state_dict(torch.load(args.model))

        self.model = self.model.module
        self.model.to(DEVICE)
        self.model.eval()

    def np2tensor(self, img):
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(DEVICE)

    @torch.no_grad()
    def inference(self, image1: np.ndarray, image2: np.ndarray):
        image1 = self.np2tensor(image1)
        image2 = self.np2tensor(image2)
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_up = self.model.forward(image1, image2, iters=10, test_mode=True)
        return self.visualize(image1, flow_up)


class GmfInference(Inference):
    def __init__(self, args):
        # Initiation
        self.model = GMFlow(num_scales=2, upsample_factor=4)

        self.model.load_state_dict(torch.load(args.model))

        self.model.to(DEVICE)
        self.model.eval()

    def _generate_padding(self, img, scale: float):
        """

        :param scale:
        :param img: cv2.imread [:, :, ::-1]
        :return:
        """
        h, w, _ = img.shape
        # if '_union' in self.model_path:  # debug
        #     scale /= 2.0
        tmp = max(32, int(32 / scale))
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        padding = (0, pw, 0, ph)
        return padding, h, w

    def _pad_image(self, img, padding):
        # TEST use scale
        _, pw, _, ph = padding
        img = F.interpolate(img, (ph, pw), mode='bilinear', align_corners=False)
        return img

    def _generate_torch_img(self, img, padding):
        """
        :param img: cv2.imread [:, :, ::-1]
        :param padding:
        :return:
        """

        """
        Multi Cards Optimization:
        OLS: send several imgs pair according to device_count (2 to be specific)
        HERE: Concat [i1, i2] [i3, i4] and send to rife
        """

        # try:
        #     img = Tools.get_u1_from_u2_img(img)
        # except ValueError:
        if img.dtype != np.uint8:
            img = img.astype(np.float32)
        img_torch = torch.from_numpy(img).to(DEVICE).float()
        img_torch = img_torch.permute(2, 0, 1).unsqueeze(0)
        IMG_SIZE_MAX = 255. if img_torch.max() <= 255. else 65535.
        img_torch = img_torch / IMG_SIZE_MAX

        return self._pad_image(img_torch, padding)

    @torch.no_grad()
    def inference(self, image1: np.ndarray, image2: np.ndarray):
        scale = 0.5
        padding, h, w = self._generate_padding(image1, scale=scale)
        image1 = self._generate_torch_img(image1, padding)
        image2 = self._generate_torch_img(image2, padding)
        # For GMFSS, scale 0.5 is a must
        image1_f = F.interpolate(image1, scale_factor=scale, mode="bilinear", align_corners=False)
        image2_f = F.interpolate(image2, scale_factor=scale, mode="bilinear", align_corners=False)

        flow_low = self.model(image1_f, image2_f)

        image1 = F.interpolate(image1, (h, w), mode='bilinear', align_corners=False)
        flow_low = F.interpolate(flow_low, (h, w), mode='bilinear', align_corners=False)

        # flow_up = self.model(image2, image1)
        return self.visualize(image1 * 255., flow_low)
