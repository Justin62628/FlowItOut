import numpy as np
import torch

from Flow.RAFT.core.raft import RAFT
from Flow.RAFT.core.utils.utils import InputPadder
from flow_it_out_base import Inference

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
