import numpy as np

from Flow.RAFT.core.utils import flow_viz


class Inference:
    """InferenceBase for Generating Flow

    For the following Inference classes, should implement the following method according to their needs.

    """
    def inference(self, image1: np.ndarray, image2: np.ndarray):
        """Inference Flow between image1 and image2

        Should at least implement this

        :param image1:
        :param image2:
        :return: Flow between image1 and image2
        """
        return image1

    def visualize(self, img, flo):
        """convert flow to RGB and Put img and flow together vertically for comparison

        Use FlowViz from RAFT

        :param img:
        :param flo:
        :return: image stacked by image and flow vertically, can be modified
        """
        img = img[0].permute(1, 2, 0).cpu().numpy()
        flo = flo[0].permute(1, 2, 0).cpu().numpy()

        # map flow to rgb image
        flo = flow_viz.flow_to_image(flo)
        img_flo = np.concatenate([img, flo], axis=0)
        return img_flo
