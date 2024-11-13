from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from .view_base import BaseView


class Rotate90CWView(BaseView):
    def __init__(self):
        pass

    def view(self, im, background=None, **kwargs):
        # Rotate the image 90 degrees clockwise
        return TF.rotate(im, -90, expand=True, interpolation=InterpolationMode.NEAREST)

    def inverse_view(self, noise, background=None, **kwargs):
        # Rotate the image 90 degrees counterclockwise
        return TF.rotate(
            noise, 90, expand=True, interpolation=InterpolationMode.NEAREST
        )
