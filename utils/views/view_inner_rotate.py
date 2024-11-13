import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode

from .view_base import BaseView

def get_circle_mask(img_size: int, r: int):
    # TODO: Implement get_circle_mask
    mid_point = (img_size // 2, img_size // 2)
    def is_interior(midpoint, x, r):
        return (mid_point[0] - x[0])**2 + (mid_point[1] - x[1])**2 <= r**2
    
    mask = torch.zeros((img_size, img_size))
    for i in range(img_size):
        for j in range(img_size):
            mask[i, j] = is_interior(mid_point, (i, j), r)
    
    return mask.float()
    
    

def inner_rotate_func_with_mask(im: torch.Tensor, mask: torch.Tensor, angle, interpolate=False):
    rotated_image = TF.rotate(im, angle, interpolation=InterpolationMode.BILINEAR, expand=False)
    # import pdb
    # pdb.set_trace()

    rotated_image = rotated_image * mask
    # print(rotated_image.shape)
    # print(im.shape)
    outside_circ = (1 - mask).unsqueeze(0).repeat(3, 1, 1)
    rotated_image = rotated_image + im * outside_circ
    return rotated_image
 
class InnerRotateView(BaseView):
    """
    Implements an "inner circle" view, where a circle inside the image spins
    but the border stays still. Inherits from `PermuteView`, which implements
    the `view` and `inverse_view` functions as permutations. We just make
    the correct permutation here, and implement the `make_frame` method
    for animation
    """

    def __init__(self, angle):
        """
        Make the correct "inner circle" permutations and pass it to the
        parent class constructor.
        """
        self.angle = angle
        self.stage_1_mask = get_circle_mask(64, 24)
        self.stage_2_mask = get_circle_mask(256, 96)

    def view(self, im, **kwargs):
        im_size = im.shape[-1]
        if im_size == 64:
            mask = self.stage_1_mask.to(im)
            self.stage_1_mask = mask
        elif im_size == 256:
            mask = self.stage_2_mask.to(im)
            self.stage_2_mask = mask

        inner_rotated = inner_rotate_func_with_mask(im, mask, -self.angle, interpolate=False)

        return inner_rotated

    def inverse_view(self, im, **kwargs):
        im_size = im.shape[-1]
        if im_size == 64:
            mask = self.stage_1_mask.to(im)
            self.stage_1_mask = mask
        elif im_size == 256:
            mask = self.stage_2_mask.to(im)
            self.stage_2_mask = mask

        inner_rotated = inner_rotate_func_with_mask(im, mask, self.angle, interpolate=False)

        return inner_rotated

