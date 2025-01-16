import os

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

from PIL import Image
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
import cv2

class RamdomLine(ImageOnlyTransform):
    """
    RamdomLine transformation
    """
    def __init__(self, p=0.5, max_weight = 5, num_lines = [0, 10], color_range = [0, 100]) -> None:
        super(RamdomLine, self).__init__()
        self.p = p
        self.max_weight = max_weight
        self.num_lines = num_lines
        self.color_range = color_range

    def apply(self, img, copy=True, max_weight = None, num_lines = None, color_range = None, **params):
        if np.random.uniform(0, 1) > self.p:
            return img
        if copy:
            img = img.copy()
        if max_weight is None:
            max_weight = self.max_weight
        if num_lines is None:
            num_lines = self.num_lines
        if color_range is None:
            color_range = self.color_range
        if isinstance(num_lines, (tuple, list)):
            num_lines = np.random.randint(num_lines[0], num_lines[1])
        if not isinstance(num_lines, int):
            raise ValueError(f"`num_lines` has unexpected type: {type(num_lines)}. Expected `int`, `tuple` or `list`.")
        
        shape = img.shape[:2]
        anchors = np.random.random(size = (num_lines, 2, 2)) * np.array(shape).reshape(1, 1, 2) # num_lines x <2 anchors per line> x <(x, y) of each anchor>
        weights = np.random.random(size = (num_lines)) * max_weight + 1
        colors = np.random.random(size = (num_lines, 3)) * (self.color_range[1] - self.color_range[0]) + self.color_range[0]
        
        anchors = anchors.astype(int)
        weights = weights.astype(int)
        colors = colors.astype(int).tolist()
        
        for (point1, point2), weight, color in zip(anchors, weights, colors):
            cv2.line(img, pt1=point1, pt2=point2, thickness=weight, color = tuple(color))
        return img   


def apply_transform(img: Image.Image) -> Image.Image:
    transforms = A.Compose([
    A.SafeRotate(p=0.2, limit=[-5, 5], border_mode = 0, fill=255), # Fill with constant https://docs.opencv.org/4.x/d2/de8/group__core__array.html
    RamdomLine(p=0.2),
    A.MotionBlur(p=0.2),
    A.SaltAndPepper(p=0.2),
    A.ColorJitter(p=0.2),
    A.ThinPlateSpline(p=0.2)
])
    new = transforms(image = np.array(img), force_apply=True)['image']
    return Image.fromarray(new).convert("RGB")

