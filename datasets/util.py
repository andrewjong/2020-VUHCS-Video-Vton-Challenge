import numpy as np
import torch
from PIL import Image
import torchvision
import math
import random

def segment_cloths_from_image(im, _parse_array):
    """
    from the original image, uses the cloth segmentation map to isolate (crop) the
    cloth-only parts from the image.
    """
    # ISOLATE CLOTH. cloth labels, combines into a 1d binary mask
    _parse_cloth = (
        (_parse_array == 5).astype(np.float32)
        + (_parse_array == 6).astype(np.float32)
        + (_parse_array == 7).astype(np.float32)
    )
    _parse_cloth_mask = torch.from_numpy(_parse_cloth)  # [0,1]
    # upper cloth, segment it from the body
    image_cloth_only = im * _parse_cloth_mask + (1 - _parse_cloth_mask)
    # [-1,1], fill 1 for other parts
    return image_cloth_only

def transformation(img):

    # input is going to be
    """


    Args:
        img:
            image of the person (im)
            head of the person (im_head)
            ground truth cloth segmented from the og image (im_cloth; note this is different from cloth/cloth_mask)
            silhouette of the person
            pose_map and im_pose in the get_input_person_pose() function. CAREFUL: this is not simply scaling up and rotating the HxW pose tensors; this would undesirably change the size and shape of the heatmap points. We want consistent heatmaps regardless of augmentation to avoid spurious signals.
            agnostic (which is just formed by concatting im_head+silhouette+pose_map)

    Returns:

    """
    im = img['image']
    im_head = img['head']
    im_cloth = img['parse_cloth']
    silhouette = img['shape']
    pose = img['im_pose']
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        lambda img: img.rotate(random.choice(list(range(-90, 90)))),
        torchvision.transforms.ColorJitter(brightness=(1,2), contrast=(1, 2))
        torchvision.transforms.ToTensor()

    ])
    result = {
        "c_name": img["c_name"],  # for visualization
        "im_name": img["im_name"],  # for visualization or ground truth
        "cloth": cloth,  # for input
        "cloth_mask": cloth_mask,  # for input
        "image": transforms(im),  # for visualization
        "agnostic": agnostic,  # for input
        "parse_cloth": im_cloth,  # for ground truth
        "shape": transforms(silhouette),  # for visualization
        "head": transforms(im_head),  # for visualization
        "pose_image": im_pose,  # for visualization
        "grid_image": im_grid,  # for visualization
    }
    return img

def ScaleRotateTranslate(image, angle, center=None, new_center=None,
                         scale=None, expand=False):
    '''
    taken from https://www.programcreek.com/python/example/69877/PIL.Image.AFFINE
    '''
    if center is None:
        return image.rotate(angle, expand)
    angle = -angle / 180.0 * math.pi
    nx, ny = x, y = center
    if new_center != center:
        (nx, ny) = new_center
    sx = sy = 1.0
    if scale:
        (sx, sy) = scale
    cosine = math.cos(angle)
    sine = math.sin(angle)
    a = cosine / sx
    b = sine / sx
    c = x - nx * a - ny * b
    d = -sine / sy
    e = cosine / sy
    f = y - nx * d - ny * e
    return image.transform(image.size, Image.AFFINE,
                           (a, b, c, d, e, f), resample=Image.BICUBIC)