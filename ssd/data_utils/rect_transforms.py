import random
from copy import deepcopy
from typing import List, Dict, Tuple
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from .rect_utils import jaccard_overlap


IMAGENET_MEANS = [0.485, 0.456, 0.406]
IMAGENET_STDS  = [0.229, 0.224, 0.225]

IMAGENET_INV_MEANS = [-mu/sigma for mu, sigma in zip(IMAGENET_MEANS, IMAGENET_STDS)]
IMAGENET_INV_STDS  = [1/sigma for sigma in IMAGENET_STDS]

crop_tfm = transforms.RandomResizedCrop(0)
color_jitter = transforms.ColorJitter(0.125, 0.5, 0.5, 0.05)
normalize = transforms.Normalize(IMAGENET_MEANS, IMAGENET_STDS)
denormalize = transforms.Normalize(IMAGENET_INV_MEANS, IMAGENET_INV_STDS)


def get_cropped_annots(annotations: List[Dict], left: int, top: int) -> Tuple[torch.Tensor, List[Dict]]:
    """Adjust annotations based on the `left` and `top` crop parameters"""
    cropped_annots = []
    annotations = deepcopy(annotations)
    for annotation in annotations:
        xmin, ymin, xmax, ymax = annotation['bbox']

        xmin = xmin - left
        ymin = ymin - top
        xmax = xmax - left
        ymax = ymax - top

        annotation['bbox'] = [xmin, ymin, xmax, ymax]
        cropped_annots.append(annotation)
    
    return cropped_annots


def filter_annotations(width: int, height: int, annotations: List[Dict]) -> List[Dict]:
    """Filter out annotations where the center is not in the frame
    (post-augmentation)"""
    filtered_annotations = []
    annotations = deepcopy(annotations)
    for annotation in annotations:
        xmin, ymin, xmax, ymax = annotation['bbox']
        xmid = (xmin + xmax) / 2
        ymid = (ymin + ymax) / 2

        if 0 < xmid < width and 0 < ymid < height:
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(xmax, width)
            ymax = min(ymax, height)
            annotation['bbox'] = [xmin, ymin, xmax, ymax]
            filtered_annotations.append(annotation)

    return filtered_annotations


def random_crop(img: torch.Tensor, original_annotations: List[Dict],
                min_scale: float = 0.1, max_scale: float = 1.0,
                min_ratio: float = 0.5, max_ratio: float = 2.0,
                max_iterations: int = 50) -> Tuple[torch.Tensor, List[Dict]]:
    """Randomly crop img and adjust bounding box annotations appropriately.
    There are three different kinds of crops, according to the SSD paper:
        1) Use the entire original input image
        2) Sample a patch so that the minimum Jaccard overlap with objects is
           0.1, 0.3, 0.5, 0.7 or 0.9
        3) Randomly sample patch
    """
    options = [None, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    option = random.choice(options)

    if option is None:
        return img, original_annotations

    highest_min_iou = float("-Inf")
    max_params = (0, 0, 0, 0)
    max_annotations = []
    for _ in range(max_iterations):
        top, left, height, width = crop_tfm.get_params(img, scale=(min_scale, max_scale),
                                                       ratio=(min_ratio, max_ratio))
        annotations = deepcopy(original_annotations)
        cropped_annotations = get_cropped_annots(annotations, left, top)
        if not len(cropped_annotations):
            continue

        cropped_img_tensor = torch.tensor([0, 0, width, height])
        cropped_bbox_tensors = torch.stack([
            torch.tensor(annot['bbox'])
            for annot in cropped_annotations
        ])
        min_overlap = jaccard_overlap(cropped_img_tensor, cropped_bbox_tensors).min()
        
        filtered_annotations = filter_annotations(width, height, cropped_annotations)

        if min_overlap > highest_min_iou:
            highest_min_iou = min_overlap
            max_params = (top, left, height, width)
            max_annotations = deepcopy(filtered_annotations)
        
        if min_overlap >= option:
            break
    
    top, left, height, width = max_params
    img = TF.crop(img, top, left, height, width)

    return img, max_annotations


def resize(img: torch.Tensor, annotations: List[Dict],
           resize_tfm: transforms.Resize) -> Tuple[torch.Tensor, List[Dict]]:
    """Resize img using resize_tfm and adjust bounding box annotations
    appropriately. Assumes that the resized shape is a square"""
    tfmd_height, tfmd_width = resize_tfm.size
    _, img_height, img_width = img.shape
    h_factor = img_height / tfmd_height
    w_factor = img_width / tfmd_width

    img = resize_tfm(img)
    for annotation in annotations:
        annotation['bbox'][0] /= w_factor
        annotation['bbox'][1] /= h_factor
        annotation['bbox'][2] /= w_factor
        annotation['bbox'][3] /= h_factor

    return img, annotations


def horizontal_flip(img: torch.Tensor, annotations: List[Dict]) -> Tuple[torch.Tensor, List[Dict]]:
    """Apply a horizontal flip to img with probability 0.5"""
    width = img.size(2)
    if random.random() < 0.5:
        img = TF.hflip(img)
        for annotation in annotations:
            xmin_new = width - annotation['bbox'][2]
            xmax_new = width - annotation['bbox'][0]
            annotation['bbox'][0] = xmin_new
            annotation['bbox'][2] = xmax_new 

    return img, annotations


def expand(img: torch.Tensor, annotations: List[Dict]) -> Tuple[torch.Tensor, List[Dict]]:
    """Apply the expansion augmentation described in section 3.6"""
    channels, height, width = img.shape
    top = int(np.random.uniform(3) * height)
    left = int(np.random.uniform(3) * width)

    expanded_img = torch.zeros(height*4, width*4, channels)
    expanded_img[:, :] = torch.tensor(IMAGENET_MEANS)
    expanded_img = expanded_img.permute(2, 0, 1)
    expanded_img[:, top:(top+height), left:(left+width)] = img

    for annotation in annotations:
        xmin, ymin, xmax, ymax = annotation['bbox']
        xmin += left
        ymin += top
        xmax += left
        ymax += top
        annotation['bbox'] = [xmin, ymin, xmax, ymax]
    
    return expanded_img, annotations
