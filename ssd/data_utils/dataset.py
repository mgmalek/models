import json
import torch
import math
import random
from PIL import Image
from pathlib import Path
from copy import deepcopy
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from fastprogress import master_bar, progress_bar
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import matplotlib.patches as patches

from .box_matcher import BoxMatcher
from .rect_utils import jaccard_overlap, tensor2boxes, filter_overlapping_boxes
from .inference import nms
from .rect_transforms import (random_crop, resize, normalize, denormalize,
    horizontal_flip, color_jitter, expand)


class SSDDataset(Dataset):
    """This class defines the functionality required for a dataset to be used
    with the SSD object detection model implemented in this package. Other
    dataset classes should inherit from this class.

    Need to set .categories, .data and .num_classes

    Subclasses only need to implement the __init__ method, which should
    process raw data records in order to set the following three properties:
    - self.categories (a Dict mapping from category indices to category names.
      index 0 should be reserved for the 'background' category)
    - self.num_classes (the number of classes/categories in the dataset,
      including the background class)
    - self.data (a List[Dict] containing the file path of each image along
      with its annotations. The exact format for each element of this list is
      described below)
    
    The format of each element in the self.data list should be:
    {
        'path': Path("path/to/image/file"),
        'annotations': [
            {
                'bbox': [xmin, ymin, xmax, ymax],
                'category': category_idx,
            },
            ...
        ]
    }
    """
    
    def __init__(self, img_size: int, matcher: BoxMatcher,
                 is_train: bool = True, categories: Dict[int, str] = None):
        """Initialize an object detection dataset in the COCO format
        for use with an SSD model"""
        self.img_size = img_size
        self.matcher = matcher
        self.is_train = is_train
        
        if categories is None:
            self.cat2idx = { 'background': 0 }
            self.categories = { 0: 'background' }
        else:
            self.categories = categories
            self.cat2idx = { cat: idx for idx, cat in self.categories.items() }

        # Initialize Image Transforms
        self.to_tensor = transforms.ToTensor()
        self.resize_tfm = transforms.Resize((self.img_size, self.img_size))    

    def __len__(self) -> int:
        """Return the number of examples in the dataset"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve an example from the dataset by its index"""
        example = self.data[idx]
        
        img = Image.open(example['path'])
        if img.mode != 'RGB':
            img = img.convert('RGB')

        annotations = deepcopy(example['annotations'])
        
        img, annotations = self.transform(img, annotations)
        if not len(annotations):
            return self[random.randint(0, len(self) - 1)]

        tgt = self.generate_target(img, annotations)

        return img, tgt
    
    def transform(self, img: Image, annotations: List[Dict]) -> Tuple[torch.Tensor, List[Dict]]:
        """Convert img to tensor then augment img and annotations"""
        img = self.to_tensor(img)
        
        if self.is_train:
            img = color_jitter(img)
            img, annotations = horizontal_flip(img, annotations)
            if random.random() < 0.5:
                img, annotations = expand(img, annotations)
            img, annotations = random_crop(img, annotations)
            if not len(annotations):
                return img, []

        img, annotations = resize(img, annotations, self.resize_tfm)
        img = normalize(img)
        
        return img, annotations

    def generate_target(self, img: torch.Tensor, annotations: List[Dict]) -> torch.Tensor:
        """Generate the target tensor for a given example"""
        # Generate target tensor placeholder
        num_boxes = self.matcher.num_boxes
        num_tgt_vals = self.num_classes + 4
        tgt = torch.zeros(num_boxes, num_tgt_vals)
        tgt[:, 0] = 1  # Select the background class by default

        # Populate target tensor
        _, img_height, img_width = img.shape
        for annotation in annotations:
            xmin, ymin, xmax, ymax = annotation['bbox']
            xmin /= img_width
            xmax /= img_width
            ymin /= img_height
            ymax /= img_height

            # Find the default boxes that match the ground truth box
            annot_bbox = torch.tensor([xmin, ymin, xmax, ymax])
            matches = self.matcher.get_matching_box_idxs(annot_bbox)

            for idx, _ in matches:
                is_best_prior = (idx == 0)  # Since matches is sorted in order of decreasing IoU
                already_matched = (tgt[idx, 0] != 1)  # Prior has already been matched if it isn't the background class
                if already_matched and not is_best_prior:
                    continue

                cx_default, cy_default, w_default, h_default = self.matcher.default_boxes[idx]
                
                cx_ground = (xmin + xmax) / 2
                cy_ground = (ymin + ymax) / 2
                w_ground = xmax - xmin
                h_ground = ymax - ymin

                cx_offset = 10 * (cx_ground - cx_default) / w_default
                cy_offset = 10 * (cy_ground - cy_default) / h_default
                w_offset = 5 * math.log(w_ground / w_default)
                h_offset = 5 * math.log(h_ground / h_default)

                tgt[idx, -4:] = torch.tensor([cx_offset, cy_offset, w_offset, h_offset])

                # Fill out the target tensor based on the matched boxes
                tgt[idx, 0] = 0
                tgt[idx, annotation['category']] = 1

        return tgt
        
    def show_img(self, img: torch.Tensor, targ_data: Tuple[torch.Tensor, torch.Tensor] = None,
                 pred_data: Tuple[torch.Tensor, torch.Tensor] = None):
        """Display an image and with bounding boxes (optionally) overlaid"""
        _, ax = plt.subplots(1)
        img = denormalize(img)
        plt.imshow(img.cpu().numpy().transpose(1, 2, 0))
        plt.axis('off')

        _, img_height, img_width = img.shape

        if targ_data is not None and len(targ_data[0]):
            matched_boxes, targ_classes = targ_data
            matched_boxes = matched_boxes.detach().clone()
            matched_boxes[:, 0] *= img_width
            matched_boxes[:, 1] *= img_height
            matched_boxes[:, 2] *= img_width
            matched_boxes[:, 3] *= img_height
            self.plot_bboxes(ax, matched_boxes, targ_classes, color='g')
        
        if pred_data is not None and len(pred_data[0]):
            predicted_boxes, pred_classes = pred_data
            predicted_boxes = predicted_boxes.detach().clone()
            predicted_boxes[:, 0] *= img_width
            predicted_boxes[:, 1] *= img_height
            predicted_boxes[:, 2] *= img_width
            predicted_boxes[:, 3] *= img_height
            self.plot_bboxes(ax, predicted_boxes, pred_classes, color='y')
        
        plt.show()
    
    def plot_bboxes(self, ax: Axes, boxes: torch.Tensor, class_idxs: torch.Tensor, color: str):
        """Add bboxes to the plot tied to `ax`"""
        for box, class_idx in zip(boxes, class_idxs):
            xmin, ymin, xmax, ymax = box
            width = xmax - xmin
            height = ymax - ymin
            
            rect = patches.Rectangle((xmin, ymin), width, height, fill=False, color=color, lw=1.5)
            ax.add_patch(rect)
            
            if isinstance(class_idx, torch.Tensor):
                class_idx = class_idx.item()
            
            class_name = self.categories[class_idx] if class_idx else None
            
            ax.text(xmin, ymin, class_name,
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2})
    
    def calculate_map(self, model: nn.Module, conf_threshold: float = 0.01,
                      iou_threshold: float = 0.50, same_threshold: float = 0.45,
                      max_preds: float = 200, plot: bool = True,
                      use_gpu: bool = True) -> float:
        """Compute the mean average precision (mAP) achieved by a model on
        this dataset"""
        model.eval()

        dl = DataLoader(self, batch_size=32, drop_last=False, num_workers=32)

        # First, populate a dictionary with class ids as keys, and tuples of
        # the form (confidence, is_correct, bbox) for each bounding box
        # predicted for that class
        predictions = defaultdict(list)
        num_targs  = defaultdict(int) # Dict with number of ground truth boxes for each class

        mb = master_bar(dl)
        for imgs, targs in mb:
            if use_gpu: imgs = imgs.cuda()
            preds = model(imgs).cpu()
            for pred, targ in progress_bar(list(zip(preds, targs)), parent=mb):
                # Process Targets
                targ_boxes, targ_classes, _ = tensor2boxes(self.matcher.default_boxes, targ)
                targ_boxes, filtered_idxs = filter_overlapping_boxes(targ_boxes, iou_threshold=0.95)
                targ_classes = targ_classes[filtered_idxs]

                for targ_class in targ_classes:
                    num_targs[targ_class.item()] += 1

                # Process Predictions
                pred_boxes, pred_classes, pred_confs = nms(
                    pred,
                    self.matcher.default_boxes,
                    conf_threshold,
                    same_threshold,
                    max_preds
                )

                # Match Prediction to Targets for each Class
                matched_targs = set()
                for pred_box, pred_class, pred_conf in zip(pred_boxes, pred_classes, pred_confs):
                    # Indices of targets in the same class as the current prediction
                    same_classes = (targ_classes == pred_class).float()
                    same_class_idxs = set(same_classes.nonzero().flatten().numpy())

                    # Indices of targets that overlap sufficiently with the current prediction
                    overlaps = jaccard_overlap(pred_box, targ_boxes)
                    above_thresholds = (overlaps > iou_threshold).float()
                    above_threshold_idxs = set(above_thresholds.nonzero().flatten().numpy())

                    # Indices of targets that are both in the same class and overlap sufficiently
                    # with the current prediction
                    valid_idxs = same_class_idxs.intersection(above_threshold_idxs)
                    
                    # Target indices in order of decreasing overlap with the current prediction
                    valid_idxs = list(valid_idxs)
                    valid_idxs.sort(key=lambda idx: overlaps[idx], reverse=True)
                    valid_idxs = [
                        idx for idx in valid_idxs
                        if idx not in matched_targs
                    ]

                    pred_box_matched = False
                    if len(valid_idxs):
                        targ_idx = valid_idxs[0]
                        matched_targs.add(targ_idx)
                        pred_box_matched = True

                    pred_conf = pred_conf.item()
                    pred_box = pred_box.detach().cpu().numpy().tolist()
                    predictions[pred_class].append((pred_conf, pred_box_matched, pred_box))

        # Calculate Average Precision for each Class
        all_classes = set(num_targs.keys()).union(predictions.keys())

        avg_precisions = []
        for class_idx in all_classes:
            tps, fps, fns = 0, 0, num_targs[class_idx]
            if fns == 0:
                avg_precisions.append(1)
                continue

            precisions, recalls = [], []

            # Sort Predictions in order of decreasing confidence
            class_preds = predictions[class_idx]
            class_preds = [(conf, is_correct) for conf, is_correct, _ in class_preds]
            class_preds.sort(reverse=True) # Sort in order of decreasing confidence

            for _, is_correct in class_preds:
                if is_correct:
                    tps += 1
                    fns -= 1
                else:
                    fps += 1

                precision = tps / (tps + fps) if tps + fps > 0 else 0
                recall    = tps / (tps + fns) if tps + fns > 0 else 0
                
                if not (recalls and recalls[-1] == recall):
                    precisions.append(precision)    
                    recalls.append(recall)

            precisions_adj = [max(precisions[idx:]) for idx in range(len(precisions))]

            avg_precision = 0
            for idx, precision in enumerate(precisions_adj[:-1]):
                increment = recalls[idx+1] - recalls[idx]
                avg_precision += precision * increment
            
            print(f"\nAP for {self.categories[class_idx].capitalize()}: {round(avg_precision, 4)}")
            
            if plot:
                plt.plot(recalls, precisions_adj)
                plt.title(self.categories[class_idx].capitalize())
                plt.xlabel("Recall")
                plt.ylim(0, 1)
                plt.xlim(0, 1)
                plt.ylabel("Precision")
                plt.show()
            
            avg_precisions.append(avg_precision)

        mean_avg_precision = np.mean(avg_precisions)
        return mean_avg_precision


class COCODataset(SSDDataset):
    """A dataset class to be used for data in the COCO dataset format"""

    def __init__(self, ann_path: Path, img_dir: Path, img_size: int, matcher: BoxMatcher,
                 is_train: bool = True, categories: Dict[int, str] = None):
        super().__init__(img_size, matcher, is_train, categories)

        # Read Raw Data
        with open(ann_path, 'r') as f:
            annotation_data = json.load(f)

        imgs = annotation_data['images']
        annotations = annotation_data['annotations']
        categories = annotation_data['categories']

        for idx, cat in enumerate(categories):
            self.cat2idx[cat['id']] = idx + 1 # Add 1 to make for the 'background' class at idx=0
            self.categories[idx + 1] = cat['name']
        
        self.num_classes = max(self.categories.keys()) + 1  # Add 1 for the 'background' class
        
        # Extract Image Paths, Bounding Box Locations and Categories
        self.img_dict = {}
        for img in imgs:
            img_id = img['id']
            path = img_dir / img['file_name']
            self.img_dict[img_id] = {'path': path, 'annotations': []}

        for annotation in annotations:
            xmin, ymin, width, height = annotation['bbox']
            xmax = xmin + width
            ymax = ymin + height
            bbox = [xmin, ymin, xmax, ymax]
            
            img_id = annotation['image_id']
            category = self.cat2idx[annotation['category_id']]
            
            self.img_dict[img_id]['annotations'].append({
                "bbox": bbox,
                "category": category,
            })
        
        self.data = list(self.img_dict.values())


class VOCDataset(SSDDataset):
    """A dataset class to be used for data in the VOC dataset format"""

    def __init__(self, ann_fns: Path, img_dir: Path, img_size: int, matcher: BoxMatcher,
                 is_train: bool = True, categories: Dict[int, str] = None,
                 ignore_difficult = False):
        super().__init__(img_size, matcher, is_train, categories)

        self.data = []

        for ann_fn in ann_fns:
            root = ET.parse(ann_fn).getroot()

            img_fn = root.find('filename').text
            img_path = img_dir / img_fn

            img_data = {
                'path': img_path,
                'annotations': []
            }

            for obj in root.findall('object'):
                if ignore_difficult and obj.find('difficult').text != '0':
                    continue
                cat_name = obj.find('name').text
                cat_idx = self.cat2idx.get(cat_name)
                if cat_idx is None:
                    cat_idx = len(self.cat2idx)
                    self.cat2idx[cat_name] = cat_idx
                    self.categories[cat_idx] = cat_name

                bbox_obj = obj.find('bndbox')
                xmin = float(bbox_obj.find('xmin').text)
                ymin = float(bbox_obj.find('ymin').text)
                xmax = float(bbox_obj.find('xmax').text)
                ymax = float(bbox_obj.find('ymax').text)
                bbox = [xmin, ymin, xmax, ymax]

                img_data['annotations'].append({
                    'bbox': bbox,
                    'category': cat_idx,
                })
            
            self.data.append(img_data)
        
        self.num_classes = len(self.categories)
