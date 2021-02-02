import torch
import torch.nn as nn
from typing import Tuple
from .rect_utils import offsets2xy, jaccard_overlap


def nms(pred: torch.Tensor, default_boxes: torch.Tensor,
        conf_threshold: float = 0.01, iou_threshold: float = 0.45,
        max_preds: int = 200) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform non-maximum suppression on predicted bounding boxes
    Reference: https://github.com/amdegroot/ssd.pytorch"""
    boxes_out = []
    confs_out = []
    classes_out = []

    all_confs = pred[:, :-4].softmax(-1)

    all_boxes = offsets2xy(default_boxes, pred[:, -4:])

    for cls_idx, cls_confs in enumerate(all_confs.split(1, 1)):
        if cls_idx == 0:
            continue

        cls_confs = cls_confs.flatten()
        mask = cls_confs > conf_threshold

        if mask.sum() == 0:
            continue

        boxes = all_boxes[mask]
        confs = cls_confs[mask]

        conf_idxs = confs.argsort(descending=True)[:max_preds]
        boxes = boxes[conf_idxs]
        confs = confs[conf_idxs]

        candidate_idxs = []
        for idx, box in enumerate(boxes):
            max_overlap = 0.0
            if candidate_idxs:
                overlaps = jaccard_overlap(box, boxes[candidate_idxs])
                max_overlap = overlaps.max().item()

            if max_overlap < iou_threshold:
                candidate_idxs.append(idx)

        boxes_out.append(boxes[candidate_idxs])
        confs_out.append(confs[candidate_idxs])
        classes_out.extend([cls_idx for _ in candidate_idxs])
    
    if len(boxes_out):
        boxes_out = torch.cat(boxes_out, dim=0)
        confs_out = torch.cat(confs_out, dim=0)

    return boxes_out, classes_out, confs_out


def predict_image(model: nn.Module, img: torch.tensor,
                  default_boxes: torch.Tensor, conf_threshold: float = 0.50,
                  iou_threshold: float = 0.45) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict bounding boxes in an image and perform non-maximum suppression"""
    assert len(img.shape) == 3, "The `img` tensor must have exactly three axes."

    pred = model(img[None])[0]
    boxes, classes, confidences = nms(pred, default_boxes, conf_threshold, iou_threshold)
    
    return boxes, classes, confidences
