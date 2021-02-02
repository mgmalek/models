import torch
from typing import List, Union, Optional


def intersection(box: torch.Tensor, other_boxes: torch.Tensor) -> torch.Tensor:
    """Find the area of the intersection between `box` and each box in `other_boxes`"""
    stacked = torch.stack([box.expand_as(other_boxes).float(), other_boxes.float()])
    
    xmins = torch.max(stacked[:, :, 0], dim=0).values
    ymins = torch.max(stacked[:, :, 1], dim=0).values
    xmaxs = torch.min(stacked[:, :, 2], dim=0).values
    ymaxs = torch.min(stacked[:, :, 3], dim=0).values

    widths  = torch.clamp(xmaxs - xmins, 0)
    heights = torch.clamp(ymaxs - ymins, 0)
    areas = widths * heights

    return areas


def jaccard_overlap(box: torch.Tensor, other_boxes: torch.Tensor) -> torch.Tensor:
    """Calculate the jaccard overlap between the ground truth box `box`
    and the set of SSD default boxes `other_boxes`"""
    # Calculate Intersections
    intersection_areas = intersection(box, other_boxes)

    # Calculate Unions
    box_area = float((box[2] - box[0]) * (box[3] - box[1]))
    other_box_widths  = (other_boxes[:, 2] - other_boxes[:, 0]).float()
    other_box_heights = (other_boxes[:, 3] - other_boxes[:, 1]).float()
    other_box_areas = other_box_widths * other_box_heights
    union_areas = box_area + other_box_areas - intersection_areas

    # Calculate Jaccard Overlap
    jacc_overlaps = intersection_areas / union_areas
    return jacc_overlaps


def cwh_to_xy(boxes_cwh: torch.Tensor) -> torch.Tensor:
    """Convert a tensor of bounding boxes from the format (cx, cy, w, h)
    to the format (xmin, ymin, xmax, ymax)"""
    cx, cy, w, h = boxes_cwh[:, 0], boxes_cwh[:, 1], boxes_cwh[:, 2], boxes_cwh[:, 3]
    
    boxes_xy = boxes_cwh.detach().clone()
    boxes_xy[:, 0] = cx - w/2
    boxes_xy[:, 1] = cy - h/2
    boxes_xy[:, 2] = cx + w/2
    boxes_xy[:, 3] = cy + h/2
    
    return boxes_xy


def offsets2xy(default_boxes: torch.Tensor, offsets: torch.Tensor,
               row_idxs: Optional[Union[List[int], int]] = None) -> torch.Tensor:
    """Convert bounding box offsets into bounding boxes of the format
    [xmin, ymin, xmax, ymax]."""
    
    if row_idxs is None:
        # Assume we want to use all default boxes
        boxes_to_use = default_boxes
    elif isinstance(row_idxs, int):
        boxes_to_use = default_boxes[row_idxs][None]
        offsets = offsets[None]
    else:
        boxes_to_use = default_boxes[row_idxs]
    
    assert (offsets.shape == boxes_to_use.shape,
            f"offsets has shape {offsets.shape} while boxes_to_use has shape {boxes_to_use.shape}")
    
    cx_default, cy_default, w_default, h_default = boxes_to_use.split(1, dim=1)
    cx_offset, cy_offset, w_offset, h_offset = offsets.split(1, dim=1)

    cx_actual = cx_offset * w_default / 10 + cx_default
    cy_actual = cy_offset * h_default / 10 + cy_default
    w_actual = torch.exp(w_offset / 5) * w_default
    h_actual = torch.exp(h_offset / 5) * h_default

    xmins = (cx_actual - w_actual / 2).flatten()
    ymins = (cy_actual - h_actual / 2).flatten()
    xmaxs = (cx_actual + w_actual / 2).flatten()
    ymaxs = (cy_actual + h_actual / 2).flatten()
    
    return torch.stack([xmins, ymins, xmaxs, ymaxs], dim=1)


def tensor2boxes(default_boxes: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Parse a target (or inference output) tensor to extract the bounding
    box locations and classes it contains
    
    Although this function could be a method of BoxMatcher, it is located here
    for easier integration into a TorchScript inference workflow."""
    all_classes = t[:, :-4].argmax(dim=1)

    rows = all_classes.nonzero().flatten()
    if len(rows) == 0:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

    classes = all_classes[rows]
    confidences = t[rows, :-4].max(dim=1).values
    offsets = t[rows, -4:]

    bboxes = []
    for row_idx, box_offsets in zip(rows, offsets):
        box = offsets2xy(default_boxes, box_offsets, row_idx.item())
        bboxes.append(box)
    
    bboxes = torch.cat(bboxes, dim=0)
    bboxes.clamp_(min=0, max=1)
    
    return bboxes, classes, confidences


def overlaps(box: torch.Tensor, boxes: List[torch.Tensor], iou_threshold: float):
    """Determine whether the jaccard overlap (i.e. IoU) between `box` and any
    of `boxes` exceeds `iou_threshold`"""
    if not len(boxes):
        return False
    
    stacked_boxes = torch.stack(boxes)

    max_overlap = jaccard_overlap(box, stacked_boxes).max()
    if max_overlap > iou_threshold:
        return True
    
    return False


def filter_overlapping_boxes(boxes: torch.Tensor, iou_threshold: float = 0.9):
    if not len(boxes):
        return [], []

    filtered_boxes = []
    filtered_idxs = []
    for idx, box in enumerate(boxes):
        if not overlaps(box, filtered_boxes, iou_threshold):
            filtered_boxes.append(box)
            filtered_idxs.append(idx)

    filtered_boxes = torch.stack(filtered_boxes)

    return filtered_boxes, filtered_idxs
