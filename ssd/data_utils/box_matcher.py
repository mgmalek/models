import math
import torch
from typing import List
from .rect_utils import jaccard_overlap, cwh_to_xy


class BoxMatcher(object):
    
    def __init__(self, map_sizes: List[int], steps: List[float], scales: List[float],
                 aspect_ratios: List[int], threshold: float = 0.5):
        """Initialise an object to match ground truth bounding boxes to the
        default boxes in an SSD"""
        assert len(map_sizes) == len(aspect_ratios)

        self.map_sizes = map_sizes
        self.steps = steps
        self.scales = scales
        self.threshold = threshold
        self.aspect_ratios = aspect_ratios
        
        self.default_boxes = []
        
        self._generate_default_boxes()
        self.default_boxes = torch.clamp(self.default_boxes, min=0, max=1)
        self.default_boxes_xy = cwh_to_xy(self.default_boxes)
        
        num_boxes = self.default_boxes.size(0)
        num_cells = sum([sz**2 for sz in self.map_sizes])
        self.boxes_per_cell = num_boxes // num_cells
    
    @property
    def num_boxes(self) -> int:
        """Return the number of default boxes generated"""
        return len(self.default_boxes)
    
    def get_matching_box_idxs(self, annot_box: torch.Tensor) -> List[int]:
        """Get the indices of the default boxes which overlap sufficiently
        with `annot_box`"""
        jaccard_overlaps = jaccard_overlap(annot_box, self.default_boxes_xy)
        
        chosen_idxs = (jaccard_overlaps > self.threshold).nonzero(as_tuple=False)
        chosen_idxs = chosen_idxs.flatten().numpy().tolist()
        highest_idx = jaccard_overlaps.argmax().item()
        if jaccard_overlaps[highest_idx] > 0.0:
            chosen_idxs.append(highest_idx)
        
        chosen_idxs = list(set(chosen_idxs))

        assert len(chosen_idxs) > 0, "Every annotation must match at least one default box"

        matches = [ (idx, jaccard_overlaps[idx]) for idx in chosen_idxs ]
        matches.sort(key=lambda o: o[1], reverse=True)
        return matches
    
    def _generate_default_boxes(self):
        """Generate all default bounding boxes"""
        for map_idx, map_size in enumerate(self.map_sizes):
            for row in range(map_size):
                for col in range(map_size):
                    self._generate_boxes_for_cell(map_idx, row, col)
        
        self.default_boxes = torch.stack(self.default_boxes)
    
    def _generate_boxes_for_cell(self, map_idx: int, row: int, col: int):
        """Generate default bounding boxes for a single cell of a feature map"""
        # scale = self._get_scale(map_idx)
        scale = self.scales[map_idx]
        aspect_ratios = [1,]
        for aspect_ratio in self.aspect_ratios[map_idx]:
            aspect_ratios.append(aspect_ratio)
            aspect_ratios.append(1 / aspect_ratio)

        for aspect_ratio in aspect_ratios:
            width  = scale * math.sqrt(aspect_ratio)
            height = scale / math.sqrt(aspect_ratio)
            self._add_box(map_idx, row, col, width, height)
        
        next_scale = self.scales[map_idx + 1]
        width = height = math.sqrt(scale * next_scale)
        self._add_box(map_idx, row, col, width, height)
    
    def _add_box(self, map_idx: int, row: int, col: int, width: float, height: float):
        """Add the dimensions (and metadata) of a new default box to the
        existing list of default boxes"""
        # map_size = self.map_sizes[map_idx]
        step_size = self.steps[map_idx]
        center_x = (col + 0.5) / step_size
        center_y = (row + 0.5) / step_size
    
        box_data = torch.tensor([center_x, center_y, width, height])
        self.default_boxes.append(box_data)
