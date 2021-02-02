import torch
import torch.nn as nn
from typing import List, Dict


class SSD(nn.Module):
    
    def __init__(self, body: nn.Module, map_data: Dict[int, Dict[str, int]],
                 boxes_per_cell: List[int], num_classes: int):
        super().__init__()
        
        self.body = body
        self.map_data = map_data
        self.boxes_per_cell = boxes_per_cell
        self.num_classes = num_classes
        
        self.vals_per_box: int = self.num_classes + 4
        self.feature_maps: List[torch.Tensor] = []
        
        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()
        
        map_data_vals = list(self.map_data.values())
        map_data_vals.sort(key=lambda o: o['layer_idx'])
        for data, num_boxes in zip(map_data_vals, self.boxes_per_cell):
            # Initialize conv layers used to predict classes and offsets
            in_channels = data["num_channels"]

            conf_layer = nn.Conv2d(in_channels, self.num_classes * num_boxes, kernel_size=3, stride=1, padding=1)
            self.conf_layers.append(conf_layer)

            loc_layer = nn.Conv2d(in_channels, 4 * num_boxes, kernel_size=3, stride=1, padding=1)
            self.loc_layers.append(loc_layer)
        
            # Register forward hooks to save the activations that will be used
            # to predict classes and offsets
            layer_idx = data["layer_idx"]
            self.body[layer_idx].register_forward_hook(self._save_activation)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.feature_maps = []
        
        # Perform feature extraction
        self.body(x)
        assert(len(self.feature_maps) == len(self.map_data))

        # Generate class and offset predictions based on extracted features
        outputs: List[torch.Tensor] = []
        for idx, (conf_layer, loc_layer) in enumerate(zip(self.conf_layers, self.loc_layers)):
            feature_map = self.feature_maps[idx]
            map_size = feature_map.size(2)
            
            conf_pred = conf_layer(feature_map)  # B, C, H, W
            conf_pred = conf_pred.permute(0, 2, 3, 1) # B, H, W, C
            conf_pred = conf_pred.reshape(-1, int(map_size**2 * self.boxes_per_cell[idx]), self.num_classes)
            
            loc_pred = loc_layer(feature_map)  # B, C, H, W
            loc_pred = loc_pred.permute(0, 2, 3, 1) # B, H, W, C
            loc_pred = loc_pred.reshape(-1, int(map_size**2 * self.boxes_per_cell[idx]), 4)

            pred = torch.cat([conf_pred, loc_pred], dim=2)
            outputs.append(pred)
        
        return torch.cat(outputs, dim=1)
    
    def _save_activation(self, m: nn.Module, inp: torch.Tensor, out: torch.Tensor):
        self.feature_maps.append(out)
