from builder_moe import build_vision_tower

import torch

class vision_tower_config:
    def __init__(
        self,
        mm_vision_tower,
        mm_vision_select_layer
    ):
        self.mm_vision_tower = mm_vision_tower
        self.mm_vision_select_layer = mm_vision_select_layer



vision_tower_cfg = vision_tower_config(mm_vision_tower = "openai/clip-vit-large-patch14-336",
                                       mm_vision_select_layer = -2
                                       )

CLIP_VisionTower = build_vision_tower(vision_tower_cfg)

x = torch.rand([5, 3, 336, 336])

output = CLIP_VisionTower(x)

print("end debug")