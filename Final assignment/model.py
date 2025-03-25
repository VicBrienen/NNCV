import torch
from transformers import SegformerForSemanticSegmentation, SegformerConfig

class Model(SegformerForSemanticSegmentation):
    def __init__(self):
        config = SegformerConfig(
            num_labels=19,
            hidden_sizes=[32, 64, 160, 256],
            depths=[2, 2, 2, 2],
            num_heads=[1, 2, 5, 8],
        )
        super().__init__(config)

    def forward(self, pixel_values, **kwargs):
        outputs = super().forward(pixel_values, **kwargs)
        return outputs.logits