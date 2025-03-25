import torch
from transformers import SegformerForSemanticSegmentation, SegformerConfig

class Model(SegformerForSemanticSegmentation):
    def __init__(self):
        config = SegformerConfig(
            num_labels=19,
            hidden_sizes=[64, 128, 320, 512],
            depths=[3, 6, 40, 3],
            num_heads=[1, 2, 5, 8],
        )
        super().__init__(config)

    def forward(self, pixel_values, **kwargs):
        outputs = super().forward(pixel_values, **kwargs)
        logits = outputs.logits
        # Upsample logits to match input resolution
        logits = torch.nn.functional.interpolate(
            logits,
            size=pixel_values.shape[2:],  # matches the height and width of the input
            mode="bilinear",
            align_corners=False,
        )
        return logits