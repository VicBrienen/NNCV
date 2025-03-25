import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

class Model(SegformerForSemanticSegmentation):
    def __init__(self):
        super().__init__.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            num_labels=19,
            ignore_mismatched_sizes=True,
        )

    def forward(self, pixel_values, **kwargs):
        outputs = super().forward(pixel_values, **kwargs)
        logits = outputs.logits
        return F.interpolate(
            logits,
            size=pixel_values.shape[2:], 
            mode="bilinear",
            align_corners=False,
        )