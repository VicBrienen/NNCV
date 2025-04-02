import torch
from transformers import Mask2FormerForUniversalSegmentation

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-large-mapillary-vistas-semantic",
            num_labels=19,
            ignore_mismatched_sizes=True
        )

    def forward(self, pixel_values, **kwargs):
        outputs = self.model(pixel_values=pixel_values)
        class_queries_logits = outputs.class_queries_logits
        return torch.nn.functional.interpolate(
            class_queries_logits,
            size=pixel_values.shape[2:],
            mode="bilinear",
            align_corners=False,
        )