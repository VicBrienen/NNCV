import torch
from transformers import Mask2FormerForUniversalSegmentation

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-large-mapillary-vistas-semantic"
        )

    def forward(self, pixel_values, **kwargs):
        outputs = self.model(pixel_values=pixel_values)
        logits = outputs.logits
        return torch.nn.functional.interpolate(
            logits,
            size=pixel_values.shape[2:],
            mode="bilinear",
            align_corners=False,
        )