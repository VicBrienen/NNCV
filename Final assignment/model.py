import torch
from transformers import SegformerForSemanticSegmentation

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b5-finetuned-ade-640-640",
            num_labels=19,
            ignore_mismatched_sizes=True
        )

    def forward(self, pixel_values, **kwargs):
        logits = self.segformer(pixel_values).logits
        return torch.nn.functional.interpolate(
            logits,
            size=pixel_values.shape[2:], 
            mode="bilinear",
            align_corners=False,
        )