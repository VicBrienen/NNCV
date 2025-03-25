import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

class Model(SegformerForSemanticSegmentation):
    def __init__(self):
        pretrained_model = "nvidia/segformer-b0-finetuned-ade-512-512"
        model = SegformerForSemanticSegmentation.from_pretrained(pretrained_model)
        
        model.config.num_labels = 19
        
        model.classifier = torch.nn.Conv2d(model.config.hidden_sizes[-1], 19, kernel_size=1)
        
        super().__init__(model.config)
        self.load_state_dict(model.state_dict(), strict=False)
    
    def forward(self, pixel_values, **kwargs):
        outputs = super().forward(pixel_values, **kwargs)
        logits = outputs.logits
        # Upsample logits to match input resolution
        logits = F.interpolate(
            logits,
            size=pixel_values.shape[2:],  # matches the height and width of the input
            mode="bilinear",
            align_corners=False,
        )
        return logits