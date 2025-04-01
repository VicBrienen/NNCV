import random
from PIL import Image
from torchvision.transforms.v2 import Transform, Resize

class RandomScale(Transform):
    def __init__(self, scale_range=(0.5, 2.0)):
        super().__init__()
        self.min_scale, self.max_scale = scale_range
        
    def forward(self, img):
        scale = random.uniform(self.min_scale, self.max_scale)
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        return Resize(new_size, interpolation=Image.BILINEAR)(img)