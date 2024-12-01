import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from typing import Union, List, Tuple

class ResizeAndPadTransform:
    def __init__(self, target_size=336, min_aspect_ratio=1/3):
        self.target_size = target_size
        self.min_aspect_ratio = min_aspect_ratio
    
    def process_single_image(self, img):
        # Get current dimensions
        w, h = img.size
        aspect_ratio = w / h
        
        # Check if aspect ratio is less than minimum (too tall)
        if aspect_ratio < self.min_aspect_ratio:
            new_h = int(w / self.min_aspect_ratio)
            excess_height = h - new_h
            crop_top = excess_height // 2
            crop_bottom = excess_height - crop_top
            img = TF.crop(img, top=crop_top, left=0, height=new_h, width=w)
            h = new_h
        
        # Get dimensions (which may have changed after cropping)
        w, h = img.size
        
        # Determine which dimension is longer
        longest_side = max(w, h)
        
        # Calculate scaling factor
        scale = self.target_size / longest_side
        
        # Calculate new dimensions maintaining aspect ratio
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        img = TF.resize(img, (new_h, new_w), interpolation=TF.InterpolationMode.BILINEAR)
        
        # Calculate padding
        pad_h = self.target_size - new_h
        pad_w = self.target_size - new_w
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # Pad image to square
        img = TF.pad(img, padding=[pad_left, pad_top, pad_right, pad_bottom], fill=0)
        
        return img
    
    def __call__(self, images: Union[Image.Image, List[Image.Image], Tuple[Image.Image, ...]]) -> Union[Image.Image, List[Image.Image]]:
        if isinstance(images, (list, tuple)):
            return [self.process_single_image(img) for img in images]
        return self.process_single_image(images)

class BatchProcessor:
    def __init__(self, target_size=512, min_aspect_ratio=1/3):
        self.CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
        self.CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
        self.resize_pad = ResizeAndPadTransform(target_size=target_size, min_aspect_ratio=min_aspect_ratio)
        
    def process_single(self, image):
        # Ensure image is PIL Image
        if not isinstance(image, Image.Image):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            else:
                raise TypeError(f"Expected PIL Image or ndarray, got {type(image)}")
        
        # Apply resize and pad
        image = self.resize_pad(image)
        
        # Convert to tensor
        tensor = TF.to_tensor(image)
        
        # Apply normalization
        tensor = TF.normalize(tensor, mean=self.CLIP_MEAN, std=self.CLIP_STD)
        
        return tensor
    
    def __call__(self, images):
        if isinstance(images, (list, tuple)):
            # Process batch of images
            tensors = [self.process_single(img) for img in images]
            # Stack into batch
            return torch.stack(tensors)
        else:
            # Process single image
            return self.process_single(images)

def get_preprocessing_pipeline(target_size=512, min_aspect_ratio=1/3):
    """
    Creates a complete preprocessing pipeline including resize, RGB conversion, 
    tensor conversion, and CLIP normalization.
    
    Args:
        target_size (int): Size of the target square image
        min_aspect_ratio (float): Minimum allowed aspect ratio before cropping
    
    Returns:
        BatchProcessor: Complete preprocessing pipeline that can handle both single images and batches
    """
    return BatchProcessor(target_size=target_size, min_aspect_ratio=min_aspect_ratio)