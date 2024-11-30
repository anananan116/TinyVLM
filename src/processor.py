import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

class ResizeAndPadTransform:
    def __init__(self, target_size=512, min_aspect_ratio=1/3):
        self.target_size = target_size
        self.min_aspect_ratio = min_aspect_ratio
    
    def __call__(self, img):
        # Get current dimensions
        w, h = img.size
        aspect_ratio = w / h
        
        # Check if aspect ratio is less than minimum (too tall)
        if aspect_ratio < self.min_aspect_ratio:
            # Calculate new height to achieve minimum aspect ratio
            new_h = int(w / self.min_aspect_ratio)
            # Calculate how much height to remove
            excess_height = h - new_h
            # Crop from top and bottom equally
            crop_top = excess_height // 2
            crop_bottom = excess_height - crop_top
            # Perform the crop
            img = TF.crop(img, top=crop_top, left=0, height=new_h, width=w)
            # Update height after crop
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
        
        # Calculate padding on each side
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # Pad image to square
        img = TF.pad(img, padding=[pad_left, pad_top, pad_right, pad_bottom], fill=0)
        
        return img

def get_preprocessing_pipeline(target_size=512, min_aspect_ratio=1/3):
    """
    Creates a complete preprocessing pipeline including resize, RGB conversion, 
    tensor conversion, and CLIP normalization.
    
    Args:
        target_size (int): Size of the target square image
        min_aspect_ratio (float): Minimum allowed aspect ratio before cropping
    
    Returns:
        torchvision.transforms.Compose: Complete preprocessing pipeline
    """
    # CLIP mean and std values
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
    
    return T.Compose([
        ResizeAndPadTransform(target_size=target_size, min_aspect_ratio=min_aspect_ratio),
        T.Lambda(lambda x: x.convert('RGB')),  # Convert to RGB
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
    ])