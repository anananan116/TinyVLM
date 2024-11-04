import os
import pandas as pd
import torch
from PIL import Image
from pathlib import Path
import open_clip
from tqdm import tqdm
from labels import CLIP_LABELS

class ImageClassifier:
    def __init__(self, label_set="natural"):
        """
        Initialize classifier with different label sets.
        
        Args:
            label_set (str): Which label set to use ("natural", "wukong", "laion", or "custom")
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14',
            pretrained='openai',
            device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer('ViT-L-14')
        
        self.labels = CLIP_LABELS
        text_tokens = self.tokenizer([self.labels[label] for label in self.labels.keys()]).to(self.device)
        self.text_features = self.model.encode_text(text_tokens)
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        
        
        
    def classify_image(self, image_path):
        """Classify a single image"""
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        text_tokens = self.tokenizer([CLIP_LABELS[label] for label in CLIP_LABELS.keys()]).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
            
            confidences, indices = similarity[0].topk(3)
        
        label_names = list(CLIP_LABELS.keys())
        results = [
            {
                'label': label_names[idx],
                'description': CLIP_LABELS[label_names[idx]],
                'confidence': conf.item()
            }
            for idx, conf in zip(indices, confidences)
        ]
        
        return results
    
    def process_directory(self, directory_path, batch_size=32):
        """
        Classify all jpg images in a directory using batched inference.
        
        Args:
            directory_path (str): Path to directory containing images
            batch_size (int): Number of images to process simultaneously
            
        Returns:
            dict: Dictionary mapping image filenames to their classification results
        """
        image_paths = []
        for ext in ('*.jpg', '*.jpeg', '*.JPG', '*.JPEG'):
            image_paths.extend(Path(directory_path).glob(ext))
        
        if not image_paths:
            return {}
        
        results = {}
        
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            for img_path in batch_paths:
                try:
                    image = Image.open(img_path).convert('RGB')
                    image_input = self.preprocess(image)
                    batch_images.append(image_input)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
            
            if not batch_images:
                continue
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(batch_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
                confidences, indices = similarity.topk(1, dim=-1)
            label_names = list(self.labels.keys())
            
            for idx, (img_path, conf_tensor, idx_tensor) in enumerate(zip(batch_paths, confidences, indices)):
                results[img_path.name] = {
                    'label': label_names[idx_tensor[0].item()],
                    'description': self.labels[label_names[idx_tensor[0].item()]],
                    'confidence': conf_tensor[0].item()
                }
        
        return results