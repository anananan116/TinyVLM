import torchvision.transforms as T
from PIL import Image
from visual_modeling import CLIPModel
from transformers import CLIPProcessor
import torch
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import numpy as np
import transformers
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
transformers.logging.set_verbosity_error()

def parse_args():
    parser = argparse.ArgumentParser(description='Batch encode images using vision transformer with similarity filtering')
    parser.add_argument('--input_path', type=str, default="../cache/image_metadata_0.csv",
                        help='Path to input CSV file containing image identifiers')
    parser.add_argument('--output_path', type=str, default="../encoded",
                        help='Path to output directory for encoded features')
    parser.add_argument('--filtered_csv_dir', type=str, default="../filtered",
                        help='Directory to save filtered CSV file')
    parser.add_argument('--reference_image', type=str, default="../reference.jpg",
                        help='Path to reference image for similarity calculation')
    parser.add_argument('--similarity_threshold', type=float, default=0.3,
                        help='Similarity threshold for filtering (default: 0.3)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for processing (default: 64)')
    parser.add_argument('--image_dir', type=str, default='../images',
                        help='Directory containing images (default: ../images)')
    parser.add_argument('--pretrained_model', type=str, default='openai/clip-vit-large-patch14-336',)
    return parser.parse_args()

def setup_model_and_transform(pretrained_model):
    # Model setup
    model = CLIPModel.from_pretrained(pretrained_model, ignore_mismatched_sizes=True)
    model = model.eval().to(torch.float16).to("cuda")
    transform = CLIPProcessor.from_pretrained(pretrained_model).image_processor

    return model, transform

def setup_reference_embedding(model, transform, reference_image_path):
    print(f"Loading reference image: {reference_image_path}")
    image = Image.open(reference_image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = transform(image, return_tensors="pt")["pixel_values"].to("cuda").to(torch.float16)
    model.set_reference_embedding(image)
    print("Reference embedding set successfully")

class ImageBatchDataset(Dataset):
    def __init__(self, df, image_dir: str, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        identifier = self.df.iloc[idx]['identifier']
        image_path = os.path.join(self.image_dir, f"{identifier}.jpg")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image, return_tensors="pt")["pixel_values"][0].half()
        return image, identifier

class AsyncWriter:
    def __init__(self, max_workers=8):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._process_queue)
        self.thread.start()

    def _process_queue(self):
        while self.running or not self.queue.empty():
            try:
                task = self.queue.get(timeout=1)
                if task is not None:
                    path, data = task
                    np.save(path, data)
                self.queue.task_done()
            except queue.Empty:
                continue

    def write(self, path, data):
        self.queue.put((path, data))

    def stop(self):
        self.running = False
        self.thread.join()
        self.executor.shutdown()

def batch_inference(model, transform, csv_path: str, image_dir: str, output_dir: str, 
                   filtered_csv_dir: str, similarity_threshold: float, batch_size: int = 32):
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(filtered_csv_dir, exist_ok=True)
    
    # Read CSV file using memory mapping
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path, memory_map=True)
    if 'identifier' not in df.columns:
        raise ValueError("CSV must contain an 'identifier' column")
    
    print(f"Found {len(df)} images to process")
    
    # Create dataset and optimized dataloader
    dataset = ImageBatchDataset(df, image_dir, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,  # Adjust based on CPU cores
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    # Initialize async writer
    writer = AsyncWriter()
    
    # Pre-allocate lists with estimated capacity
    estimated_capacity = len(df) // 2  # Assuming roughly half will be filtered
    filtered_identifiers = []
    filtered_similarities = []
    filtered_out = []
    
    # Perform batch inference
    device = next(model.parameters()).device
    print(f"Processing images in batches of {batch_size}")
    
    with torch.no_grad():
        for batch_images, batch_identifiers in tqdm(dataloader):
            # Transfer batch to GPU
            batch_images = batch_images.to(device, non_blocking=True, memory_format=torch.channels_last)
            
            # Process batch
            features, similarities = model.encode_image_w_similarity(batch_images)
            
            # Process results in batches
            features_cpu = features.cpu()
            similarities_cpu = similarities.cpu()
            
            # Numpy conversion (single batch operation)
            features_np = features_cpu.numpy()
            similarities_np = similarities_cpu.numpy()
            
            # Process batch results
            for i, (identifier, similarity) in enumerate(zip(batch_identifiers, similarities_np)):
                if similarity < similarity_threshold:
                    output_path = os.path.join(output_dir, f"{identifier}.npy")
                    writer.write(output_path, features_np[i])
                    filtered_identifiers.append(identifier)
                    filtered_similarities.append(similarity)
                else:
                    filtered_out.append(identifier)
    
    # Stop the async writer
    writer.stop()
    
    # Create filtered DataFrame efficiently
    filtered_df = pd.DataFrame({
        'identifier': filtered_identifiers,
        'similarity': filtered_similarities
    })
    
    # Efficient merge using indexes
    df.set_index('identifier', inplace=True)
    filtered_df.set_index('identifier', inplace=True)
    filtered_df = df.join(filtered_df, how='inner').reset_index()
    
    # Save filtered CSV
    csv_basename = os.path.splitext(os.path.basename(csv_path))[0]
    filtered_csv_path = os.path.join(filtered_csv_dir, f"{csv_basename}_filtered.csv")
    filtered_df.to_csv(filtered_csv_path, index=False)
    
    # Batch write filtered out identifiers
    with open("../reports/filtered_out.txt", "w") as f:
        f.writelines(f"{item}\n" for item in filtered_out)
    
    print(f"Processed {len(df)} images")
    print(f"Kept {len(filtered_df)} images with similarity < {similarity_threshold}")
    print(f"Filtered CSV saved to: {filtered_csv_path}")
    print(f"Encoded features saved to: {output_dir}")

def main():
    args = parse_args()
    
    # Validate input paths
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input CSV file not found: {args.input_path}")
    
    if not os.path.exists(args.image_dir):
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
        
    if not os.path.exists(args.reference_image):
        raise FileNotFoundError(f"Reference image not found: {args.reference_image}")
    
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    if not os.path.exists(args.filtered_csv_dir):
        os.mkdir(args.filtered_csv_dir)
    
    # Setup model and transform
    model, transform = setup_model_and_transform(args.pretrained_model)
    
    # Setup reference embedding
    setup_reference_embedding(model, transform, args.reference_image)
    
    # Run batch inference
    batch_inference(
        model=model,
        transform = transform,
        csv_path=args.input_path,
        image_dir=args.image_dir,
        output_dir=args.output_path,
        filtered_csv_dir=args.filtered_csv_dir,
        similarity_threshold=args.similarity_threshold,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()