import torchvision.transforms as T
from PIL import Image
from visual_modeling import EVAVisionTransformer
from visual_config import EVAVisionConfig
import torch
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import numpy as np

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
    parser.add_argument('--similarity_threshold', type=float, default=0.18,
                        help='Similarity threshold for filtering (default: 0.18)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing (default: 32)')
    parser.add_argument('--image_dir', type=str, default='../images',
                        help='Directory containing images (default: ../images)')
    return parser.parse_args()

def setup_model_and_transform():
    # Model setup
    vision_config = EVAVisionConfig()
    model = EVAVisionTransformer(
        img_size=vision_config.image_size,
        patch_size=vision_config.patch_size,
        embed_dim=vision_config.width,
        depth=vision_config.layers,
        num_heads=vision_config.width // vision_config.head_width,
        mlp_ratio=vision_config.mlp_ratio,
        qkv_bias=vision_config.qkv_bias,
        drop_path_rate=vision_config.drop_path_rate,
        xattn=vision_config.xattn,
        postnorm=vision_config.postnorm,
    )

    with open("visual_model.pth", "rb") as f:
        state_dict = torch.load(f, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.eval().to(torch.float16).to("cuda")
    print("Model loaded successfully")
    # Transform setup
    OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
    OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
    image_size: int = 448
    transform = T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD),
    ])

    return model, transform

def setup_reference_embedding(model, transform, reference_image_path):
    print(f"Loading reference image: {reference_image_path}")
    image = Image.open(reference_image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = transform(image).unsqueeze(0).to("cuda").to(torch.float16)
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
            image = self.transform(image)
        return image, identifier

def batch_inference(model, csv_path: str, image_dir: str, output_dir: str, filtered_csv_dir: str, 
                   similarity_threshold: float, batch_size: int = 32):
    # Create output directories if they don't exist
    print(f"Threshold: {similarity_threshold}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(filtered_csv_dir, exist_ok=True)
    
    # Read CSV file
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    if 'identifier' not in df.columns:
        raise ValueError("CSV must contain an 'identifier' column")
    
    print(f"Found {len(df)} images to process")
    
    # Setup transform
    OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
    OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
    image_size: int = 448
    transform = T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD),
    ])
    
    # Create dataset and dataloader
    dataset = ImageBatchDataset(df, image_dir, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    
    # Perform batch inference
    device = next(model.parameters()).device
    print(f"Processing images in batches of {batch_size}")
    
    # Lists to store filtered data
    filtered_identifiers = []
    filtered_similarities = []
    
    filtered_out = []
    with torch.no_grad():
        for batch_images, batch_identifiers in tqdm(dataloader):
            batch_images = batch_images.to(device).to(torch.float16)
            features, similarities = model.encode_image_w_similarity(batch_images)
            features = features.cpu().numpy()
            similarities = similarities.cpu().numpy()

            # Process each image in the batch
            for i, (identifier, similarity) in enumerate(zip(batch_identifiers, similarities)):
                
                # Only save features and add to filtered list if similarity is below threshold
                if similarity < similarity_threshold:
                    output_path = os.path.join(output_dir, f"{identifier}.npy")
                    np.save(output_path, features[i])
                    filtered_identifiers.append(identifier)
                    filtered_similarities.append(similarity)
                else:
                    filtered_out.append(identifier)
    
    # Create and save filtered DataFrame
    filtered_df = pd.DataFrame({
        'identifier': filtered_identifiers,
        'similarity': filtered_similarities
    })
    
    # Merge with original DataFrame to keep all columns
    filtered_df = df[df['identifier'].isin(filtered_identifiers)].merge(
        filtered_df, on='identifier'
    )
    
    # Save filtered CSV
    csv_basename = os.path.splitext(os.path.basename(csv_path))[0]
    filtered_csv_path = os.path.join(filtered_csv_dir, f"{csv_basename}_filtered.csv")
    filtered_df.to_csv(filtered_csv_path, index=False)
    
    print(f"Processed {len(df)} images")
    print(f"Kept {len(filtered_df)} images with similarity < {similarity_threshold}")
    print(f"Filtered CSV saved to: {filtered_csv_path}")
    print(f"Encoded features saved to: {output_dir}")
    
    with open("../reports/filtered_out.txt", "w") as f:
        for item in filtered_out:
            f.write("%s\n" % item)

def main():
    args = parse_args()
    
    # Validate input paths
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input CSV file not found: {args.input_path}")
    
    if not os.path.exists(args.image_dir):
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
        
    if not os.path.exists(args.reference_image):
        raise FileNotFoundError(f"Reference image not found: {args.reference_image}")
    
    if not os.path.exists("visual_model.pth"):
        raise FileNotFoundError("Visual model not found: visual_model.pth")
    
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    if not os.path.exists(args.filtered_csv_dir):
        os.mkdir(args.filtered_csv_dir)
    
    # Setup model and transform
    model, transform = setup_model_and_transform()
    
    # Setup reference embedding
    setup_reference_embedding(model, transform, args.reference_image)
    
    # Run batch inference
    batch_inference(
        model=model,
        csv_path=args.input_path,
        image_dir=args.image_dir,
        output_dir=args.output_path,
        filtered_csv_dir=args.filtered_csv_dir,
        similarity_threshold=args.similarity_threshold,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()