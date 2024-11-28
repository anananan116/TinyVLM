import pandas as pd
import numpy as np
from PIL import Image
import random
from tqdm import tqdm
import concurrent.futures
import os
from transformers import AutoProcessor

test_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

def check_image(args):
    """
    Helper function to check if an image is valid
    
    Args:
        args: tuple of (index, image_path)
        
    Returns:
        index if image is valid, None otherwise
    """
    try:
        idx, image_path = args
        if os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
            image = test_processor(image, return_tensors="pt")["pixel_values"]
            return idx
        else:
            return None
    except:
        return None

def process_dataframe(df_path, output_path, num_workers=None):
    """
    Process dataframe by:
    1. Handling NaN inputs with 50% probability of moving instruction to input
    2. Validating image paths using multi-threading
    
    Args:
        df_path (str): Path to the input dataframe
        output_path (str): Path to save the processed dataframe
        num_workers (int, optional): Number of worker threads. Defaults to None (CPU count)
    """
    # Set number of workers if not specified
    if num_workers is None:
        num_workers = os.cpu_count()
    
    # Read the dataframe
    print("Reading dataframe...")
    df = pd.read_csv(df_path)
    total_rows = len(df)
    
    print("\nProcessing inputs...")
    for idx in tqdm(df.index, desc="Handling NaN values"):
        if random.random() < 0.33:
            df.loc[idx, 'inputs'] = df.loc[idx, 'inputs'] + df.loc[idx, 'instruction']
            df.loc[idx, 'instruction'] = ''
        elif random.random() < 0.66:
            df.loc[idx, 'inputs'] = df.loc[idx, 'instruction'] + df.loc[idx, 'inputs']
            df.loc[idx, 'instruction'] = ''
        else:
            pass
    
    # Prepare arguments for parallel processing
    check_args = list(zip(df.index, df['image_path']))
    
    # Validate images using multiple threads
    print(f"\nValidating images using {num_workers} threads...")
    valid_indices = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks and wrap with tqdm for progress
        futures = list(tqdm(
            executor.map(check_image, check_args),
            total=len(check_args),
            desc="Checking images"
        ))
        
        # Collect valid indices
        valid_indices = [idx for idx in futures if idx is not None]
    
    # Keep only rows with valid images
    df = df.loc[valid_indices]
    
    # Reset index
    df = df.reset_index(drop=True)
    
    # Print summary
    print("\nProcessing complete!")
    print(f"Initial rows: {total_rows}")
    print(f"Rows with valid images: {len(df)}")
    print(f"Removed rows: {total_rows - len(df)}")
    
    df.to_csv(output_path, index=False)
    print(f"\nProcessed dataframe saved to: {output_path}")

# Example usage
if __name__ == "__main__":
    input_path = 'data/it_train.csv'
    output_path = 'data/it_train_processed.csv'
    
    # Optional: specify number of worker threads
    num_workers = os.cpu_count()  # Use all available CPU cores
    
    try:
        process_dataframe(input_path, output_path, num_workers)
    except Exception as e:
        print(f"An error occurred: {str(e)}")