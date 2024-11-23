from datasets import load_dataset
from io import BytesIO
from base64 import b64decode
from PIL import Image
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Manager, Pool, Lock
import numpy as np
import os

def process_image(args):
    """Process a single image with its counter value"""
    image_str, counter, output_dir = args
    try:
        image = Image.open(BytesIO(b64decode(image_str[0]))).convert("RGB")
        image_path = f"{output_dir}/{counter}.jpg"
        image.save(image_path)
        return image_path
    except Exception as e:
        print(f"Error processing image {counter}: {e}")
        return None

def process_dataset_chunk(args):
    """Process a chunk of images from a dataset"""
    chunk_data, start_counter, dataset_name, output_dir, chunk_id, progress_queue = args
    
    # Calculate total images for this chunk
    total_images = len(chunk_data)
    processed_images = 0
    
    # Process images in the chunk
    image_paths = []
    for idx, image_str in enumerate(chunk_data["image_base64_str"]):
        current_counter = start_counter + idx
        image_path = process_image((image_str, current_counter, output_dir))
        image_paths.append(image_path)
        processed_images += 1
        # Update progress through queue
        progress_queue.put((chunk_id, processed_images, total_images))
    
    # Prepare the chunk results
    chunk_df = pd.DataFrame(chunk_data)
    chunk_df = chunk_df.drop("image_base64_str", axis=1)
    chunk_df["image_path"] = image_paths
    
    return chunk_df

def progress_tracker(progress_queue, num_chunks, dataset_name):
    """Track progress across all processes"""
    progress_bars = {}
    
    with tqdm(total=100, desc=f"{dataset_name}", position=1, leave=True) as main_pbar:
        while True:
            try:
                chunk_id, processed, total = progress_queue.get()
                
                if chunk_id not in progress_bars:
                    progress_bars[chunk_id] = {
                        'processed': 0,
                        'total': total
                    }
                
                progress_bars[chunk_id]['processed'] = processed
                
                # Calculate overall progress
                total_processed = sum(bar['processed'] for bar in progress_bars.values())
                total_images = sum(bar['total'] for bar in progress_bars.values())
                overall_progress = (total_processed / total_images) * 100
                
                # Update main progress bar
                main_pbar.n = int(overall_progress)
                main_pbar.refresh()
                
                # Check if all chunks are complete
                if len(progress_bars) == num_chunks and \
                   all(bar['processed'] == bar['total'] for bar in progress_bars.values()):
                    break
                
            except Exception as e:
                print(f"Error in progress tracking: {e}")
                break

def process_single_dataset(args):
    """Process a complete dataset using multiple processes for image processing"""
    dataset_name, output_dir, global_counter, counter_lock, num_processes = args
    
    try:
        # Load the dataset
        print(f"\nProcessing dataset: {dataset_name}")
        dataset = load_dataset("MMInstruction/M3IT", dataset_name, trust_remote_code=True)
        train_set = pd.DataFrame(dataset["train"])
        
        # Calculate chunk sizes and prepare chunks
        total_images = len(train_set)
        chunk_size = total_images // num_processes
        chunks = []
        
        # Create a progress queue
        manager = Manager()
        progress_queue = manager.Queue()
        
        # Get current counter value and update it
        with counter_lock:
            start_counter = global_counter.value
            global_counter.value += total_images
        
        # Prepare chunks
        for i in range(num_processes):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_processes - 1 else total_images
            chunk = train_set.iloc[start_idx:end_idx]
            chunk_start_counter = start_counter + start_idx
            chunks.append((chunk, chunk_start_counter, dataset_name, output_dir, i, progress_queue))
        
        # Start progress tracker process
        tracker_process = mp.Process(
            target=progress_tracker,
            args=(progress_queue, num_processes, dataset_name)
        )
        tracker_process.start()
        
        # Process chunks in parallel
        with Pool(processes=num_processes) as pool:
            chunk_dfs = pool.map(process_dataset_chunk, chunks)
        
        # Wait for progress tracker to finish
        tracker_process.join()
        
        # Combine results
        result_df = pd.concat(chunk_dfs, ignore_index=True)
        return result_df
    
    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {e}")
        return pd.DataFrame()

def main():
    # Initialize multiprocessing manager and shared counter
    manager = Manager()
    global_counter = manager.Value('i', 0)
    counter_lock = manager.Lock()  # Create a separate lock for the counter
    
    # Configuration
    output_dir = "data/it_images"
    os.makedirs(output_dir, exist_ok=True)
    num_processes = mp.cpu_count()  # Number of processes for image processing
    print(f"Number of processes: {num_processes}")
    
    dataset_list = [
        "coco", "textcap", "image-paragraph-captioning", "coco-goi",
        "coco-text", "imagenet", "coco-itm", "snli-ve", "mocheg",
        "iqa", "vqa-v2", "shapes", "docvqa", "ocr-vqa", "st-vqa",
        "text-vqa", "gqa", "okvqa", "a-okvqa", "science-qa",
        "viquae", "clevr", "nlvr", "vcr", "visual-mrc",
        "winoground", "vist", "visual-dialog", "multi30k"
    ]
    
    # Overall progress bar for datasets
    print("\nStarting dataset processing...")
    with tqdm(total=len(dataset_list), desc="Overall Progress", position=0, leave=True) as pbar:
        result_dfs = []
        for dataset_name in dataset_list:
            dataset_df = process_single_dataset(
                (dataset_name, output_dir, global_counter, counter_lock, num_processes)
            )
            result_dfs.append(dataset_df)
            pbar.update(1)
    
    # Combine all results
    final_df = pd.concat(result_dfs, ignore_index=True)
    final_df.to_csv("data/it_train.csv", index=False)
    print("\nProcessing completed!")

if __name__ == "__main__":
    # Fix for Windows multiprocessing
    mp.freeze_support()
    main()