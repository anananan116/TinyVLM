import requests
import os
import concurrent.futures
import pandas as pd
from PIL import Image
from io import BytesIO
import uuid
import time
import argparse
from datetime import datetime
from collections import defaultdict
import warnings
from queue import Queue
from threading import Lock
from tqdm import tqdm

# Suppress PIL warnings about palette images
warnings.filterwarnings('ignore', category=UserWarning, module='PIL.Image')
class ImageProcessor:
    def __init__(self, timeout=10, max_workers=4, resolution=224, pbar=None):
        self.timeout = timeout
        self.max_workers = max_workers
        self.resolution = resolution
        self.stats_lock = Lock()
        self.pbar = pbar
        self.reset_stats()
        
    def reset_stats(self):
        """Reset statistics for new chunk processing"""
        self.stats = {
            'download_times': [],
            'failed_downloads': [],
            'success_count': 0,
            'start_time': time.time()
        }

    def process_single_image(self, row):
        """Process a single image"""
        start_time = time.time()
        try:
            response = requests.get(row['image_url'], timeout=self.timeout)
            response.raise_for_status()
            
            img = Image.open(BytesIO(response.content))
            
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                img = img.convert('RGBA')
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'RGBA':
                    background.paste(img, mask=img.split()[3])
                else:
                    background.paste(img, mask=img.split()[1])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            img.thumbnail((self.resolution, self.resolution))
            new_img = Image.new('RGB', (self.resolution, self.resolution), (255, 255, 255))
            paste_x = (self.resolution - img.width) // 2
            paste_y = (self.resolution - img.height) // 2
            new_img.paste(img, (paste_x, paste_y))
            
            output_path = os.path.join("images", f"{row['identifier']}.jpg")
            new_img.save(output_path, "JPEG", quality=95)
            
            download_time = time.time() - start_time
            
            with self.stats_lock:
                self.stats['download_times'].append(download_time)
                self.stats['success_count'] += 1
                if self.pbar:
                    self.pbar.update(1)
                    self.pbar.set_postfix({
                        'Success': self.stats['success_count'],
                        'Failed': len(self.stats['failed_downloads'])
                    })
            
            return True, row['identifier']
            
        except Exception as e:
            with self.stats_lock:
                self.stats['failed_downloads'].append({
                    'url': row['image_url'],
                    'identifier': row['identifier'],
                    'error': str(e)
                })
                if self.pbar:
                    self.pbar.update(1)
                    self.pbar.set_postfix({
                        'Success': self.stats['success_count'],
                        'Failed': len(self.stats['failed_downloads'])
                    })
            return False, row['identifier']

def process_chunk(chunk_df, processor):
    """Process a single chunk of the dataframe"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=processor.max_workers) as executor:
        # Submit all tasks and store futures with their corresponding rows
        future_to_row = {
            executor.submit(processor.process_single_image, row): row 
            for _, row in chunk_df.iterrows()
        }
        
        # Process completed futures
        for future in concurrent.futures.as_completed(future_to_row):
            success, identifier = future.result()
            # We don't need to track the results by index anymore

def process_dataframe_in_chunks(df, chunk_size=100000, timeout=10, max_workers=4, resolution=224):
    """Process a large dataframe by breaking it into smaller chunks"""
    num_chunks = (len(df) + chunk_size - 1) // chunk_size
    
    # Initialize combined results
    result_dfs = []
    combined_stats = {
        'download_times': [],
        'failed_downloads': [],
        'success_count': 0,
        'start_time': time.time(),
        'end_time': None,
        'total_time': 0,
        'average_download_time': 0
    }
    
    # Create progress bar for chunks
    chunk_pbar = tqdm(total=num_chunks, desc="Processing chunks", unit="chunk", position=0)
    print(f"\nProcessing {len(df)} rows in {num_chunks} chunks of {chunk_size} rows each")
    
    # Initialize processor
    processor = ImageProcessor(timeout, max_workers, resolution)
    
    # Process each chunk
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df))
        chunk_df = df.iloc[start_idx:end_idx].copy()
        chunk_df['identifier'] = [str(uuid.uuid4()) for _ in range(len(chunk_df))]
        
        # Reset processor stats for new chunk
        processor.reset_stats()
        
        # Initialize progress bar for current chunk
        pbar = tqdm(total=len(chunk_df), desc=f"Chunk {i+1}/{num_chunks}", unit="img", position=1, leave=False)
        processor.pbar = pbar
        
        # Process the chunk
        process_chunk(chunk_df, processor)
        
        # Update combined statistics
        combined_stats['download_times'].extend(processor.stats['download_times'])
        combined_stats['failed_downloads'].extend(processor.stats['failed_downloads'])
        combined_stats['success_count'] += processor.stats['success_count']
        
        # Append results
        result_dfs.append(chunk_df[['image_url', 'capsfusion', 'identifier']])
        
        # Update chunk progress bar and close it
        chunk_pbar.update(1)
        pbar.close()
    
    # Close chunk progress bar
    chunk_pbar.close()
    
    # Calculate final statistics
    combined_stats['end_time'] = time.time()
    combined_stats['total_time'] = combined_stats['end_time'] - combined_stats['start_time']
    combined_stats['average_download_time'] = (
        sum(combined_stats['download_times']) / len(combined_stats['download_times'])
        if combined_stats['download_times'] else 0
    )
    
    # Combine all results
    final_df = pd.concat(result_dfs, ignore_index=True)
    
    return final_df, combined_stats

def generate_report(stats, total_images):
    """Generate a formatted report of the download statistics"""
    report = [
        "Download Report",
        "=" * 50,
        f"Total images processed: {total_images}",
        f"Successfully downloaded: {stats['success_count']}",
        f"Failed downloads: {len(stats['failed_downloads'])}",
        f"Success rate: {(stats['success_count']/total_images)*100:.2f}%",
        f"Average download time per image: {stats['average_download_time']:.2f} seconds",
        f"Total processing time: {stats['total_time']:.2f} seconds",
        f"Processing speed: {stats['success_count']/stats['total_time']:.2f} images/second",
        f"Theoretical max speed: {stats['success_count']/sum(stats['download_times']):.2f} images/second",
        "=" * 50
    ]
    
    if stats['failed_downloads']:
        report.extend([
            "Failed downloads have been saved to: failed_downloads.csv",
            "Common error types:"
        ])
        
        error_types = defaultdict(int)
        for failed in stats['failed_downloads']:
            error_msg = failed['error'].lower()
            if 'timeout' in error_msg:
                error_types['Timeout'] += 1
            elif 'connection' in error_msg:
                error_types['Connection Error'] += 1
            elif '404' in error_msg:
                error_types['Not Found (404)'] += 1
            elif 'ssl' in error_msg:
                error_types['SSL Error'] += 1
            elif 'memory' in error_msg:
                error_types['Memory Error'] += 1
            else:
                error_types['Other'] += 1
        
        for error_type, count in error_types.items():
            report.append(f"  - {error_type}: {count}")
    
    return "\n".join(report)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Download and process images from URLs in a CSV file.')
    
    parser.add_argument(
        '--input-file',
        type=str,
        default='data/capsfusion_head.csv',
        help='Path to input file containing image_url and capsfusion columns'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=5,
        help='Timeout in seconds for each image download (default: 10)'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=128,
        help='Maximum number of concurrent download threads (default: 4)'
    )
    
    parser.add_argument(
        '--resolution',
        type=int,
        default=224,
        help='Output resolution for images in pixels (default: 224)'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=100000,
        help='Number of rows to process in each chunk (default: 100000)'
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input CSV file not found: {args.input_file}")
    
    try:
        # Create necessary directories
        os.makedirs("images", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        
        # Read input file
        if args.input_file.endswith('.parquet'):
            df = pd.read_parquet(args.input_file)
        else:
            df = pd.read_csv(args.input_file)
        
        # Validate columns
        required_columns = ['image_url', 'capsfusion']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {', '.join(missing_columns)}")
        
        # Process dataframe in chunks
        result_df, stats = process_dataframe_in_chunks(
            df=df,
            chunk_size=args.chunk_size,
            timeout=args.timeout,
            max_workers=args.max_workers,
            resolution=args.resolution
        )
        
        # Save final results
        result_df.to_csv('data/image_metadata_complete.csv', index=False)
        
        # Generate and save report
        report = generate_report(stats, len(df))
        print("\nFinal " + report)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f'reports/download_report_{timestamp}.txt', 'w') as f:
            f.write(report)
        
    except pd.errors.EmptyDataError:
        print("Error: The input CSV file is empty")
    except pd.errors.ParserError:
        print("Error: Unable to parse the input CSV file")
    except Exception as e:
        print(f"Error: {str(e)}")