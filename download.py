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
from torchvision.transforms import Resize, CenterCrop, InterpolationMode

# Suppress PIL warnings about palette images
warnings.filterwarnings('ignore', category=UserWarning, module='PIL.Image')

class ImageProcessor:
    def __init__(self, timeout=10, max_workers=4, image_size=224, aspect_ratio_threshold=0.6, pbar=None):
        self.timeout = timeout
        self.max_workers = max_workers
        self.image_size = image_size
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.stats_lock = Lock()
        self.pbar = pbar
        self.crop_transforms = [
            Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
        ]
        self.reset_stats()
        
    def reset_stats(self):
        """Reset statistics for new chunk processing"""
        self.stats = {
            'download_times': [],
            'failed_downloads': [],
            'success_count': 0,
            'start_time': time.time(),
            'crop_count': 0,
            'pad_count': 0
        }

    def resize_and_pad(self, img):
        """
        Resize image based on longest side and pad with black
        """
        width, height = img.size
        
        # Calculate scaling factor based on longest side
        longest_side = max(width, height)
        scale = self.image_size / longest_side
        
        # Calculate new dimensions maintaining aspect ratio
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        
        # Create new square image with black background
        new_img = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))
        
        # Calculate paste position to center the image
        paste_x = (self.image_size - new_width) // 2
        paste_y = (self.image_size - new_height) // 2
        
        # Paste resized image onto black background
        new_img.paste(img, (paste_x, paste_y))
        
        return new_img

    def preprocess_image(self, img):
        """
        Adaptively choose preprocessing method based on aspect ratio
        """
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        width, height = img.size
        aspect_ratio = min(width, height) / max(width, height)
        
        # Track which method was used
        if aspect_ratio >= self.aspect_ratio_threshold:
            # Use center crop for images with good aspect ratio
            for transform in self.crop_transforms:
                img = transform(img)
            with self.stats_lock:
                self.stats['crop_count'] += 1
        else:
            # Use padding for images with extreme aspect ratios
            img = self.resize_and_pad(img)
            with self.stats_lock:
                self.stats['pad_count'] += 1
        
        return img

    def process_single_image(self, row):
        """Process a single image"""
        start_time = time.time()
        try:
            response = requests.get(row['image_url'], timeout=self.timeout)
            response.raise_for_status()
            
            img = Image.open(BytesIO(response.content))
            original_width, original_height = img.size
            
            # Apply transforms
            processed_img = self.preprocess_image(img)
            
            output_path = os.path.join("images", f"{row['identifier']}.jpg")
            processed_img.save(output_path, "JPEG", quality=95)
            
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
            
            return True, row['identifier'], original_width, original_height
            
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
            return False, row['identifier'], None, None

def generate_report(stats, total_images):
    """Generate a formatted report of the download statistics"""
    report = [
        "Download Report",
        "=" * 50,
        f"Total images processed: {total_images}",
        f"Successfully downloaded: {stats['success_count']}",
        f"Failed downloads: {len(stats['failed_downloads'])}",
        f"Images center cropped: {stats.get('crop_count', 0)}",
        f"Images padded with black: {stats.get('pad_count', 0)}",
        f"Success rate: {(stats['success_count']/total_images)*100:.2f}%",
        f"Average download time per image: {stats['average_download_time']:.2f} seconds",
        f"Total processing time: {stats['total_time']:.2f} seconds",
        f"Processing speed: {stats['success_count']/stats['total_time']:.2f} images/second",
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

def process_chunk(chunk_df, processor):
    """Process a single chunk of the dataframe"""
    resolutions = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=processor.max_workers) as executor:
        future_to_row = {
            executor.submit(processor.process_single_image, row): row 
            for _, row in chunk_df.iterrows()
        }
        
        for future in concurrent.futures.as_completed(future_to_row):
            success, identifier, width, height = future.result()
            if success:
                resolutions[identifier] = (width, height)
    
    return resolutions

def process_dataframe_in_chunks(df, chunk_size=100000, timeout=10, max_workers=4, image_size=224, aspect_ratio_threshold=0.6):
    """Process a large dataframe by breaking it into smaller chunks"""
    num_chunks = (len(df) + chunk_size - 1) // chunk_size
    
    result_dfs = []
    combined_stats = {
        'download_times': [],
        'failed_downloads': [],
        'success_count': 0,
        'start_time': time.time(),
        'end_time': None,
        'total_time': 0,
        'average_download_time': 0,
        'crop_count': 0,
        'pad_count': 0
    }
    
    chunk_pbar = tqdm(total=num_chunks, desc="Processing chunks", unit="chunk", position=0)
    print(f"\nProcessing {len(df)} rows in {num_chunks} chunks of {chunk_size} rows each")
    
    processor = ImageProcessor(
        timeout=timeout, 
        max_workers=max_workers, 
        image_size=image_size,
        aspect_ratio_threshold=aspect_ratio_threshold
    )
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df))
        chunk_df = df.iloc[start_idx:end_idx].copy()
        chunk_df['identifier'] = [str(uuid.uuid4()) for _ in range(len(chunk_df))]
        
        processor.reset_stats()
        
        pbar = tqdm(total=len(chunk_df), desc=f"Chunk {i+1}/{num_chunks}", unit="img", position=1, leave=False)
        processor.pbar = pbar
        
        resolutions = process_chunk(chunk_df, processor)
        
        chunk_df['original_width'] = chunk_df['identifier'].map(lambda x: resolutions.get(x, (None, None))[0])
        chunk_df['original_height'] = chunk_df['identifier'].map(lambda x: resolutions.get(x, (None, None))[1])
        
        # Update combined statistics
        combined_stats['download_times'].extend(processor.stats['download_times'])
        combined_stats['failed_downloads'].extend(processor.stats['failed_downloads'])
        combined_stats['success_count'] += processor.stats['success_count']
        combined_stats['crop_count'] += processor.stats['crop_count']
        combined_stats['pad_count'] += processor.stats['pad_count']
        
        result_dfs.append(chunk_df[['image_url', 'capsfusion', 'identifier', 'original_width', 'original_height']])
        
        chunk_pbar.update(1)
        pbar.close()
    
    chunk_pbar.close()
    
    combined_stats['end_time'] = time.time()
    combined_stats['total_time'] = combined_stats['end_time'] - combined_stats['start_time']
    combined_stats['average_download_time'] = (
        sum(combined_stats['download_times']) / len(combined_stats['download_times'])
        if combined_stats['download_times'] else 0
    )
    
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
        f"Images center cropped: {stats.get('crop_count', 0)}",
        f"Images padded with black: {stats.get('pad_count', 0)}",
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
        default=448,
        help='Output resolution for images in pixels (default: 448)'
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
            image_size=args.resolution
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