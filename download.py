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

# Suppress PIL warnings about palette images
warnings.filterwarnings('ignore', category=UserWarning, module='PIL.Image')

class ImageProcessor:
    def __init__(self, timeout=10, max_workers=4, resolution=224):
        self.timeout = timeout
        self.max_workers = max_workers
        self.resolution = resolution
        self.stats_lock = Lock()
        self.stats = {
            'download_times': [],
            'failed_downloads': [],
            'success_count': 0,
            'start_time': time.time()
        }
        
    def process_single_image(self, task):
        """Process a single image from the task queue"""
        row, idx = task
        start_time = time.time()
        try:
            response = requests.get(row['image_url'], timeout=self.timeout)
            response.raise_for_status()
            
            img = Image.open(BytesIO(response.content))
            
            # Handle different image modes
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
            
            return True, idx
            
        except Exception as e:
            with self.stats_lock:
                self.stats['failed_downloads'].append({
                    'url': row['image_url'],
                    'identifier': row['identifier'],
                    'error': str(e)
                })
            return False, idx

    def worker(self, task_queue, results):
        """Worker function that processes tasks from the queue"""
        while True:
            try:
                task = task_queue.get_nowait()
            except Queue.Empty:
                break
            
            success, idx = self.process_single_image(task)
            results[idx] = success
            task_queue.task_done()

def download_and_process_images(df, timeout=10, max_workers=4, resolution=224):
    """
    Download images from URLs, process them, and save metadata using dynamic thread assignment
    """
    os.makedirs("images", exist_ok=True)
    
    df = df.copy()
    df['identifier'] = [str(uuid.uuid4()) for _ in range(len(df))]
    
    # Initialize processor and queues
    processor = ImageProcessor(timeout, max_workers, resolution)
    task_queue = Queue()
    results = [None] * len(df)
    
    # Fill task queue
    for idx, row in df.iterrows():
        task_queue.put((row, idx))
    
    # Create thread pool and start processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        workers = []
        for _ in range(max_workers):
            worker = executor.submit(processor.worker, task_queue, results)
            workers.append(worker)
        
        # Wait for all tasks to complete
        concurrent.futures.wait(workers)
    
    # Calculate final statistics
    processor.stats['end_time'] = time.time()
    processor.stats['total_time'] = processor.stats['end_time'] - processor.stats['start_time']
    processor.stats['average_download_time'] = (
        sum(processor.stats['download_times']) / len(processor.stats['download_times'])
        if processor.stats['download_times'] else 0
    )
    
    # Save metadata
    output_df = df[['image_url', 'capsfusion', 'identifier']]
    output_df.to_csv('data/image_metadata.csv', index=False)
    
    # Save failed downloads
    if processor.stats['failed_downloads']:
        failed_df = pd.DataFrame(processor.stats['failed_downloads'])
        failed_df.to_csv('data/failed_downloads.csv', index=False)
    
    return output_df, processor.stats

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
        '--input-csv',
        type=str,
        default='data/capsfusion_head.csv',
        help='Path to input CSV file containing image_url and capsfusion columns'
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
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input CSV file not found: {args.input_csv}")
    
    try:
        df = pd.read_csv(args.input_csv)
        
        required_columns = ['image_url', 'capsfusion']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {', '.join(missing_columns)}")
        
        result_df, stats = download_and_process_images(
            df=df,
            timeout=args.timeout,
            max_workers=args.max_workers,
            resolution=args.resolution
        )
        
        report = generate_report(stats, len(df))
        print(report)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not os.path.exists('reports'):
            os.mkdir('reports')
        with open(f'reports/download_report_{timestamp}.txt', 'w') as f:
            f.write(report)
        
    except pd.errors.EmptyDataError:
        print("Error: The input CSV file is empty")
    except pd.errors.ParserError:
        print("Error: Unable to parse the input CSV file")
    except Exception as e:
        print(f"Error: {str(e)}")