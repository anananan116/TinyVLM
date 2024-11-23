from datasets import load_dataset
from io import BytesIO
from base64 import b64decode
from PIL import Image
import pandas as pd
from tqdm import tqdm
from multiprocessing import Process, Queue, Lock, Value
import os
import sys
import traceback

# Dataset list
dataset_list = [
    "coco", "textcap", "image-paragraph-captioning", "coco-goi", "coco-text",
    "imagenet", "coco-itm", "snli-ve", "mocheg", "iqa", "vqa-v2", "shapes",
    "docvqa", "ocr-vqa", "st-vqa", "text-vqa", "gqa", "okvqa", "a-okvqa",
    "science-qa", "viquae", "clevr", "nlvr", "vcr", "visual-mrc", "winoground",
    "vist", "visual-dialog", "multi30k"
]

dataset_list = [
"vqa-v2", "shapes"
]
# Function to process a single partition
def process_partition(partition, start_counter, output_queue, lock, position):
    try:
        image_paths = []
        counter = start_counter
        temp_df = pd.DataFrame(partition).copy()

        images = temp_df["image_base64_str"]
        temp_df = temp_df.drop("image_base64_str", axis=1)

        with tqdm(
            total=len(images),
            desc=f"Partition {position + 1}",
            position=position,
            leave=True,
            file=sys.stdout
        ) as pbar:
            for image in images:
                img = Image.open(BytesIO(b64decode(image[0])))
                img_path = f"data/it_images/{counter}.jpg"
                img.save(img_path)
                image_paths.append(img_path)

                with lock:
                    counter += 1

                pbar.update(1)

        temp_df["image_path"] = image_paths
        output_queue.put(temp_df)
        print(f"Partition {position + 1} done")
    except Exception as e:
        print(f"Error in partition {position + 1}: {e}")
        traceback.print_exc()

# Main function to process datasets
def process_datasets(dataset_list):
    if not os.path.exists("data/it_images"):
        os.makedirs("data/it_images")

    result_df = pd.DataFrame()
    image_counter = Value('i', 0)  # Shared counter
    lock = Lock()

    for dataset_name in tqdm(dataset_list, desc="Datasets", file=sys.stdout):
        print(f"Processing {dataset_name}")
        dataset = load_dataset("MMInstruction/M3IT", dataset_name, trust_remote_code=True)
        train_set = pd.DataFrame(dataset["train"])

        # Split dataset into partitions
        num_processes = os.cpu_count()  # Use number of CPU cores
        partition_size = len(train_set) // num_processes
        partitions = [
            train_set.iloc[i * partition_size: (i + 1) * partition_size]
            for i in range(num_processes)
        ]

        # Handle any remaining data
        if len(train_set) % num_processes != 0:
            partitions.append(train_set.iloc[num_processes * partition_size:])

        # Use multiprocessing
        processes = []
        output_queue = Queue()

        try:
            for idx, partition in enumerate(partitions):
                with image_counter.get_lock():
                    start_counter = image_counter.value

                process = Process(
                    target=process_partition,
                    args=(partition, start_counter, output_queue, lock, idx)
                )
                processes.append(process)
                process.start()

                with lock:
                    image_counter.value += len(partition)

            # Wait for all processes to finish
            for process in processes:
                process.join()

        except Exception as e:
            print(f"Error during multiprocessing: {e}")
            traceback.print_exc()

        # Collect results from processes
        while not output_queue.empty():
            result_df = pd.concat([result_df, output_queue.get()])

    # Save results to CSV
    result_df.to_csv("data/it_train.csv", index=False)

if __name__ == "__main__":
    try:
        process_datasets(dataset_list)
    except Exception as e:
        print(f"Critical error in main process: {e}")
        traceback.print_exc()