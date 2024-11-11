#!/bin/bash

# Check if the number of files to process is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <number_of_files_to_process>"
    exit 1
fi

# Get the number of files to process from the first argument
num_files=$1

# Loop through the specified number of CSV files
for ((i = 0; i < num_files; i++))
do
    input_path="../cache/image_metadata_${i}.csv"
    echo "Processing $input_path"
    python encode_images.py --input_path "$input_path"
done

echo "All specified files processed successfully."
