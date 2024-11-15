import torch
import os
import random
import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset
from torch.distributed import get_rank
from tqdm import tqdm

np.random.seed(42)

class VLMDataset(Dataset):
    def __init__(self, data, encoded_images_file_path: str):
        self.encoded_images_file_path = encoded_images_file_path
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        file_path = os.path.join(self.encoded_images_file_path, data["identifier"]+".npy")
        image = np.load(file_path)
        caption = data["capsfusion"]
        return image, caption

class VLMCollator():
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int, special_token_map: dict, num_patches: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.special_token_map = special_token_map
        self.image_start_token = self.special_token_map["Image"][0]
        self.image_end_token = self.special_token_map["Image_End"][0]
        self.caption_start_token = self.special_token_map["Caption"][0]
        self.caption_start_id = self.special_token_map["Caption"][1]
        self.image_token = self.special_token_map["Image_Token"][0]
        self.num_patches = num_patches
    
    def __call__(self, batch):
        images, captions = zip(*batch)
        images = torch.tensor(np.array(images))
        
        # Format captions with special tokens and space for image tokens
        image_placeholder = "".join([self.image_token] * self.num_patches)
        formatted_captions = [f"{self.image_start_token}{image_placeholder}{self.image_end_token}\n{self.caption_start_token}{caption}" for caption in captions]
        
        # Encode captions
        encoded_captions = self.tokenizer(
            formatted_captions,
            max_length=self.max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            padding_side="right"
        )
        
        # Create labels tensor
        labels = encoded_captions["input_ids"].clone()
        
        # For each sequence in the batch
        for i in range(labels.shape[0]):
            # Find position of caption_start_token
            caption_start_pos = (labels[i] == self.caption_start_id).nonzero(as_tuple=True)[0][0]
            
            # Set all tokens before caption (including caption_start_token) to -100
            labels[i, :caption_start_pos + 1] = -100
            
            # Set padding tokens to -100
            padding_mask = encoded_captions["input_ids"][i] == self.tokenizer.pad_token_id
            labels[i, padding_mask] = -100
        
        eval_prompts = [f"{self.image_start_token}{image_placeholder}{self.image_end_token}\n{self.caption_start_token}" for _ in captions]
        
        # Encode prompts
        eval_encoded_prompts = self.tokenizer(
            eval_prompts, 
            max_length=self.max_length, 
            padding="longest", 
            truncation=True, 
            return_tensors="pt",
            padding_side="right"
        )
        
        return {
            "encoded_image": images,
            "input_ids": encoded_captions["input_ids"],
            "labels": labels,
            "reference_captions": captions,
            "eval_input_ids": eval_encoded_prompts["input_ids"],
        }
    
class VLMData():
    def __init__(self, args, tokenizer: PreTrainedTokenizer, special_token_map: dict):
        self.data_path = args.data_path
        self.encoded_images_file_path = args.encoded_images_file_path
        self.tokenizer = tokenizer
        self.max_length = args.max_length
        self.num_data_partitions = args.data_amount
        self.num_patches = args.num_patches
        self.data = []
        try:
            is_main_process = get_rank() == 0
        except:
            is_main_process = True
        if is_main_process:
            print(f"Loading {args.data_amount} partitions from {self.data_path}")
            for i in tqdm(range(self.num_data_partitions)):
                data_partition = f"{self.data_path}/image_metadata_{i}_filtered.csv"
                one_partition = pd.read_csv(data_partition).loc[:, ["identifier", "capsfusion"]]
                self.data.append(one_partition)
            print("Data loaded")
        else:
            for i in range(self.num_data_partitions):
                data_partition = f"{self.data_path}/image_metadata_{i}_filtered.csv"
                one_partition = pd.read_csv(data_partition).loc[:, ["identifier", "capsfusion"]]
                self.data.append(one_partition)
        self.data = pd.concat(self.data, axis=0, ignore_index=True)
        self.training_indices = np.random.choice(self.data.index, int(len(self.data) * (1 - args.validation_proportion)), replace=False)
        self.validation_indices = np.setdiff1d(self.data.index, self.training_indices)
        self.training_data = self.data.loc[self.training_indices]
        self.validation_data = self.data.loc[self.validation_indices]
        if is_main_process:
            print(f"Training data size: {len(self.training_data)}")
            print(f"Validation data size: {len(self.validation_data)}")
        self.training_dataset = VLMDataset(self.training_data, self.encoded_images_file_path)
        self.validation_dataset = VLMDataset(self.validation_data, self.encoded_images_file_path)
        self.collator = VLMCollator(tokenizer, self.max_length, special_token_map, self.num_patches)
    
    def get_data(self):
        return self.training_dataset, self.validation_dataset
    
    def get_collator(self):
        return self.collator
