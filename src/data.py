import torch
import os
import random
import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset
from torch.distributed import get_rank
from tqdm import tqdm
from .prompts import CAPTION_PROMPTS
np.random.seed(42)

SYSTEM_PROMPT = "You are a powerful visual assistant."

def get_random_prompt():
    return random.choice(CAPTION_PROMPTS)

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
        return image, caption, data["identifier"]

class VLMCollator:
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
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.image_placeholders = "".join([self.image_token] * self.num_patches)
        self.user_prompt = f"Here's an image: {self.image_start_token}{self.image_placeholders}{self.image_end_token}"
        self.special_ids_series = torch.tensor([128006, 78191, 128007], dtype=torch.long)

    def apply_chat_format(self, caption):
        user_prompt = self.user_prompt + get_random_prompt()
        
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": caption}
        ]
        conversation = self.tokenizer.apply_chat_template(conversation, tokenize=True, return_tensors="pt")
        return conversation[0]
    
    def __call__(self, batch):
        images, captions, identifiers = zip(*batch)
        if isinstance(images[0], np.ndarray):
            images = torch.tensor(np.array(images))
        else:
            images = torch.stack(images)
        
        # Process all examples in the batch
        all_input_ids = []
        all_labels = []
        all_eval_prompts = []
        
        for caption in captions:
            # Get the full conversation input ids
            input_ids = self.apply_chat_format(caption)
            
            # Find the position of special_ids_series in input_ids
            for i in range(len(input_ids) - len(self.special_ids_series) + 1):
                if torch.equal(input_ids[i:i + len(self.special_ids_series)], self.special_ids_series):
                    split_idx = i + len(self.special_ids_series)
                    break
            else:
                raise ValueError("Could not find special_ids_series in input_ids")
            
            # Create evaluation prompt (everything up to and including special_ids_series)
            eval_prompt = input_ids[:split_idx].clone()
            
            # Create labels: -100 for everything before assistant's response
            labels = torch.full_like(input_ids, -100)
            labels[split_idx:] = input_ids[split_idx:]
            
            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_eval_prompts.append(eval_prompt)
        
        # Find max lengths for padding
        max_input_length = min(max(len(ids) for ids in all_input_ids), self.max_length)
        max_eval_length = min(max(len(ids) for ids in all_eval_prompts), self.max_length)
        
        # Pad all sequences
        padded_input_ids = []
        padded_labels = []
        padded_eval_prompts = []
        
        for input_ids, labels, eval_prompt in zip(all_input_ids, all_labels, all_eval_prompts):
            # Truncate if necessary and pad input_ids
            input_ids = input_ids[:max_input_length]
            padding_length = max_input_length - len(input_ids)
            padded_input = torch.cat([
                input_ids,
                torch.full((padding_length,), self.pad_token_id, dtype=torch.long)
            ])
            padded_input_ids.append(padded_input)
            
            # Truncate if necessary and pad labels
            labels = labels[:max_input_length]
            padded_label = torch.cat([
                labels,
                torch.full((padding_length,), -100, dtype=torch.long)
            ])
            padded_labels.append(padded_label)
            
            # Truncate if necessary and pad eval prompts
            eval_prompt = eval_prompt[:max_eval_length]
            eval_padding_length = max_eval_length - len(eval_prompt)
            padded_eval = torch.cat([
                eval_prompt,
                torch.full((eval_padding_length,), self.pad_token_id, dtype=torch.long)
            ])
            padded_eval_prompts.append(padded_eval)
        
        # Stack all tensors
        input_ids = torch.stack(padded_input_ids)
        labels = torch.stack(padded_labels)
        eval_encoded_prompts = torch.stack(padded_eval_prompts)
        image_path = []
        for identifier in identifiers:
            image_path.append(os.path.join("images/", identifier+".jpg"))
        return {
            "encoded_image": images,
            "input_ids": input_ids,
            "labels": labels,
            "reference_captions": captions,
            "eval_input_ids": eval_encoded_prompts,
            "image": image_path
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
