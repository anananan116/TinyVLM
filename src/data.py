import torch
import os
import random
import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset
from torch.distributed import get_rank
from tqdm import tqdm
from PIL import Image
np.random.seed(42)

SYSTEM_PROMPT = "You are a powerful visual assistant. "

class VLMDataset(Dataset):
    def __init__(self, data, image_file_path: str, image_placeholder: str = "<IMGPLH>"):
        self.image_file_path = image_file_path
        self.data = data
        self.image_placeholder = image_placeholder
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        file_path = os.path.join(self.image_file_path, data["image_path"])
        image = Image.open(file_path)
        instruction = data["instruction"]
        inputs = data["inputs"]
        if self.image_placeholder not in inputs:
            inputs = self.image_placeholder + inputs
        outputs = data["outputs"]
        return instruction, inputs, outputs, image

class VLMCollator:
    def __init__(self, processor, tokenizer: PreTrainedTokenizer, max_length: int, special_token_map: dict, num_patches: int):
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
        self.image_placeholders = f"{self.image_start_token}{"".join([self.image_token] * self.num_patches)}{self.image_end_token}"
        self.processor = processor
        self.image_placeholder_token = "<IMGPLH>"

    def apply_chat_format(self, instruction, inputs, outputs):
        
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT + instruction},
            {"role": "user", "content": [input.replace(self.image_placeholder_token, self.image_placeholders) for input in inputs]},
            {"role": "assistant", "content": outputs}
        ]
        conversation = self.tokenizer.apply_chat_template(conversation, tokenize=True, return_tensors="pt", return_assistant_tokens_mask=True)
        eval_conversation = self.tokenizer.apply_chat_template(conversation, tokenize=True, return_tensors="pt", add_generation_prompt=True, padding="left")
        return conversation, eval_conversation
    
    def create_labels(self, conversation):
        labels = conversation["input_ids"].clone()
        labels[labels == self.pad_token_id] = -100
        labels[conversation["assistant_tokens_mask"] != 1] = -100
        return labels
    
    def __call__(self, batch):
        instruction, inputs, outputs, images = zip(*batch)
        pixel_values = self.processor(images, return_tensors="pt")["pixel_values"]
        
        inputs, eval_inputs = self.apply_chat_format(instruction, inputs, outputs)
        labels = self.create_labels(inputs)
        
        return {
            "images": pixel_values,
            "inputs": inputs,
            "labels": labels,
            "eval_inputs": eval_inputs,
            "reference_answer": outputs
        }
    
class VLMData():
    def __init__(self, args, tokenizer: PreTrainedTokenizer, special_token_map: dict, prosessor):
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
        self.data = pd.read_csv(self.data_path)
        self.training_indices = np.random.choice(self.data.index, int(len(self.data) * (1 - args.validation_proportion)), replace=False)
        self.validation_indices = np.setdiff1d(self.data.index, self.training_indices)
        self.training_data = self.data.loc[self.training_indices]
        self.validation_data = self.data.loc[self.validation_indices]
        if is_main_process:
            print(f"Training data size: {len(self.training_data)}")
            print(f"Validation data size: {len(self.validation_data)}")
        self.training_dataset = VLMDataset(self.training_data, self.encoded_images_file_path)
        self.validation_dataset = VLMDataset(self.validation_data, self.encoded_images_file_path)
        self.collator = VLMCollator(prosessor, tokenizer, self.max_length, special_token_map, self.num_patches)
    
    def get_data(self):
        return self.training_dataset, self.validation_dataset
    
    def get_collator(self):
        return self.collator
