from src.model.llama import get_model_and_tokenizer
import json
from PIL import Image
import torch

model_args = {}
with open("configs/Specialtokens/default.json") as f:
    special_token_map = json.load(f)
model_args["pretrained_model"] = "results/checkpoint-10000"
additional_tokens_dict = {x['type']: x['token'] for x in special_token_map['added_tokens']}
model, tokenizer, _, _ = get_model_and_tokenizer(model_args, additional_tokens_dict, load_vision_model=True)
model = model.to(torch.float16)

model.push_to_hub("anananan116/TinyVLM")
tokenizer.push_to_hub("anananan116/TinyVLM")