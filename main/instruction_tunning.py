import logging
import sys
import os
import yaml
import pytz
import json
import datetime
import torch
from torch import distributed as dist

from transformers import HfArgumentParser
import transformers
from src import (
    VLMData,
    VLMTrainer,
    get_model_and_tokenizer,
    CustomTrainingArgs,
    ModelArguments,
    VLMTrainingArguments,
)
from .utils import TensorBoardCallback

logger = logging.getLogger(__name__)

console_handler = logging.StreamHandler(sys.stdout)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

def main():
    parser = HfArgumentParser([CustomTrainingArgs, VLMTrainingArguments, ModelArguments])

    training_args, custom_training_args, model_args = parser.parse_args_into_dataclasses()
    for key, value in vars(custom_training_args).items():
        setattr(training_args, key, value)
    is_main_process = dist.get_rank() == 0
    if is_main_process:
        print(f"Training arguments: {training_args}")
    model_args_path = model_args.model_config_path
    additional_tokens_path =  model_args.special_token_config
    flashattention = False
    with open(model_args_path, 'r') as f:
        model_args = yaml.load(f, Loader=yaml.FullLoader)
    if flashattention:
        model_args['flashattention'] = flashattention
    with open(additional_tokens_path, 'r') as f:
        added_tokens = json.load(f)
    additional_tokens_dict = {x['type']: x['token'] for x in added_tokens['added_tokens']}
    if is_main_process:
        logger.info(f"Additional tokens: {additional_tokens_dict}")
    model_args["pretrained_model"] = "results/checkpoint-29300"
    model, tokenizer, special_token_map, prosessor = get_model_and_tokenizer(model_args, additional_tokens_dict, load_vision_model=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_token_id = tokenizer.eos_token_id

    data = VLMData(training_args, tokenizer, special_token_map, prosessor = prosessor)
    train_dataset, eval_dataset = data.get_data()
    collate_fn = data.get_collator()
    tensorboard_callback = TensorBoardCallback(log_dir=training_args.logging_dir)
    if is_main_process:
        print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    trainer = VLMTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        callbacks=[tensorboard_callback],
    )

    trainer.train()
    trainer.evaluate()
if __name__ == "__main__":
    main()