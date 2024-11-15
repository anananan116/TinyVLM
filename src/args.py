from transformers.training_args import TrainingArguments
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Union, Dict

@dataclass
class CustomTrainingArgs(TrainingArguments):
    # ==============================
    # Common arguments
    # ==============================
    output_dir: str = field(
        default="results/outputs/pretrain",
    )
    learning_rate: float = field(
        default=1e-5,
        metadata={'help': 'Learning rate.'}
    )
    per_device_train_batch_size: int = field(
        default=16,
        metadata={'help': 'Train batch size.'}
    )
    per_device_eval_batch_size: int = field(
        default=1,
        metadata={'help': 'Evaluation batch size.'}
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={'help': 'Gradient accumulation steps.'}
    )
    bf16: bool = field(
        default=False,
        metadata={'help': 'Use BF16?'}
    )
    fp16: bool = field(
        default=False,
        metadata={'help': 'Use FP16?'}
    )
    eval_strategy: str = field(
        default='steps',
        metadata={'help': 'Strategy for evaluation.'}
    )
    eval_steps: int = field(
        default=200,
        metadata={'help': 'Number of steps between evaluations.'}
    )
    logging_steps: int = field(
        default=50,
        metadata={'help': 'Number of steps between logs.'}
    )
    save_steps: int = field(
        default=1000,
        metadata={'help': 'Number of steps between saves.'}
    )
    save_total_limit: int = field(
        default=3,
        metadata={'help': 'Number of saves to keep.'}
    )
    save_strategy: str = field(
        default='steps',
        metadata={'help': 'Strategy for saving models.'}
    )
    deepspeed: str = field(
        default='',
        metadata={'help': 'Deepspeed config file.'}
    )
    num_train_epochs: int = field(
        default=1,
        metadata={'help': 'Number of training epochs.'}
    )
    dataloader_num_workers: int = field(
        default=8,
        metadata={'help': 'Number of dataloader workers.'}
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={'help': 'Use gradient checkpointing?'}
    )
    
    
@dataclass
class VLMTrainingArguments:
    validation_proportion: float = field(
        default=0.001,
        metadata={'help': 'Proportion of data to use for validation.'}
    )
    
    data_path: str = field(
        default='filtered/',
        metadata={'help': 'Path to data files.'}
    )
    
    data_amount: int = field(
        default=1,
        metadata={'help': 'Amount of data partitions to be used. One partition usually contains ~75k samples.'}
    )
    
    encoded_images_file_path: str = field(
        default='encoded/',
        metadata={'help': 'Path to encoded images'}
    )
    
    max_length: int = field(
        default=192,
        metadata={'help': 'Maximum length of input sequences.'}
    )
    
    num_patches: int = field(
        default=64,
        metadata={'help': 'Number of patches in an image.'}
    )
    
@dataclass
class ModelArguments:
    special_token_config: Optional[str] = field(
        default="configs/Specialtokens/default.json",
        metadata={'help': 'Special token config file.'}
    )
    
    model_config_path: str = field(
        default='configs/VLM/llama_full.yaml',
        metadata={'help': 'Path to model config file.'}
    )
    
    flashattention: bool = field(
        default=False,
        metadata={'help': 'Use FlashAttention?'}
    )