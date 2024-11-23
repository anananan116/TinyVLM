import torch

from transformers import Trainer
from torch import nn
import torch.nn.functional as F
from dataclasses import asdict
from torch import distributed as dist

from .eval import evaluate_caption


class VLMTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute the loss for the given inputs.
        """
        input_ids = inputs["inputs"]["input_ids"]
        attention_mask = inputs["inputs"]["attention_mask"]
        labels = inputs["labels"]
        images = inputs["images"]
        outputs = model(input_ids=input_ids, labels=labels, images=images, attention_mask=attention_mask)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
    
    @torch.no_grad()
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        if eval_dataset is None and self.eval_dataset is None:
            return
        self._memory_tracker.start()
        
        dataloader = self.get_eval_dataloader()
        
        metrics = evaluate_caption(self.model, dataloader, self.accelerator, self.tokenizer)
        
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)
        metrics_with_prefix = {}
        for key, value in metrics.items():
            metrics_with_prefix[f"{metric_key_prefix}_{key}"] = value
        return metrics