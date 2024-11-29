import torch
from tqdm import tqdm
from torch import distributed as dist
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

def evaluate_caption(model, dataloader, accelerator, tokenizer):
    """
    Evaluate the model's caption generation performance using various metrics.
    """
    if accelerator is not None and type(dataloader) == torch.utils.data.DataLoader:
        dataloader = accelerator.prepare(dataloader)

    device = accelerator.device
    model.eval()
    
    local_metrices = []
    is_main_process = dist.get_rank() == 0
    data_iter = tqdm(dataloader) if is_main_process else dataloader
    
    with torch.no_grad():
        for batch in data_iter:
            batch_metrics = defaultdict(float)
            # Prepare KV cache
            images = batch['images'].to(device)
            encoded_images = model.visual.encode_image(images)
            inputs = batch['eval_inputs']
            # Generate captions
            outputs = model.generate(
                input_ids=inputs['input_ids'].to(device), 
                attention_mask=inputs['attention_mask'].to(device), 
                encoded_image = encoded_images, 
                max_new_tokens=128, 
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_captions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated_captions = [text.split("assistant\n\n")[1] for text in generated_captions]
            reference_captions = batch['reference_answer']  # Use raw reference captions directly
            
            # Calculate metrics for batch
            num_samples = len(generated_captions)
            smoothing = SmoothingFunction().method1
            
            for gen_cap, ref_cap in zip(generated_captions, reference_captions):
                # Tokenize captions
                gen_tokens = word_tokenize(gen_cap.lower())
                ref_tokens = word_tokenize(ref_cap.lower())
                
                # BLEU scores (1-4)
                for n in range(1, 5):
                    bleu_score = sentence_bleu(
                        [ref_tokens],
                        gen_tokens,
                        weights=tuple([1.0/n] * n + [0.0] * (4-n)),
                        smoothing_function=smoothing
                    )
                    batch_metrics[f'bleu_{n}'] += bleu_score
            
            # forward pass
            batch_metrics['eval_loss'] = model(
                batch['inputs']["input_ids"].to(device),
                batch['images'].to(device), 
                labels= batch['labels'].to(device)
            ).loss.item() * num_samples
            
            # Average metrics for this batch
            batch_metrics = {k: v / num_samples for k, v in batch_metrics.items()}
            local_metrices.append(batch_metrics)
    
    # Aggregate results across all batches and devices
    avg_metrices = {}
    for key in local_metrices[0].keys():
        local_sum = sum([metric[key] for metric in local_metrices])
        local_avg = local_sum / len(local_metrices)
        tensor_avg = torch.tensor(local_avg, device=device)
        dist.all_reduce(tensor_avg, op=dist.ReduceOp.SUM)
        tensor_avg /= dist.get_world_size()
        avg_metrices[key] = tensor_avg.item()

    model.train()
    if is_main_process:
        print("\nEvaluation Results:")
        for metric, score in avg_metrices.items():
            print(f"{metric}: {score:.4f}")
            
    return avg_metrices