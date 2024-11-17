import torch
from tqdm import tqdm
from torch import distributed as dist
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

def generate_with_cache(
    model,
    past_key_values,
    input_ids: torch.Tensor,
    eos_id: int,
    max_length: int = 196,
    temperature: float = 0.8,
    top_p: float = 0.8,
) -> Tuple[torch.Tensor, List[int]]:
    
    # Initialize generation variables
    batch_size = input_ids.shape[0]
    current_length = input_ids.shape[1]
    generated_tokens = []
    
    # Track which sequences have finished
    finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
    
    # Generation loop
    while current_length < max_length and not finished_sequences.all():
        # Forward pass with caching
        with torch.no_grad():
            outputs = model._native_forward(
                input_ids if past_key_values is None else input_ids[:, -1:],
                attention_mask=None,
                position_ids=None,
                labels=None,
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            # Get logits and update cache
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            
            # Apply temperature
            scaled_logits = next_token_logits / temperature
            
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
            
            # Apply top-p (nucleus) sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Zero out filtered logits
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            probs[indices_to_remove] = 0
            
            # For finished sequences, force EOS token by zeroing out all other probabilities
            probs[finished_sequences] = 0
            probs[finished_sequences, eos_id] = 1
            
            next_token = torch.multinomial(probs, num_samples=1)

            
            # Mask tokens with EOS for finished sequences
            next_token = torch.where(finished_sequences.unsqueeze(1), 
                                   torch.tensor(eos_id, device=next_token.device),
                                   next_token)
            
            # Update finished sequences mask
            finished_sequences = finished_sequences | (next_token.squeeze(-1) == eos_id)
            
            # Store generated tokens
            generated_tokens.append(next_token.clone())
            
            # Update input_ids for next iteration
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            current_length += 1
            
            # If all sequences are finished, break early
            if finished_sequences.all():
                break
    
    # Stack all generated tokens
    all_tokens = torch.cat(generated_tokens, dim=1)
    
    return all_tokens

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
            eval_input_ids = batch['eval_input_ids'].to(device)
            # Prepare KV cache
            past_key_values = model.prepare_for_generation(
                eval_input_ids[:, :-1], 
                batch['encoded_image'].to(device)
            )
            
            # Generate captions
            generated_ids = generate_with_cache(model, past_key_values, eval_input_ids, tokenizer.eos_token_id)
            
            generated_captions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            reference_captions = batch['reference_captions']  # Use raw reference captions directly
            
            # Calculate metrics for batch
            num_samples = len(generated_captions)
            smoothing = SmoothingFunction().method1
            
            for gen_cap, ref_cap in zip(generated_captions, reference_captions):
                # Tokenize captions
                gen_tokens = word_tokenize(gen_cap.lower())
                ref_tokens = word_tokenize(ref_cap.lower())
                
                # BLEU scores (1-3)
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
                batch['input_ids'].to(device), 
                batch['encoded_image'].to(device), 
                labels=eval_input_ids
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