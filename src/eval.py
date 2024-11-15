import torch
from tqdm import tqdm
from torch import distributed as dist
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from collections import defaultdict

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
            past_key_values = model.prepare_kv_cache(
                batch['eval_input_ids'][:, :-1].to(device), 
                batch['images'].to(device)
            )
            
            # Generate captions
            generated_ids = model.generate(
                input_ids=batch['eval_input_ids'][:, -1:].to(device),  # Only use last token (caption token)
                past_key_values=past_key_values,
                max_length=64,
                pad_token_id=model.config.pad_token_id,
                bos_token_id=model.config.bos_token_id,
                eos_token_id=model.config.eos_token_id,
            )
            
            generated_captions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            reference_captions = batch['reference_captions']  # Use raw reference captions directly
            
            # Calculate metrics for batch
            num_samples = len(generated_captions)
            smoothing = SmoothingFunction().method1
            
            for gen_cap, ref_cap in zip(generated_captions, reference_captions):
                # Tokenize captions
                gen_tokens = word_tokenize(gen_cap.lower())
                ref_tokens = word_tokenize(ref_cap.lower())
                
                # Exact Match
                batch_metrics['exact_match'] += float(gen_cap.lower() == ref_cap.lower())
                
                # BLEU scores (1-3)
                for n in range(1, 4):
                    bleu_score = sentence_bleu(
                        [ref_tokens],
                        gen_tokens,
                        weights=tuple([1.0/n] * n + [0.0] * (4-n)),
                        smoothing_function=smoothing
                    )
                    batch_metrics[f'bleu_{n}'] += bleu_score
            
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