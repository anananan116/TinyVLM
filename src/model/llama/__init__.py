from .modeling_llama import LlamaForCausalLM, AtriVLM
from .configuration_llama import LlamaConfig, VLMConfig
from transformers import AutoTokenizer

try:
    from .creditials import hugging_face_token
except:
    hugging_face_token = None


def get_model_and_tokenizer(model_args, additional_tokens_dict, device="cuda"):
    pretrained_model = 'meta-llama/Llama-3.2-1B-Instruct' if 'pretrained_model' not in model_args else model_args['pretrained_model']
    
    config = VLMConfig.from_pretrained(pretrained_model, **model_args)
    config.pretrained_model = pretrained_model
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model, token = hugging_face_token)
    tokenizer.add_tokens([v for k,v in additional_tokens_dict.items()], special_tokens=True)
    special_token_map = {k: (v, tokenizer.convert_tokens_to_ids(v)) for k,v in additional_tokens_dict.items()}
    new_tokenizer_len = len(tokenizer)
    config.adjust_embedding_len = new_tokenizer_len
    config.special_token_map = special_token_map
    
    model = AtriVLM.from_pretrained(config.pretrained_model, config = config, token = hugging_face_token)
    
    if config.adjust_embedding_len:
        model.resize_token_embeddings(config.adjust_embedding_len, mean_resizing=True)
    
    if config.lora:
        from peft import get_peft_model, LoraConfig, TaskType
        lora_config = LoraConfig(r=config.lora_rank, lora_alpha=config.lora_alpha, target_modules=config.lora_modules, task_type=config.task_type)
        model = get_peft_model(model, lora_config)
    
    return model, tokenizer, special_token_map