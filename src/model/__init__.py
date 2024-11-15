from .retrieval_models import RetrievalModel, RetrievalModelConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .llama.creditials import hugging_face_token
except:
    hugging_face_token = None

MODEL_MAP = {
    "Retrieval": RetrievalModel,
}

CONFIG_MAP = {
    "Retrieval": RetrievalModelConfig,
}

def get_model_and_tokenizer(model_args, additional_tokens_dict, device="cuda"):
    config = CONFIG_MAP[model_args['model_type']](**model_args)
    
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model, token = hugging_face_token)
    tokenizer.add_tokens([v for k,v in additional_tokens_dict.items()], special_tokens=True)
    special_token_map = {k: (v, tokenizer.convert_tokens_to_ids(v)) for k,v in additional_tokens_dict.items()}
    new_tokenizer_len = len(tokenizer)
    config.adjust_embedding_len = new_tokenizer_len
    config.special_token_map = special_token_map
    
    model = MODEL_MAP[model_args['model_type']](config).to(device)
    return model, tokenizer, special_token_map