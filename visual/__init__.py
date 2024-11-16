from transformers import CLIPProcessor
from .visual_modeling import CLIPModel
import transformers
transformers.logging.set_verbosity_error()

def get_model_and_processor(pretrained_model_name_or_path = "openai/clip-vit-large-patch14-336", device = "cuda"):
    model = CLIPModel.from_pretrained(pretrained_model_name_or_path)
    processor = CLIPProcessor.from_pretrained(pretrained_model_name_or_path)
    model = model.half().to(device)
    processor = processor.image_processor
    return model, processor