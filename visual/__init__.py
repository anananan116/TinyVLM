from transformers import CLIPProcessor, CLIPModel

def get_model_and_processor(pretrained_model_name_or_path, device):
    model = CLIPModel.from_pretrained(pretrained_model_name_or_path)
    processor = CLIPProcessor.from_pretrained(pretrained_model_name_or_path)
    model = model.vision_model.half().to(device)
    processor = processor.image_processor
    return model, processor