from .modeling_llama import AdapterMLP, DEFAULT_SYSTEM_PROMPT, LlamaForCausalLM
from .configuration_llama import VLMConfig
from .configuration_clip import CLIPConfig
from .visual_modeling import CLIPModel
import torch
from torch import nn
from transformers import AutoProcessor

class AtriVLM(LlamaForCausalLM):
    def __init__(self, config: VLMConfig):
        super().__init__(config)
        if config.special_token_map:
            self.image_start_token_id = config.special_token_map['Image'][1]
            self.image_end_token_id = config.special_token_map['Image_End'][1]
            self.caption_token_id = config.special_token_map['Caption'][1]
            self.image_token_id = config.special_token_map['Image_Token'][1]
        else:
            raise ValueError("Special token map not found")
        self.image_adapter = AdapterMLP(config)
        self.num_patches = config.num_patches
        self.processor = AutoProcessor.from_pretrained(config.pretrained_vision_model).image_processor
        self.img_place_holder = "<IMGPLH>"
        self.img_start_token = "<IMAGE>"
        self.img_end_token = "<IMAGE_END>"
        self.image_token = "<Image_Token>"
        if config.load_vision_model:
            if isinstance(config.visual_config, dict):
                self.visual = CLIPModel(CLIPConfig(**config.visual_config))
            else:
                self.visual = CLIPModel(config.visual_config)
        else:
            self.visual = None
    
    def forward(self, input_ids=None, images= None, encoded_image=None, labels=None, past_key_values = None, attention_mask = None, inputs_embeds = None, **kwargs):
        """
        Forward pass for the VLM model that combines image and text embeddings.
        
        Args:
            input_ids (torch.LongTensor): Input token ids of shape (batch_size, seq_len)
            encoded_image (torch.FloatTensor): Encoded image features of shape (batch_size, num_patches, hidden_dim)
            labels (torch.LongTensor): Labels for computing the language modeling loss
        """
        if images is not None:
            encoded_image = self.visual.encode_image(images)
        if not past_key_values and (encoded_image is not None):
            encoded_image = encoded_image.to(self.get_input_embeddings().weight.dtype)
            # Process image features through the adapter
            processed_image = self.image_adapter(encoded_image)

            # Get embeddings for all input tokens
            token_embeddings = self.get_input_embeddings()(input_ids)
            
            # Find positions of image tokens and replace them with processed image embeddings
            image_token_positions = (input_ids == self.image_token_id).nonzero(as_tuple=True)
            token_embeddings = token_embeddings
            token_embeddings[image_token_positions] = processed_image.reshape(-1, processed_image.size(-1))
        else:
            token_embeddings = self.get_input_embeddings()(input_ids)
        # Call the native forward method with the modified embeddings
        outputs = self._native_forward(
            inputs_embeds=token_embeddings,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        return outputs
    
    
    def prepare_input_ids_for_generation(self, prompts, images, tokenizer, system_prompt=DEFAULT_SYSTEM_PROMPT):
        """
        Prepare input ids and images for generation.
        
        Args:
            prompts (List[str]): List of text prompts
            images (List[Image]): List of images corresponding to prompts
            tokenizer: Tokenizer instance
            system_prompt (str): System prompt to be prepended
            
        Returns:
            dict: Contains input_ids, attention_mask, and processed images
        """
        # Process the images first
        processed_images = []
        for image in images:
            # Process image through vision encoder
            pixel_values = self.processor(image, return_tensors="pt")["pixel_values"].to(self.visual.vision_model.embeddings.patch_embedding.weight.device)
            image_features = self.visual.encode_image(pixel_values)
            processed_images.append(image_features)
        
        # Stack all processed images
        if processed_images:
            processed_images = torch.cat(processed_images, dim=0)
        
        # Process each prompt
        formatted_prompts = []
        for prompt in prompts:
            # Replace image placeholder with tokens
            if self.img_place_holder in prompt:
                image_token_sequence = (
                    f"{self.img_start_token}" + 
                    f"{self.image_token}" * self.num_patches +
                    f"{self.img_end_token}"
                )
                formatted_prompt = prompt.replace(self.img_place_holder, image_token_sequence)
            else:
                formatted_prompt = prompt
                
            # Create conversation format
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_prompt},
            ]
            
            # Apply chat template
            formatted_conversation = tokenizer.apply_chat_template(
                conversation, 
                tokenize=False,
                add_generation_prompt=True
            )
            formatted_prompts.append(formatted_conversation)
        
        # Tokenize all prompts together
        tokenized_output = tokenizer(
            formatted_prompts,
            padding=True,
            return_tensors="pt",
            padding_side="left"  # Use left padding since we're generating on the right
        )
        
        return {
            "input_ids": tokenized_output["input_ids"],
            "attention_mask": tokenized_output["attention_mask"],
            "encoded_image": processed_images if processed_images.size(0) > 0 else None
        }
    
    def prepare_for_generation(self, input_ids, images, **kwargs):
        """
        Prepare KV cache for generation by processing the image and initial tokens.
        
        Args:
            input_ids (torch.LongTensor): Input token ids of shape (batch_size, seq_len)
            encoded_image (torch.FloatTensor): Encoded image features of shape (batch_size, num_patches, hidden_dim)
            
        Returns:
            past_key_values: Tuple containing the key and value states to be used for subsequent generation
        """
        encoded_image = self.visual.encode_image(images)
        # Process image features through the adapter
        processed_image = self.image_adapter(encoded_image)
        
        # Get embeddings for all input tokens
        token_embeddings = self.get_input_embeddings()(input_ids)
        
        # Find positions of image tokens and replace them with processed image embeddings
        image_token_positions = (input_ids == self.image_token_id).nonzero(as_tuple=True)
        token_embeddings[image_token_positions] = processed_image.reshape(-1, processed_image.size(-1))
        
        # Forward pass with cache preparation
        outputs = self._native_forward(
            inputs_embeds=token_embeddings,
            use_cache=True,
            **kwargs
        )
        
        return outputs.past_key_values