import torch.nn as nn 
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, ViTModel

from src.projector import VisionProjector

class VLM(nn.Module):
    """VLM model for image-conditioned caption generation."""
    
    def __init__(
        self,
        vision_encoder: ViTModel,
        language_model: AutoModelForCausalLM,
        projector: VisionProjector,
        tokenizer: AutoTokenizer,
        freeze_language_model: bool = False,
    ):
        """Initialize the multimodal model and optionally freeze the language model."""
        super().__init__()
        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.projector = projector
        self.tokenizer = tokenizer
        self.freeze_language_model = freeze_language_model

        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        if self.freeze_language_model:
            for param in self.language_model.parameters():
                param.requires_grad = False
    
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode an image and project its features into the language space."""
        with torch.no_grad():
            vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        image_features = vision_outputs.last_hidden_state
        projected = self.projector(image_features)
        return projected
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        """Run a training forward pass over image tokens and caption tokens."""
        batch_size = pixel_values.shape[0]

        image_embeds = self.encode_image(pixel_values)
        num_image_tokens = image_embeds.shape[1]

        text_embeds = self.language_model.get_input_embeddings()(input_ids)

        lm_dtype = self.language_model.get_input_embeddings().weight.dtype
        image_embeds = image_embeds.to(dtype=lm_dtype)
        text_embeds = text_embeds.to(dtype=lm_dtype)

        combined_embeds = torch.cat([image_embeds, text_embeds], dim=1)

        image_attention = torch.ones(
            (batch_size, num_image_tokens),
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )
        combined_attention = torch.cat([image_attention, attention_mask], dim=1)

        if labels is not None:
            image_labels = torch.full(
                (batch_size, num_image_tokens),
                fill_value=-100,
                dtype=labels.dtype,
                device=labels.device
            )
            combined_labels = torch.cat([image_labels, labels], dim=1)
        else:
            combined_labels = None

        outputs = self.language_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention,
            labels=combined_labels,
            return_dict=True,
        )
        
        return outputs
    
    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        """Autoregressively generate a caption conditioned on an input image."""
        self.eval()

        lm_weight = self.language_model.get_input_embeddings().weight
        lm_device = lm_weight.device
        lm_dtype = lm_weight.dtype

        pixel_values = pixel_values.to(lm_device)

        image_embeds = self.encode_image(pixel_values)
        image_embeds = image_embeds.to(device=lm_device, dtype=lm_dtype)

        prompt = "This image shows"
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(lm_device)

        generated_ids = prompt_ids.clone()

        for _ in range(max_new_tokens):
            current_embeds = self.language_model.get_input_embeddings()(generated_ids)
            current_embeds = current_embeds.to(device=lm_device, dtype=lm_dtype)

            full_embeds = torch.cat([image_embeds, current_embeds], dim=1)
            full_embeds = full_embeds.to(device=lm_device, dtype=lm_dtype)

            outputs = self.language_model(inputs_embeds=full_embeds)
            next_token_logits = outputs.logits[:, -1, :]

            if do_sample:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
