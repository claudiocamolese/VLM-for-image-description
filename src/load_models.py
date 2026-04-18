from transformers import AutoModelForCausalLM, AutoTokenizer, ViTModel, ViTImageProcessor

import yaml
import torch

def load_models(vision_model_name, language_model_name, device):
    """Load the vision encoder, image processor, language model, and tokenizer."""
    vision_encoder = ViTModel.from_pretrained(vision_model_name)
    image_processor = ViTImageProcessor.from_pretrained(vision_model_name)

    text_encoder = AutoModelForCausalLM.from_pretrained(language_model_name)
    tokenizer = AutoTokenizer.from_pretrained(language_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return vision_encoder, image_processor, text_encoder, tokenizer

if __name__ == "__main__":
    with open("config.yaml") as file:
        config = yaml.safe_load(file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    vision_encoder, image_processor, text_encoder, tokenizer= load_models(config["models"]["vision_model_name"], config["models"]["language_model_name"], device)
