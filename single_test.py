import os
import argparse
import yaml
import torch
from PIL import Image, ImageOps

from src.load_models import load_models
from src.projector import VisionProjector
from src.vlm import VLM


def _resolve_image_size(image_processor):
    """Extract the target image size expected by the image processor."""
    size = getattr(image_processor, "size", 224)

    if isinstance(size, dict):
        if "height" in size and "width" in size:
            return (size["width"], size["height"])
        if "shortest_edge" in size:
            edge = size["shortest_edge"]
            return (edge, edge)

    if isinstance(size, int):
        return (size, size)

    if isinstance(size, (tuple, list)) and len(size) == 2:
        return (size[0], size[1])

    return (224, 224)


def _prepare_image_for_inference(image, image_processor):
    """Resize and center-crop an image to the processor target resolution."""
    target_size = _resolve_image_size(image_processor)
    resampling = getattr(Image, "Resampling", Image).BICUBIC

    if image.mode != "RGB":
        image = image.convert("RGB")

    return ImageOps.fit(image, target_size, method=resampling)


def _load_checkpoint(model, checkpoint_path, device):
    """Load model weights from a checkpoint file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        return

    if "projector_state_dict" in checkpoint:
        model.projector.load_state_dict(checkpoint["projector_state_dict"])
    if "language_model_state_dict" in checkpoint:
        model.language_model.load_state_dict(checkpoint["language_model_state_dict"])
    if "vision_encoder_state_dict" in checkpoint:
        model.vision_encoder.load_state_dict(checkpoint["vision_encoder_state_dict"])


def load_trained_model(config_path="config.yaml", checkpoint_path="checkpoints/vlm_best.pt", device=None):
    """Build the model stack and load trained weights for inference."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    training_config = config["training"]
    projector_config = config.get("projector", {})

    vision_encoder, image_processor, text_encoder, tokenizer = load_models(
        config["models"]["vision_model_name"],
        config["models"]["language_model_name"],
        device,
    )

    projector = VisionProjector(
        vision_dim=vision_encoder.config.hidden_size,
        language_dim=text_encoder.config.hidden_size,
        hidden_multiplier=projector_config.get("hidden_multiplier", 4),
        num_layers=projector_config.get("num_layers", 3),
        dropout=projector_config.get("dropout", 0.1),
        use_gated_blocks=projector_config.get("use_gated_blocks", True),
    )

    model = VLM(
        vision_encoder=vision_encoder,
        language_model=text_encoder,
        projector=projector,
        tokenizer=tokenizer,
        freeze_language_model=training_config.get("freeze_language_model", False),
    ).to(device)

    _load_checkpoint(model, checkpoint_path, device)
    model.eval()

    return model, image_processor, tokenizer, device


@torch.no_grad()
def run_single_test(
    image_path="image.png",
    checkpoint_path="checkpoints/vlm_best.pt",
    config_path="config.yaml",
    device=None,
    max_new_tokens=64,
    temperature=0.8,
    do_sample=False,
):
    """Run caption generation for a single local image."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    model, image_processor, _, device = load_trained_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    image = Image.open(image_path)
    prepared_image = _prepare_image_for_inference(image, image_processor)
    pixel_values = image_processor(prepared_image, return_tensors="pt").pixel_values.to(device)

    caption = model.generate(
        pixel_values=pixel_values,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
    )

    return caption


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run single-image inference with the VLM")
    parser.add_argument("--image", default="image.png", help="Path to the image to caption")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/vlm_best.pt",
        help="Checkpoint file to load",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to the configuration file")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Maximum number of generated tokens")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature for generation")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling during generation")

    args = parser.parse_args()
    caption = run_single_test(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
    )
    print(f"Caption: {caption}")
