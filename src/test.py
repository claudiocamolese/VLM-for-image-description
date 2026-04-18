import os
import torch
import numpy as np
from PIL import Image


def _tensor_to_pil(pixel_values, image_processor):
    """Convert normalized image tensor values back into a PIL image."""
    image = pixel_values.detach().cpu().clone()

    if image.ndim == 4:
        image = image[0]

    mean = torch.tensor(image_processor.image_mean).view(3, 1, 1)
    std = torch.tensor(image_processor.image_std).view(3, 1, 1)

    image = image * std + mean
    image = image.clamp(0, 1)

    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)

    return Image.fromarray(image)


@torch.no_grad()
def test(
    model,
    test_loader,
    tokenizer,
    image_processor,
    device,
    checkpoint_path="checkpoints/vlm_best.pt",
    results_dir="results",
    max_new_tokens=64,
    temperature=0.8,
    do_sample=False,
):
    """Generate captions for the test split and save images plus text outputs."""

    os.makedirs(results_dir, exist_ok=True)
    images_dir = os.path.join(results_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        if "projector_state_dict" in checkpoint:
            model.projector.load_state_dict(checkpoint["projector_state_dict"])
        if "language_model_state_dict" in checkpoint:
            model.language_model.load_state_dict(checkpoint["language_model_state_dict"])
        if "vision_encoder_state_dict" in checkpoint:
            model.vision_encoder.load_state_dict(checkpoint["vision_encoder_state_dict"])

    model.to(device)
    model.eval()

    captions_file = os.path.join(results_dir, "captions.txt")

    sample_idx = 0
    with open(captions_file, "w", encoding="utf-8") as f:
        for batch_idx, batch in enumerate(test_loader):
            pixel_values = batch["pixel_values"].to(device)

            input_ids = batch.get("input_ids", None)
            if input_ids is not None:
                input_ids = input_ids.to(device)

            batch_size = pixel_values.size(0)

            for i in range(batch_size):
                single_image = pixel_values[i].unsqueeze(0)

                generated_caption = model.generate(
                    pixel_values=single_image,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                )

                ground_truth = ""
                if input_ids is not None:
                    gt_ids = input_ids[i]
                    ground_truth = tokenizer.decode(gt_ids, skip_special_tokens=True)

                pil_img = _tensor_to_pil(pixel_values[i], image_processor)
                image_name = f"test_{sample_idx:05d}.png"
                image_path = os.path.join(images_dir, image_name)
                pil_img.save(image_path)

                f.write(f"{image_name}\n")
                f.write(f"Generated: {generated_caption}\n")
                if ground_truth:
                    f.write(f"Ground truth: {ground_truth}\n")
                f.write("\n")

                print(f"[{sample_idx}] saved {image_name} -> {generated_caption}")
                sample_idx += 1

    print("\nTest completed.")
    print(f"Images saved to: {images_dir}")
    print(f"Captions saved to: {captions_file}")
