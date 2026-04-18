import comet_ml
import os
import yaml
import argparse
import torch
try:
    from comet_ml import Experiment
except ImportError:
    Experiment = None

from src.load_models import load_models
from src.projector import VisionProjector
from src.dataset import create_flickr8k_loaders
from src.vlm import VLM
from src.train import train
from src.test import test
from single_test import run_single_test


def main(args):
    """Run training, evaluation, or single-image inference from the CLI."""
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    training_config = config["training"]
    projector_config = config.get("projector", {})
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.command == "single_test":
        caption = run_single_test(
            image_path=args.image,
            checkpoint_path=args.checkpoint,
            config_path="config.yaml",
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
        )
        print(f"Caption: {caption}")
        return

    vision_encoder, image_processor, text_encoder, tokenizer = load_models(
        config["models"]["vision_model_name"],
        config["models"]["language_model_name"],
        device,
    )

    vit_to_lm_projector = VisionProjector(
        vision_dim=vision_encoder.config.hidden_size,
        language_dim=text_encoder.config.hidden_size,
        hidden_multiplier=projector_config.get("hidden_multiplier", 4),
        num_layers=projector_config.get("num_layers", 3),
        dropout=projector_config.get("dropout", 0.1),
        use_gated_blocks=projector_config.get("use_gated_blocks", True),
    )

    train_loader, val_loader, test_loader = create_flickr8k_loaders(
        image_processor=image_processor,
        tokenizer=tokenizer,
        batch_size=training_config["batch_size"],
        max_length=training_config.get("max_length", 64),
        num_workers=training_config.get("num_workers", 0),
        val_split=training_config.get("val_split", 0.1),
        test_split=training_config.get("test_split", 0.1),
        use_train_augmentation=training_config.get("use_train_augmentation", True),
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    vlm = VLM(
        vision_encoder=vision_encoder,
        language_model=text_encoder,
        projector=vit_to_lm_projector,
        tokenizer=tokenizer,
        freeze_language_model=training_config.get("freeze_language_model", False)).to(device)

    total_params = sum(p.numel() for p in vlm.parameters())
    trainable_params = sum(p.numel() for p in vlm.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")

    os.makedirs("checkpoints", exist_ok=True)

    if args.train:
        history = train(
            model=vlm,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=training_config["epochs"],
            lr=float(training_config["lr"]),
            eval_freq=training_config.get("eval_freq", 1),
            weight_decay=float(training_config.get("weight_decay", 0.05)),
            max_grad_norm=float(training_config.get("max_grad_norm", 1.0)),
            early_stopping_patience=training_config.get("early_stopping_patience"),
            early_stopping_min_delta=float(training_config.get("early_stopping_min_delta", 0.0)),
            checkpoint_path="checkpoints/vlm_checkpoint.pt",
            best_model_path="checkpoints/vlm_best.pt",
            comet_project_name=training_config.get("comet_project_name", "VLM"),
            comet_workspace=training_config.get("comet_workspace", None),
            comet_experiment_name=training_config.get("comet_experiment_name", None),
            use_comet=Experiment is not None and args.comet
        )

        print("Training completed.")
        print(f"Best validation loss: {history['best_val_loss']:.4f}")

    if args.test:
        test(
            model=vlm,
            test_loader=test_loader,
            tokenizer=tokenizer,
            image_processor=image_processor,
            device=device,
            checkpoint_path=args.checkpoint,
            results_dir="results",
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training and testing script for the VLM")

    parser.add_argument(
        "command",
        nargs="?",
        choices=["single_test"],
        help="Run single-image inference on a local image",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Run model training",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run evaluation on the test split",
    )


    parser.add_argument(
        "--comet",
        action="store_true",
        help="Enable Comet logging during training",
    )

    parser.add_argument(
        "--image",
        default="image.png",
        help="Path to the image used by the single_test command",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/vlm_best.pt",
        help="Checkpoint path used for test and single_test",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum number of tokens to generate at inference time",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature used during generation",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Enable sampling during generation",
    )

    args = parser.parse_args()
    main(args)
