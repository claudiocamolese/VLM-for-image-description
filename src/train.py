import os
import torch
from tqdm.auto import tqdm

try:
    from comet_ml import Experiment
except ImportError:
    Experiment = None

from torch.optim.lr_scheduler import StepLR

@torch.no_grad()
def evaluate_vlm(model, val_loader, device):
    """Evaluate the model on the validation split and return the average loss."""
    model.eval()

    total_loss = 0.0

    for batch in tqdm(val_loader, desc="Validation", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        total_loss += outputs.loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss


def train(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=3,
    lr=1e-4,
    weight_decay=0.05,
    max_grad_norm=1.0,
    eval_freq=1,
    early_stopping_patience=None,
    early_stopping_min_delta=0.0,
    checkpoint_path="checkpoints/vlm_checkpoint.pt",
    best_model_path="checkpoints/vlm_best.pt",
    comet_project_name=None,
    comet_workspace=None,
    comet_experiment_name=None,
    use_comet=True,
):
    """Train the VLM with checkpointing, validation, and optional Comet logging."""

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    model = model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.75)
    
    start_epoch = 0
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    last_completed_epoch = 0

    experiment = None
    if use_comet and Experiment is not None:
        try:
            experiment = Experiment(
                project_name=comet_project_name,
                workspace=comet_workspace,
                auto_output_logging="simple",
            )

            if comet_experiment_name is not None:
                experiment.set_name(comet_experiment_name)

            experiment.log_parameters({
                "num_epochs": num_epochs,
                "learning_rate": lr,
                "batch_size": train_loader.batch_size,
                "eval_freq": eval_freq,
                "weight_decay": weight_decay,
                "max_grad_norm": max_grad_norm,
                "early_stopping_patience": early_stopping_patience,
                "early_stopping_min_delta": early_stopping_min_delta,
                "optimizer": "AdamW",
                "checkpoint_path": checkpoint_path,
                "best_model_path": best_model_path,
            })
        except Exception as e:
            print(f"Failed to initialize Comet: {e}")
            experiment = None

    if os.path.exists(checkpoint_path):
        print(f"Checkpoint found: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)

            start_epoch = checkpoint.get("epoch", 0)
            train_losses = checkpoint.get("train_losses", [])
            val_losses = checkpoint.get("val_losses", [])
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
            last_completed_epoch = start_epoch

            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer_state = checkpoint.get("optimizer_state_dict")
            if optimizer_state is not None:
                try:
                    optimizer.load_state_dict(optimizer_state)
                except ValueError as e:
                    print(f"Incompatible optimizer state, using a fresh optimizer: {e}")

            scheduler_state = checkpoint.get("scheduler_state_dict")
            if scheduler_state is not None:
                try:
                    scheduler.load_state_dict(scheduler_state)
                except ValueError as e:
                    print(f"Incompatible scheduler state, using a fresh scheduler: {e}")

            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

            print(f"Training resumed from epoch {start_epoch}")

            if experiment is not None:
                experiment.log_other("resumed_from_checkpoint", True)
                experiment.log_other("resume_epoch", start_epoch)

        except Exception as e:
            print(f"Error while loading the checkpoint: {e}")
            print("Training started from scratch.")

    def save_checkpoint(epoch):
        """Persist the current training state to the resume checkpoint file."""
        torch.save(
            {
                "epoch": epoch,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "best_val_loss": best_val_loss,
                "epochs_without_improvement": epochs_without_improvement,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved to: {checkpoint_path}")

    def save_best_model(epoch, val_loss):
        """Save the best-performing model checkpoint according to validation loss."""
        torch.save(
            {
                "epoch": epoch,
                "val_loss": val_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            best_model_path,
        )
        print(f"New best model saved to: {best_model_path} | val_loss={val_loss:.4f}")

    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()

            if hasattr(model, "vision_encoder"):
                model.vision_encoder.eval()

            epoch_loss = 0.0
            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                leave=True,
            )

            for step, batch in enumerate(progress_bar):
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad(set_to_none=True)

                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs.loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
                optimizer.step()

                loss_value = loss.item()
                epoch_loss += loss_value
                current_lr = optimizer.param_groups[0]["lr"]

                progress_bar.set_postfix({
                    "train_loss": f"{loss_value:.4f}",
                    "lr": f"{current_lr:.2e}",
                })

                if experiment is not None:
                    global_step = epoch * len(train_loader) + step
                    experiment.log_metric("train_batch_loss", loss_value, step=global_step)
                    experiment.log_metric("learning_rate", current_lr, step=global_step)

            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")

            if experiment is not None:
                experiment.log_metric("train_epoch_loss", avg_train_loss, epoch=epoch + 1)

            save_checkpoint(epoch + 1)
            last_completed_epoch = epoch + 1

            if (epoch + 1) % eval_freq == 0:
                avg_val_loss = evaluate_vlm(model, val_loader, device)
                val_losses.append(avg_val_loss)

                print(f"Epoch {epoch + 1}/{num_epochs} - Val Loss: {avg_val_loss:.4f}")

                if experiment is not None:
                    experiment.log_metric("val_loss", avg_val_loss, epoch=epoch + 1)

                if avg_val_loss < (best_val_loss - early_stopping_min_delta):
                    best_val_loss = avg_val_loss
                    epochs_without_improvement = 0
                    save_best_model(epoch + 1, avg_val_loss)
                else:
                    epochs_without_improvement += 1

                save_checkpoint(epoch + 1)

                if (
                    early_stopping_patience is not None
                    and epochs_without_improvement >= early_stopping_patience
                ):
                    print(
                        f"Early stopping triggered at epoch {epoch + 1}: "
                        f"no validation loss improvement for {epochs_without_improvement} evaluations."
                    )
                    break

            scheduler.step()


    except KeyboardInterrupt:
        print("\n" + "=" * 70)
        print("Training interrupted manually.")
        save_checkpoint(last_completed_epoch)
        print("Checkpoint saved. You can resume training by running the script again.")
        print("=" * 70)

        if experiment is not None:
            experiment.log_other("interrupted", True)

    finally:
        if experiment is not None:
            experiment.end()

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
    }
