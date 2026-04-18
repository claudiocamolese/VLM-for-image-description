import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

try:
    from torchvision import transforms
except ImportError:
    transforms = None


def _resolve_image_size(image_processor):
    """Infer the image size expected by the processor configuration."""
    size = getattr(image_processor, "size", 224)

    if isinstance(size, dict):
        if "height" in size and "width" in size:
            return (size["height"], size["width"])
        if "shortest_edge" in size:
            edge = size["shortest_edge"]
            return (edge, edge)

    if isinstance(size, int):
        return (size, size)

    if isinstance(size, (tuple, list)) and len(size) == 2:
        return tuple(size)

    return (224, 224)

class Flickr8kDataset(Dataset):
    """Dataset wrapper for Flickr8k image-caption samples."""
    
    def __init__(
        self,
        hf_dataset,
        image_processor,
        tokenizer,
        max_length=64,
        is_train=False,
        use_augmentation=False,
        deterministic_caption_idx=0,
    ):
        """Initialize the dataset wrapper and optional train-time augmentation."""
        self.dataset = hf_dataset
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        self.deterministic_caption_idx = deterministic_caption_idx
        self.image_transforms = None

        if is_train and use_augmentation and transforms is not None:
            image_size = _resolve_image_size(image_processor)
            self.image_transforms = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            ])
        elif is_train and use_augmentation:
            print("torchvision is not available: training augmentation has been disabled.")

    def __len__(self):
        """Return the number of samples in the wrapped dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Load, preprocess, and tokenize a single Flickr8k example."""
        item = self.dataset[idx]

        image = item["image"].convert("RGB")
        if self.image_transforms is not None:
            image = self.image_transforms(image)
        pixel_values = self.image_processor(
            image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        if self.is_train:
            caption_idx = random.randint(0, 4)
        else:
            caption_idx = self.deterministic_caption_idx

        caption = item[f"caption_{caption_idx}"]

        eos_token_id = self.tokenizer.eos_token_id
        max_text_length = self.max_length - 1 if eos_token_id is not None else self.max_length

        encoding = self.tokenizer(
            caption,
            max_length=max_text_length,
            truncation=True,
            add_special_tokens=False,
        )

        input_ids = encoding["input_ids"]
        if eos_token_id is not None:
            input_ids = input_ids + [eos_token_id]

        attention_mask = [1] * len(input_ids)

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("The tokenizer must define a pad_token_id for batching.")

        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def create_flickr8k_loaders(
    image_processor,
    tokenizer,
    batch_size,
    max_length=64,
    num_workers=0,
    val_split=0.1,
    test_split=0.1,
    data_dir="./dataset",
    use_train_augmentation=True,
):
    """Create train, validation, and test dataloaders for Flickr8k."""

    os.makedirs(data_dir, exist_ok=True)

    dataset = load_dataset(
        "jxie/flickr8k",
        split="train",
        cache_dir=data_dir
    )

    dataset = dataset.shuffle(seed=42)

    test_size = int(len(dataset) * test_split)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size - test_size

    train_data = dataset.select(range(0, train_size))
    val_data = dataset.select(range(train_size, train_size + val_size))
    test_data = dataset.select(range(train_size + val_size, len(dataset)))

    train_dataset = Flickr8kDataset(
        train_data,
        image_processor,
        tokenizer,
        max_length=max_length,
        is_train=True,
        use_augmentation=use_train_augmentation,
    )
    val_dataset = Flickr8kDataset(
        val_data,
        image_processor,
        tokenizer,
        max_length=max_length,
        is_train=False,
    )
    test_dataset = Flickr8kDataset(
        test_data,
        image_processor,
        tokenizer,
        max_length=max_length,
        is_train=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
