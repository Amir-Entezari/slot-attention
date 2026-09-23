import torch
from torch.utils.data import DataLoader


def build_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size=4,
    num_workers=2,
    seed=42,
):
    """
    Build the canonical train/validation/test DataLoaders.

    Behavior matches the current experiment notebook:

    - training is shuffled
    - validation and test are not shuffled
    - training shuffle uses a seeded torch.Generator
    - drop_last=False for all splits
    - pin_memory is enabled when CUDA is available
    """

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        generator=loader_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )
    

from .ssv2 import SSV2BaselineDataset


def build_ssv2_datasets(
    train_metadata,
    val_metadata,
    test_metadata,
    clips_dir,
    num_frames,
    processor_name,
):
    """
    Build the canonical SSV2 train/validation/test datasets
    using the same preprocessing configuration.
    """

    train_dataset = SSV2BaselineDataset(
        train_metadata,
        clips_dir,
        num_frames=num_frames,
        processor_name=processor_name,
    )

    val_dataset = SSV2BaselineDataset(
        val_metadata,
        clips_dir,
        num_frames=num_frames,
        processor_name=processor_name,
    )

    test_dataset = SSV2BaselineDataset(
        test_metadata,
        clips_dir,
        num_frames=num_frames,
        processor_name=processor_name,
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )