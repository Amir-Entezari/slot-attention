import torch

from torch.cuda.amp import autocast


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
):
    """
    Train a standard classification model for one epoch.

    This reproduces the training loop used by the canonical
    baseline and vanilla Slot Attention experiments.

    Returns
    -------
    train_loss : float
        Sample-weighted average training loss.

    train_top1 : float
        Top-1 training accuracy in [0, 1].
    """

    device = torch.device(
        device
    )

    model.train()

    total = 0
    total_loss = 0.0
    top1_correct = 0

    for videos, labels in loader:

        videos = videos.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with autocast(
            enabled=(
                device.type == "cuda"
            )
        ):
            outputs = model(
                videos
            )

            loss = criterion(
                outputs,
                labels,
            )

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        batch_size = labels.size(0)

        total += batch_size

        total_loss += (
            loss.item()
            * batch_size
        )

        top1_correct += (
            outputs
            .argmax(dim=1)
            .eq(labels)
            .sum()
            .item()
        )

    train_loss = (
        total_loss
        / max(1, total)
    )

    train_top1 = (
        top1_correct
        / max(1, total)
    )

    return (
        train_loss,
        train_top1,
    )