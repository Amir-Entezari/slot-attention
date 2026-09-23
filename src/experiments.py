import csv
import json
import os
import subprocess

import torch
import torch.nn as nn

from torch.cuda.amp import GradScaler

from .checkpoints import (
    get_checkpoint_model_state,
    load_checkpoint_model_state,
)
from .data import (
    build_dataloaders,
    build_ssv2_datasets,
    load_class_subset,
)
from .evaluation import (
    evaluate,
    evaluate_slot_per_video_init,
    normalize_topk,
)
from .models import (
    build_baseline_model,
    build_slot_model,
    get_baseline_processor_name,
    get_slot_processor_name,
)
from .training import (
    train_one_epoch,
)
from .utils import (
    seed_everything,
)

def _get_git_commit():
    try:
        repo_root = os.path.dirname(
            os.path.dirname(__file__)
        )

        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
        ).strip()

    except Exception:
        return None


def _append_history_row(
    path,
    row,
):
    file_exists = os.path.exists(path)

    with open(
        path,
        "a",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(row.keys()),
        )

        if (
            not file_exists
            or os.path.getsize(path) == 0
        ):
            writer.writeheader()

        writer.writerow(row)

def resolve_selection_metric(
    selection_metric,
    topk_values,
):
    """
    Resolve the validation metric used for checkpoint
    selection.

    If selection_metric is None, reproduce the historical
    notebook behavior:

        [1]       -> top1
        [1, 3]    -> top3
        [1, 5]    -> top5

    For new experiments, an explicit value such as
    "top1" is preferred.
    """

    if selection_metric is None:
        return (
            f"top{max(topk_values)}"
        )

    if isinstance(
        selection_metric,
        int,
    ):
        selection_metric = (
            f"top{selection_metric}"
        )

    selection_metric = str(
        selection_metric
    ).lower()

    valid_metrics = {
        f"top{k}"
        for k in topk_values
    }

    if (
        selection_metric
        not in valid_metrics
    ):
        raise ValueError(
            "selection_metric must be one "
            "of the reported metrics. "
            f"Got {selection_metric}; "
            f"available: "
            f"{sorted(valid_metrics)}"
        )

    return selection_metric


def run_baseline_experiment(
    *,
    model_name="vit",
    num_classes=5,
    topk=1,
    selection_metric=None,
    num_frames=8,
    batch_size=4,
    epochs=5,
    learning_rate=1e-4,
    num_workers=2,
    seed=42,
    resume=True,
    save_every=1,
    evaluate_test=True,
    splits_dir,
    clips_dir,
    checkpoint_root="checkpoints",
    device=None,
):
    """
    Run a canonical baseline experiment.

    This is the clean replacement for the historical
    notebook's run_experiment().

    Parameters
    ----------
    topk:
        Metrics to report.

        Examples:
            1
            3
            [1, 3, 5]

    selection_metric:
        Validation metric used to select the best checkpoint.

        None reproduces the historical behavior of selecting
        by the largest reported k.

        Examples:
            None
            "top1"
            "top3"
            1
            3
    """

    # ========================================================
    # Basic setup
    # ========================================================

    if epochs < 1:
        raise ValueError(
            "epochs must be >= 1"
        )

    if save_every < 1:
        raise ValueError(
            "save_every must be >= 1"
        )

    seed_everything(
        seed
    )

    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:
        device = torch.device(
            device
        )

    topk_values = normalize_topk(
        topk,
        num_classes,
    )

    best_key = (
        resolve_selection_metric(
            selection_metric,
            topk_values,
        )
    )

    processor_name = (
        get_baseline_processor_name(
            model_name
        )
    )

    # ========================================================
    # Experiment / checkpoint paths
    # ========================================================

    experiment_name = (
        f"baseline_{model_name}"
        f"_classes{num_classes}"
        f"_frames{num_frames}"
        f"_seed{seed}"
    )

    output_dir = os.path.join(
        checkpoint_root,
        experiment_name,
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )
    history_path = os.path.join(
        output_dir,
        "history.csv",
    )

    metrics_path = os.path.join(
        output_dir,
        "metrics.json",
    )

    git_commit = _get_git_commit()


    if not resume:
        for path in [
            history_path,
            metrics_path,
        ]:
            if os.path.exists(path):
                os.remove(path)
                
    latest_checkpoint_path = (
        os.path.join(
            output_dir,
            "latest.pt",
        )
    )

    best_checkpoint_path = (
        os.path.join(
            output_dir,
            "best.pt",
        )
    )

    print()
    print(
        "=" * 60
    )
    print(
        "Experiment:",
        experiment_name,
    )
    print(
        "Output directory:",
        output_dir,
    )
    print(
        "Device:",
        device,
    )
    print(
        "Processor:",
        processor_name,
    )
    print(
        "Checkpoint selection:",
        best_key,
    )
    print(
        "=" * 60
    )

    # ========================================================
    # Dataset
    # ========================================================

    (
        train_metadata,
        val_metadata,
        test_metadata,
    ) = load_class_subset(
        num_classes,
        splits_dir=splits_dir,
    )

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = build_ssv2_datasets(
        train_metadata,
        val_metadata,
        test_metadata,
        clips_dir=clips_dir,
        num_frames=num_frames,
        processor_name=processor_name,
    )

    (
        train_loader,
        val_loader,
        test_loader,
    ) = build_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )

    # ========================================================
    # Model / optimizer
    # ========================================================

    model = build_baseline_model(
        model_name=model_name,
        num_classes=num_classes,
        num_frames=num_frames,
    ).to(
        device
    )

    criterion = (
        nn.CrossEntropyLoss()
    )

    trainable_params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
    )

    scaler = GradScaler(
        enabled=(
            device.type == "cuda"
        )
    )

    # ========================================================
    # Training state
    # ========================================================

    start_epoch = 1

    best_val = -float(
        "inf"
    )

    best_epoch = None

    # ========================================================
    # Resume
    # ========================================================

    if (
        resume
        and os.path.exists(
            latest_checkpoint_path
        )
    ):
        print()
        print(
            "[RESUME] Loading checkpoint:"
        )
        print(
            latest_checkpoint_path
        )

        checkpoint = torch.load(
            latest_checkpoint_path,
            map_location=device,
        )

        # ----------------------------------------------------
        # Protect against accidentally resuming an experiment
        # under a different checkpoint-selection policy.
        #
        # Old notebook checkpoints do not contain
        # "selection_metric", so infer their historical policy
        # from topk_values.
        # ----------------------------------------------------

        checkpoint_topk = checkpoint.get(
            "topk_values",
            topk_values,
        )

        checkpoint_selection = (
            checkpoint.get(
                "selection_metric",
                f"top{max(checkpoint_topk)}",
            )
        )

        if (
            checkpoint_selection
            != best_key
        ):
            raise ValueError(
                "Checkpoint selection metric "
                "does not match the current "
                "experiment.\n"
                f"Checkpoint: "
                f"{checkpoint_selection}\n"
                f"Current: {best_key}"
            )

        load_checkpoint_model_state(
            model,
            checkpoint[
                "model_state_dict"
            ],
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        if (
            "scaler_state_dict"
            in checkpoint
        ):
            scaler.load_state_dict(
                checkpoint[
                    "scaler_state_dict"
                ]
            )

        start_epoch = (
            checkpoint["epoch"]
            + 1
        )

        best_val = checkpoint.get(
            "best_val",
            -float("inf"),
        )

        best_epoch = checkpoint.get(
            "best_epoch",
            None,
        )

        print(
            "[RESUME] Last completed "
            f"epoch: {checkpoint['epoch']}"
        )

        print(
            "[RESUME] Continuing from "
            f"epoch: {start_epoch}"
        )

        print(
            f"[RESUME] Best {best_key} "
            f"so far: {best_val:.4f} "
            f"(epoch {best_epoch})"
        )

    # ========================================================
    # Experiment summary
    # ========================================================

    print()
    print(
        "Starting Training Pipeline..."
    )

    print(
        "MODEL:",
        model_name,
    )

    print(
        "classes:",
        num_classes,
    )

    print(
        "train samples:",
        len(train_dataset),
    )

    print(
        "val samples:",
        len(val_dataset),
    )

    print(
        "test samples:",
        len(test_dataset),
    )

    print(
        "reported top-k:",
        topk_values,
    )

    print(
        "selection metric:",
        best_key,
    )

    print(
        "target epochs:",
        epochs,
    )

    print(
        "starting epoch:",
        start_epoch,
    )

    # ========================================================
    # Epoch loop
    # ========================================================

    for epoch in range(
        start_epoch,
        epochs + 1,
    ):

        (
            train_loss,
            train_top1,
        ) = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )

        (
            val_loss,
            val_metrics,
        ) = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            topk_values=topk_values,
            device=device,
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if (
            val_metrics[best_key]
            > best_val
        ):
            best_val = (
                val_metrics[
                    best_key
                ]
            )

            best_epoch = epoch

            torch.save(
                {
                    "epoch":
                        epoch,

                    "model_state_dict":
                        get_checkpoint_model_state(
                            model,
                            model_name,
                        ),

                    "best_val":
                        best_val,

                    "best_epoch":
                        best_epoch,

                    # Experiment metadata
                    "model_name":
                        model_name,

                    "num_classes":
                        num_classes,

                    "topk_values":
                        topk_values,

                    "selection_metric":
                        best_key,

                    "num_frames":
                        num_frames,

                    "batch_size":
                        batch_size,

                    "learning_rate":
                        learning_rate,

                    "seed":
                        seed,

                    "processor_name":
                        processor_name,
                },
                best_checkpoint_path,
            )

            print(
                "[BEST CHECKPOINT SAVED] "
                f"epoch={epoch} "
                f"{best_key}="
                f"{best_val:.4f}"
            )

        # ----------------------------------------------------
        # Epoch log
        # ----------------------------------------------------

        metrics_str = " ".join(
            f"{key}={value:.4f}"
            for key, value
            in val_metrics.items()
        )


        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_top1": train_top1,
            "val_loss": val_loss,
        }

        history_row.update(
            {
                f"val_{key}": value
                for key, value
                in val_metrics.items()
            }
        )

        _append_history_row(
            history_path,
            history_row,
        )
        print(
            f"epoch={epoch:03d} "
            f"train_loss="
            f"{train_loss:.4f} "
            f"train_top1="
            f"{train_top1:.4f} "
            f"val_loss="
            f"{val_loss:.4f} "
            f"{metrics_str} "
            f"best_{best_key}="
            f"{best_val:.4f}"
        )

        # ----------------------------------------------------
        # Latest checkpoint
        # ----------------------------------------------------

        if (
            epoch
            % save_every
            == 0
        ):
            torch.save(
                {
                    "epoch":
                        epoch,

                    "model_state_dict":
                        get_checkpoint_model_state(
                            model,
                            model_name,
                        ),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "scaler_state_dict":
                        scaler.state_dict(),

                    "best_val":
                        best_val,

                    "best_epoch":
                        best_epoch,

                    # Experiment metadata
                    "model_name":
                        model_name,

                    "num_classes":
                        num_classes,

                    "topk_values":
                        topk_values,

                    "selection_metric":
                        best_key,

                    "num_frames":
                        num_frames,

                    "batch_size":
                        batch_size,

                    "learning_rate":
                        learning_rate,

                    "seed":
                        seed,

                    "processor_name":
                        processor_name,
                },
                latest_checkpoint_path,
            )

            print(
                "[CHECKPOINT SAVED] "
                f"epoch={epoch} -> "
                f"{latest_checkpoint_path}"
            )

    # ========================================================
    # Restore validation-selected best checkpoint
    # ========================================================

    if os.path.exists(
        best_checkpoint_path
    ):
        best_checkpoint = torch.load(
            best_checkpoint_path,
            map_location=device,
        )

        load_checkpoint_model_state(
            model,
            best_checkpoint[
                "model_state_dict"
            ],
        )

        best_val = (
            best_checkpoint[
                "best_val"
            ]
        )

        best_epoch = (
            best_checkpoint[
                "best_epoch"
            ]
        )

    else:
        raise RuntimeError(
            "Best checkpoint was not found."
        )

    print()
    print(
        f"[BEST] epoch={best_epoch} "
        f"{best_key}="
        f"{best_val:.4f}"
    )

    # ========================================================
    # Re-evaluate selected checkpoint on validation
    # ========================================================

    (
        best_val_loss,
        best_val_metrics,
    ) = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        topk_values=topk_values,
        device=device,
    )

    best_val_metrics_str = (
        " ".join(
            f"{key}={value:.4f}"
            for key, value
            in best_val_metrics.items()
        )
    )

    print(
        "[VAL@BEST] "
        f"loss={best_val_loss:.4f} "
        f"{best_val_metrics_str}"
    )

    # ========================================================
    # Held-out test
    # ========================================================
    test_loss = None
    test_metrics = None
    
    if evaluate_test:

        (
            test_loss,
            test_metrics,
        ) = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            topk_values=topk_values,
            device=device,
        )

        test_metrics_str = (
            " ".join(
                f"{key}={value:.4f}"
                for key, value
                in test_metrics.items()
            )
        )

        print(
            "[TEST] "
            f"loss={test_loss:.4f} "
            f"{test_metrics_str}"
        )

    else:
        print(
            "[TEST] skipped "
            "(validation-only experiment)"
        )
    summary = {
        "run_id": experiment_name,
        "git_commit": git_commit,

        "model_name": model_name,
        "num_classes": num_classes,
        "num_frames": num_frames,
        "seed": seed,

        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,

        "topk_values": topk_values,
        "selection_metric": best_key,

        "best_epoch": best_epoch,
        "best_val": float(best_val),
        "best_val_loss": float(
            best_val_loss
        ),
        "best_val_metrics": {
            key: float(value)
            for key, value
            in best_val_metrics.items()
        },

        "test_loss": (
            None
            if test_loss is None
            else float(test_loss)
        ),

        "test_metrics": (
            None
            if test_metrics is None
            else {
                key: float(value)
                for key, value
                in test_metrics.items()
            }
        ),
    }


    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )


    print(
        "[ARTIFACTS]",
        history_path,
        metrics_path,
    )
    return model



def run_slot_experiment(
    *,
    model_name="vit",
    num_classes=5,
    topk=1,
    selection_metric=None,
    num_frames=8,
    batch_size=4,
    epochs=5,
    learning_rate=1e-4,
    num_workers=2,
    seed=42,
    num_slots=4,
    slot_dim=128,
    slot_iters=3,
    eval_seed=0,
    slot_eval_mode="legacy_shared",
    slot_eval_seed=12345,
    resume=True,
    save_every=1,
    evaluate_test=True,
    splits_dir,
    clips_dir,
    checkpoint_root="checkpoints",
    device=None,
):
    """
    Run a canonical Slot Attention experiment.

    Evaluation modes
    ----------------
    legacy_shared:
        Reproduce the historical evaluation behavior.
        SlotAttention uses its deterministic internal
        _eval_noise, shared across examples.

    per_video_deterministic:
        Use an independent deterministic initialization
        for every validation/test example.

        The initialization bank is generated from
        slot_eval_seed.
    """

    # ========================================================
    # Basic validation
    # ========================================================

    if epochs < 1:
        raise ValueError(
            "epochs must be >= 1"
        )

    if save_every < 1:
        raise ValueError(
            "save_every must be >= 1"
        )

    valid_eval_modes = {
        "legacy_shared",
        "per_video_deterministic",
    }

    if (
        slot_eval_mode
        not in valid_eval_modes
    ):
        raise ValueError(
            "slot_eval_mode must be one of "
            f"{sorted(valid_eval_modes)}. "
            f"Got: {slot_eval_mode}"
        )

    if num_slots < 1:
        raise ValueError(
            "num_slots must be >= 1"
        )

    if slot_dim < 1:
        raise ValueError(
            "slot_dim must be >= 1"
        )

    if slot_iters < 1:
        raise ValueError(
            "slot_iters must be >= 1"
        )

    # The migrated historical layer3 wrapper does not
    # accept external initial_slots.
    if (
        model_name
        == "r3d_18_layer3"
        and slot_eval_mode
        == "per_video_deterministic"
    ):
        raise ValueError(
            "r3d_18_layer3 does not support "
            "per_video_deterministic evaluation "
            "in the canonical migrated wrapper."
        )

    # ========================================================
    # Reproducibility / device
    # ========================================================

    seed_everything(
        seed
    )

    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:
        device = torch.device(
            device
        )

    topk_values = normalize_topk(
        topk,
        num_classes,
    )

    best_key = (
        resolve_selection_metric(
            selection_metric,
            topk_values,
        )
    )

    processor_name = (
        get_slot_processor_name(
            model_name,
            slot_eval_mode=(
                slot_eval_mode
            ),
        )
    )

    # ========================================================
    # Experiment paths
    # ========================================================

    experiment_name = (
        f"slot_{model_name}"
        f"_classes{num_classes}"
        f"_frames{num_frames}"
        f"_K{num_slots}"
        f"_D{slot_dim}"
        f"_I{slot_iters}"
        f"_{slot_eval_mode}"
        f"_seed{seed}"
    )

    output_dir = os.path.join(
        checkpoint_root,
        experiment_name,
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    latest_checkpoint_path = (
        os.path.join(
            output_dir,
            "latest.pt",
        )
    )

    best_checkpoint_path = (
        os.path.join(
            output_dir,
            "best.pt",
        )
    )

    print()
    print(
        "=" * 60
    )

    print(
        "Experiment:",
        experiment_name,
    )

    print(
        "Output directory:",
        output_dir,
    )

    print(
        "Device:",
        device,
    )

    print(
        "Processor:",
        processor_name,
    )

    print(
        "Slot evaluation mode:",
        slot_eval_mode,
    )

    if (
        slot_eval_mode
        == "legacy_shared"
    ):
        print(
            "Legacy Slot eval seed:",
            eval_seed,
        )

    else:
        print(
            "Per-video Slot eval seed:",
            slot_eval_seed,
        )

    print(
        "Checkpoint selection:",
        best_key,
    )

    print(
        "=" * 60
    )

    # ========================================================
    # Dataset
    # ========================================================

    (
        train_metadata,
        val_metadata,
        test_metadata,
    ) = load_class_subset(
        num_classes,
        splits_dir=splits_dir,
    )

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = build_ssv2_datasets(
        train_metadata,
        val_metadata,
        test_metadata,
        clips_dir=clips_dir,
        num_frames=num_frames,
        processor_name=processor_name,
    )

    (
        train_loader,
        val_loader,
        test_loader,
    ) = build_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )

    # ========================================================
    # Model / optimizer
    # ========================================================

    model = build_slot_model(
        model_name=model_name,
        num_classes=num_classes,
        num_frames=num_frames,
        num_slots=num_slots,
        slot_dim=slot_dim,
        slot_iters=slot_iters,
        eval_seed=eval_seed,
    ).to(
        device
    )

    criterion = (
        nn.CrossEntropyLoss()
    )

    trainable_params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
    )

    scaler = GradScaler(
        enabled=(
            device.type == "cuda"
        )
    )

    # ========================================================
    # Slot evaluator
    # ========================================================

    def evaluate_current(
        loader,
    ):
        if (
            slot_eval_mode
            == "legacy_shared"
        ):
            return evaluate(
                model=model,
                loader=loader,
                criterion=criterion,
                topk_values=topk_values,
                device=device,
            )

        return (
            evaluate_slot_per_video_init(
                model=model,
                loader=loader,
                criterion=criterion,
                topk_values=topk_values,
                device=device,
                init_seed=(
                    slot_eval_seed
                ),
            )
        )

    # ========================================================
    # Training state
    # ========================================================

    start_epoch = 1

    best_val = -float(
        "inf"
    )

    best_epoch = None

    # ========================================================
    # Resume
    # ========================================================

    if (
        resume
        and os.path.exists(
            latest_checkpoint_path
        )
    ):
        print()
        print(
            "[RESUME] Loading checkpoint:"
        )

        print(
            latest_checkpoint_path
        )

        checkpoint = torch.load(
            latest_checkpoint_path,
            map_location=device,
        )

        checkpoint_topk = (
            checkpoint.get(
                "topk_values",
                topk_values,
            )
        )

        checkpoint_selection = (
            checkpoint.get(
                "selection_metric",
                f"top{max(checkpoint_topk)}",
            )
        )

        if (
            checkpoint_selection
            != best_key
        ):
            raise ValueError(
                "Checkpoint selection metric "
                "does not match the current "
                "experiment.\n"
                f"Checkpoint: "
                f"{checkpoint_selection}\n"
                f"Current: {best_key}"
            )

        # ----------------------------------------------------
        # Protect the scientifically important Slot settings.
        #
        # Only check a field when it exists so that older
        # historical checkpoints remain loadable.
        # ----------------------------------------------------

        expected_slot_config = {
            "num_slots":
                num_slots,

            "slot_dim":
                slot_dim,

            "slot_iters":
                slot_iters,

            "eval_seed":
                eval_seed,

            "slot_eval_mode":
                slot_eval_mode,

            "slot_eval_seed":
                slot_eval_seed,
        }

        for (
            key,
            expected_value,
        ) in expected_slot_config.items():

            if (
                key in checkpoint
                and checkpoint[key]
                != expected_value
            ):
                raise ValueError(
                    "Checkpoint configuration "
                    "does not match the current "
                    "experiment.\n"
                    f"{key}: checkpoint="
                    f"{checkpoint[key]!r}, "
                    f"current="
                    f"{expected_value!r}"
                )

        load_checkpoint_model_state(
            model,
            checkpoint[
                "model_state_dict"
            ],
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        if (
            "scaler_state_dict"
            in checkpoint
        ):
            scaler.load_state_dict(
                checkpoint[
                    "scaler_state_dict"
                ]
            )

        start_epoch = (
            checkpoint["epoch"]
            + 1
        )

        best_val = checkpoint.get(
            "best_val",
            -float("inf"),
        )

        best_epoch = checkpoint.get(
            "best_epoch",
            None,
        )

        print(
            "[RESUME] Last completed "
            f"epoch: "
            f"{checkpoint['epoch']}"
        )

        print(
            "[RESUME] Continuing from "
            f"epoch: {start_epoch}"
        )

        print(
            f"[RESUME] Best {best_key} "
            f"so far: {best_val:.4f} "
            f"(epoch {best_epoch})"
        )

    # ========================================================
    # Experiment summary
    # ========================================================

    print()
    print(
        "Starting Slot Training Pipeline..."
    )

    print(
        "MODEL:",
        model_name,
    )

    print(
        "classes:",
        num_classes,
    )

    print(
        "train samples:",
        len(train_dataset),
    )

    print(
        "val samples:",
        len(val_dataset),
    )

    print(
        "test samples:",
        len(test_dataset),
    )

    print(
        "num slots:",
        num_slots,
    )

    print(
        "slot dim:",
        slot_dim,
    )

    print(
        "slot iterations:",
        slot_iters,
    )

    print(
        "reported top-k:",
        topk_values,
    )

    print(
        "selection metric:",
        best_key,
    )

    print(
        "evaluation mode:",
        slot_eval_mode,
    )

    print(
        "target epochs:",
        epochs,
    )

    print(
        "starting epoch:",
        start_epoch,
    )

    # ========================================================
    # Epoch loop
    # ========================================================

    for epoch in range(
        start_epoch,
        epochs + 1,
    ):

        (
            train_loss,
            train_top1,
        ) = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )

        (
            val_loss,
            val_metrics,
        ) = evaluate_current(
            val_loader
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if (
            val_metrics[best_key]
            > best_val
        ):
            best_val = (
                val_metrics[
                    best_key
                ]
            )

            best_epoch = epoch

            torch.save(
                {
                    "epoch":
                        epoch,

                    "model_state_dict":
                        get_checkpoint_model_state(
                            model,
                            model_name,
                        ),

                    "best_val":
                        best_val,

                    "best_epoch":
                        best_epoch,

                    # Experiment metadata
                    "model_name":
                        model_name,

                    "num_classes":
                        num_classes,

                    "topk_values":
                        topk_values,

                    "selection_metric":
                        best_key,

                    "num_frames":
                        num_frames,

                    "batch_size":
                        batch_size,

                    "learning_rate":
                        learning_rate,

                    "seed":
                        seed,

                    "processor_name":
                        processor_name,

                    # Slot configuration
                    "num_slots":
                        num_slots,

                    "slot_dim":
                        slot_dim,

                    "slot_iters":
                        slot_iters,

                    "eval_seed":
                        eval_seed,

                    "slot_eval_mode":
                        slot_eval_mode,

                    "slot_eval_seed":
                        slot_eval_seed,
                },
                best_checkpoint_path,
            )

            print(
                "[BEST CHECKPOINT SAVED] "
                f"epoch={epoch} "
                f"{best_key}="
                f"{best_val:.4f}"
            )

        # ----------------------------------------------------
        # Epoch log
        # ----------------------------------------------------

        metrics_str = " ".join(
            f"{key}={value:.4f}"
            for key, value
            in val_metrics.items()
        )

        print(
            f"epoch={epoch:03d} "
            f"train_loss="
            f"{train_loss:.4f} "
            f"train_top1="
            f"{train_top1:.4f} "
            f"val_loss="
            f"{val_loss:.4f} "
            f"{metrics_str} "
            f"best_{best_key}="
            f"{best_val:.4f}"
        )

        # ----------------------------------------------------
        # Latest checkpoint
        # ----------------------------------------------------

        if (
            epoch
            % save_every
            == 0
        ):
            torch.save(
                {
                    "epoch":
                        epoch,

                    "model_state_dict":
                        get_checkpoint_model_state(
                            model,
                            model_name,
                        ),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "scaler_state_dict":
                        scaler.state_dict(),

                    "best_val":
                        best_val,

                    "best_epoch":
                        best_epoch,

                    # Experiment metadata
                    "model_name":
                        model_name,

                    "num_classes":
                        num_classes,

                    "topk_values":
                        topk_values,

                    "selection_metric":
                        best_key,

                    "num_frames":
                        num_frames,

                    "batch_size":
                        batch_size,

                    "learning_rate":
                        learning_rate,

                    "seed":
                        seed,

                    "processor_name":
                        processor_name,

                    # Slot configuration
                    "num_slots":
                        num_slots,

                    "slot_dim":
                        slot_dim,

                    "slot_iters":
                        slot_iters,

                    "eval_seed":
                        eval_seed,

                    "slot_eval_mode":
                        slot_eval_mode,

                    "slot_eval_seed":
                        slot_eval_seed,
                },
                latest_checkpoint_path,
            )

            print(
                "[CHECKPOINT SAVED] "
                f"epoch={epoch} -> "
                f"{latest_checkpoint_path}"
            )

    # ========================================================
    # Restore validation-selected best checkpoint
    # ========================================================

    if os.path.exists(
        best_checkpoint_path
    ):
        best_checkpoint = torch.load(
            best_checkpoint_path,
            map_location=device,
        )

        load_checkpoint_model_state(
            model,
            best_checkpoint[
                "model_state_dict"
            ],
        )

        best_val = (
            best_checkpoint[
                "best_val"
            ]
        )

        best_epoch = (
            best_checkpoint[
                "best_epoch"
            ]
        )

    else:
        raise RuntimeError(
            "Best checkpoint was not found."
        )

    print()
    print(
        f"[BEST] epoch={best_epoch} "
        f"{best_key}="
        f"{best_val:.4f}"
    )

    # ========================================================
    # Re-evaluate selected checkpoint on validation
    # ========================================================

    (
        best_val_loss,
        best_val_metrics,
    ) = evaluate_current(
        val_loader
    )

    best_val_metrics_str = (
        " ".join(
            f"{key}={value:.4f}"
            for key, value
            in best_val_metrics.items()
        )
    )

    print(
        "[VAL@BEST] "
        f"loss={best_val_loss:.4f} "
        f"{best_val_metrics_str}"
    )

    # ========================================================
    # Held-out test
    # ========================================================

    if evaluate_test:

        (
            test_loss,
            test_metrics,
        ) = evaluate_current(
            test_loader
        )

        test_metrics_str = (
            " ".join(
                f"{key}={value:.4f}"
                for key, value
                in test_metrics.items()
            )
        )

        print(
            "[TEST] "
            f"loss={test_loss:.4f} "
            f"{test_metrics_str}"
        )

    else:
        print(
            "[TEST] skipped "
            "(validation-only experiment)"
        )

    return model