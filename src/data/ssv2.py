import csv
import os
from collections import Counter

import cv2
import torch
from torch.utils.data import Dataset
from transformers import AutoImageProcessor


def read_split_csv(path):
    """
    Read one canonical SSV2 split CSV.

    Returns
    -------
    list[dict]
        Rows exactly as stored in the CSV.
    """
    with open(
        path,
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        return list(csv.DictReader(f))


def load_class_subset(
    num_classes,
    splits_dir,
):
    """
    Load the canonical train/val/test splits and keep the
    `num_classes` most frequent training-set classes.

    This preserves the class-selection logic used in the
    canonical experiment notebook.

    Original label IDs are remapped to:
        0, 1, ..., num_classes - 1

    Parameters
    ----------
    num_classes : int
        Number of classes to retain.

    splits_dir : str or os.PathLike
        Directory containing:
            train.csv
            val.csv
            test.csv

    Returns
    -------
    train_rows, val_rows, test_rows : tuple[list[dict], ...]
    """
    train_rows = read_split_csv(
        os.path.join(
            splits_dir,
            "train.csv",
        )
    )

    val_rows = read_split_csv(
        os.path.join(
            splits_dir,
            "val.csv",
        )
    )

    test_rows = read_split_csv(
        os.path.join(
            splits_dir,
            "test.csv",
        )
    )

    # Canonical notebook behavior:
    # select the most common labels according to TRAIN.
    counts = Counter(
        int(row["label_id"])
        for row in train_rows
    )

    selected = [
        label
        for label, _ in counts.most_common(
            num_classes
        )
    ]

    selected_set = set(selected)

    remap = {
        old: new
        for new, old in enumerate(selected)
    }

    def filter_rows(rows):
        out = []

        for row in rows:
            old_label = int(
                row["label_id"]
            )

            if old_label not in selected_set:
                continue

            new_row = dict(row)

            new_row["label_id"] = (
                remap[old_label]
            )

            out.append(new_row)

        return out

    train_rows = filter_rows(
        train_rows
    )

    val_rows = filter_rows(
        val_rows
    )

    test_rows = filter_rows(
        test_rows
    )

    print(
        "selected original labels:",
        selected,
    )

    return (
        train_rows,
        val_rows,
        test_rows,
    )


def resolve_video_path(
    clips_dir,
    video_id,
):
    """
    Resolve an SSV2 video ID to an extracted .webm or .mp4 file.

    Supports both layouts used during the project:

        clips_dir/123.webm

    and:

        clips_dir/clips/123.webm
    """
    candidates = [
        os.path.join(
            clips_dir,
            f"{video_id}.webm",
        ),
        os.path.join(
            clips_dir,
            f"{video_id}.mp4",
        ),
        os.path.join(
            clips_dir,
            "clips",
            f"{video_id}.webm",
        ),
        os.path.join(
            clips_dir,
            "clips",
            f"{video_id}.mp4",
        ),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


def uniform_frame_indices(
    total_frames,
    num_frames,
):
    """
    Canonical uniform frame sampling used by the current
    experiment notebook.
    """
    return (
        torch.linspace(
            0,
            total_frames - 1,
            steps=num_frames,
        )
        .long()
        .tolist()
    )


class SSV2BaselineDataset(Dataset):
    """
    Canonical SSV2 RGB dataset used by the controlled
    backbone experiments.

    Dataset output contract:

        pixel_values:
            [T, C, H, W]

        label:
            scalar torch.long

    The Hugging Face processor is selected explicitly so
    each backbone can use its own preprocessing.
    """

    def __init__(
        self,
        metadata,
        clips_dir,
        num_frames=8,
        processor_name=(
            "google/vit-base-patch16-224"
        ),
    ):
        self.metadata = metadata
        self.clips_dir = clips_dir
        self.num_frames = num_frames
        self.processor_name = processor_name

        self.processor = (
            AutoImageProcessor.from_pretrained(
                processor_name
            )
        )

    def __len__(self):
        return len(self.metadata)

    def _resolve_video_path(
        self,
        video_id,
    ):
        return resolve_video_path(
            self.clips_dir,
            video_id,
        )

    def _load_video_frames(
        self,
        video_id,
    ):
        video_path = (
            self._resolve_video_path(
                video_id
            )
        )

        if video_path is None:
            return torch.zeros(
                (
                    self.num_frames,
                    3,
                    224,
                    224,
                ),
                dtype=torch.float32,
            )

        cap = cv2.VideoCapture(
            video_path
        )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if total_frames <= 0:
            cap.release()

            return torch.zeros(
                (
                    self.num_frames,
                    3,
                    224,
                    224,
                ),
                dtype=torch.float32,
            )

        indices = uniform_frame_indices(
            total_frames,
            self.num_frames,
        )

        index_set = set(indices)

        frames = []

        for i in range(total_frames):
            ret, frame = cap.read()

            if not ret:
                break

            if i in index_set:
                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                frames.append(frame)

                if (
                    len(frames)
                    == self.num_frames
                ):
                    break

        cap.release()

        if len(frames) == 0:
            frames = [
                torch.zeros(
                    (224, 224, 3),
                    dtype=torch.uint8,
                ).numpy()
            ]

        while (
            len(frames)
            < self.num_frames
        ):
            frames.append(
                frames[-1].copy()
            )

        frames = frames[
            :self.num_frames
        ]

        inputs = self.processor(
            images=frames,
            return_tensors="pt",
        )

        pixel_values = (
            inputs["pixel_values"]
        )

        # Different HF image/video processors
        # return one of:
        #
        # ViT:
        #   [T, C, H, W]
        #
        # TimeSformer / video processors:
        #   [1, T, C, H, W]
        #
        # Every dataset sample must expose:
        #   [T, C, H, W]
        if pixel_values.ndim == 5:
            if pixel_values.size(0) != 1:
                raise ValueError(
                    "Unexpected processor "
                    "output shape: "
                    f"{tuple(pixel_values.shape)}"
                )

            pixel_values = (
                pixel_values.squeeze(0)
            )

        if pixel_values.ndim != 4:
            raise ValueError(
                "Expected processed video "
                "[T,C,H,W], got "
                f"{tuple(pixel_values.shape)}"
            )

        if (
            pixel_values.size(0)
            != self.num_frames
        ):
            raise ValueError(
                "Frame-count mismatch after "
                "preprocessing: "
                f"got {pixel_values.size(0)}, "
                f"expected {self.num_frames}"
            )

        return pixel_values

    def __getitem__(
        self,
        idx,
    ):
        item = self.metadata[idx]

        pixel_values = (
            self._load_video_frames(
                item["id"]
            )
        )

        label_idx = int(
            item["label_id"]
        )

        return (
            pixel_values,
            torch.tensor(
                label_idx,
                dtype=torch.long,
            ),
        )