import json
from pathlib import Path

import pandas as pd


def _normalize_template(template):
    return str(template).replace("[something]", "something").strip()


def _build_split(annotation_path, class_names):
    with open(annotation_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    df = pd.DataFrame(annotations)

    df["label_name"] = df["template"].map(_normalize_template)

    class_to_id = {
        name: idx
        for idx, name in enumerate(class_names)
    }

    df = df[
        df["label_name"].isin(class_to_id)
    ].copy()

    df["video_id"] = df["id"].astype(str)
    df["label_id"] = (
        df["label_name"]
        .map(class_to_id)
        .astype(int)
    )

    return df[
        ["video_id", "label_id", "label_name"]
    ].reset_index(drop=True)


def create_class_split(
    train_json,
    val_json,
    class_names,
    output_dir=None,
):
    train_df = _build_split(
        train_json,
        class_names,
    )

    val_df = _build_split(
        val_json,
        class_names,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        train_df.to_csv(
            output_dir / "train.csv",
            index=False,
        )

        val_df.to_csv(
            output_dir / "val.csv",
            index=False,
        )

    return train_df, val_df