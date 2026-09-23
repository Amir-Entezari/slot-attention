from pathlib import Path
import shutil

import kagglehub


def _dataset_handle(
    username,
    dataset_name="slot-attention-checkpoints",
):
    return f"{username}/{dataset_name}"


def sync_from_kaggle(
    *,
    username,
    local_root,
    dataset_name="slot-attention-checkpoints",
):
    handle = _dataset_handle(
        username,
        dataset_name,
    )

    local_root = Path(local_root)
    local_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        remote_root = Path(
            kagglehub.dataset_download(
                handle
            )
        )

    except Exception as exc:
        print(
            "[SYNC] No existing remote "
            f"dataset loaded: {exc}"
        )
        return

    copied = 0

    for source in remote_root.rglob("*"):
        if not source.is_file():
            continue

        relative = source.relative_to(
            remote_root
        )

        destination = (
            local_root / relative
        )

        # Local files win.
        if destination.exists():
            continue

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            source,
            destination,
        )

        copied += 1

    print(
        f"[SYNC] Restored {copied} "
        f"missing files from {handle}."
    )


def backup_to_kaggle(
    *,
    username,
    local_root,
    version_notes,
    dataset_name="slot-attention-checkpoints",
):
    handle = _dataset_handle(
        username,
        dataset_name,
    )

    # Merge existing remote files first.
    sync_from_kaggle(
        username=username,
        dataset_name=dataset_name,
        local_root=local_root,
    )

    kagglehub.dataset_upload(
        handle,
        str(local_root),
        version_notes=version_notes,
    )

    print(
        f"[BACKUP] Uploaded to {handle}."
    )