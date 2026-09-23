from pathlib import Path
import shutil

import kagglehub


def sync_from_kaggle(
    *,
    handle,
    local_root,
):
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
        "missing files."
    )


def backup_to_kaggle(
    *,
    handle,
    local_root,
    version_notes,
):
    local_root = Path(local_root)

    # Merge old remote files first so uploading
    # a new version does not delete older runs.
    sync_from_kaggle(
        handle=handle,
        local_root=local_root,
    )

    kagglehub.dataset_upload(
        handle,
        str(local_root),
        version_notes=version_notes,
    )

    print(
        "[BACKUP] Upload submitted."
    )