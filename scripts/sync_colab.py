from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]

OUTPUT = ROOT / "colab_code.zip"


INCLUDE = [
    "src",
    "scripts",
    "requirements.txt",
]


EXCLUDE_PARTS = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".ipynb_checkpoints",
}


def should_include(path: Path):
    parts = set(path.parts)

    if parts & EXCLUDE_PARTS:
        return False

    return True


def create_zip():

    if OUTPUT.exists():
        OUTPUT.unlink()

    with zipfile.ZipFile(
        OUTPUT,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as z:

        for item in INCLUDE:

            path = ROOT / item

            if path.is_file():

                z.write(
                    path,
                    path.relative_to(ROOT),
                )

            else:

                for file in path.rglob("*"):

                    if (
                        file.is_file()
                        and should_include(file)
                    ):

                        z.write(
                            file,
                            file.relative_to(ROOT),
                        )

    print(
        "Created:",
        OUTPUT,
    )


if __name__ == "__main__":
    create_zip()