import json
import shutil
from pathlib import Path


INDEX_FILE_NAME = "index.faiss"
MANIFEST_FILE_NAME = "manifest.json"
METADATA_FILE_NAME = "metadata.parquet"


def resolve_index_paths(index_path):
    index_dir = Path(index_path)
    return {
        "index_dir": index_dir,
        "index_file": index_dir / INDEX_FILE_NAME,
        "manifest_file": index_dir / MANIFEST_FILE_NAME,
        "metadata_file": index_dir / METADATA_FILE_NAME,
    }


def prepare_index_directory(index_path, overwrite):
    paths = resolve_index_paths(index_path)
    index_dir = paths["index_dir"]

    if index_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {index_dir}. Pass --overwrite to replace it."
            )

        shutil.rmtree(index_dir)

    index_dir.parent.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    return paths


def write_manifest(manifest_path, values):
    manifest_path.write_text(
        json.dumps(values, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_manifest(manifest_path):
    return json.loads(manifest_path.read_text(encoding="utf-8"))
