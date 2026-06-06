import requests
import zipfile
import shutil
from pathlib import Path

def download_file(url: str, dest: Path) -> None:
    """
    Download a file from the given URL and save it to the destination path.
    """
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses
    dest.write_bytes(response.content)

def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """
    Extracts the given zip file into the specified directory.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def fix_extracted_structure(extract_dir: Path, target_dir: Path) -> None:
    """
    Move all files from extract_dir (recursively) to target_dir,
    flattening any folder structure.
    """
    for path in extract_dir.rglob('*'):
        if path.is_file():
            destination = target_dir / path.name
            try:
                path.rename(destination)
            except FileExistsError:
                # when flattening directories and a duplicate filename is found
                print(f"Warning: Duplicate filename '{path.name}' found. The first one was kept.")
                pass
    shutil.rmtree(extract_dir)
