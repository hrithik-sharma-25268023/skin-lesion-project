"""prepares the image data for the project"""

import os


def rename_files(source_path: str, destination_path: str) -> None:
    """renames the file as per year"""

    os.rename(src=source_path, dst=destination_path)
