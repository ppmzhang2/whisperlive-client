"""Export transcripts to text file."""

import os

import loguru
import numpy as np

logger = loguru.logger


def npy2txt(folder_path: str, output_path: str) -> None:
    """Load and combine .npy transcript files into a single text file.

    Args:
        folder_path (str): Path to the folder containing .npy transcript files.
        output_path (str): Path to the output .txt file.
    """
    if not os.path.isdir(folder_path):
        logger.error(f"Directory not found: {folder_path}")
        raise FileNotFoundError()

    # Get all .npy files sorted by filename (timestamped if saved that way)
    npy_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(".npy")],
        key=lambda f: os.path.getmtime(os.path.join(folder_path, f)))

    combined_text = []

    for file in npy_files:
        file_path = os.path.join(folder_path, file)
        try:
            arr = np.load(file_path)
            if isinstance(arr, np.ndarray) and "txt" in arr.dtype.names:
                combined_text.extend(arr["txt"])
        except Exception:
            logger.exception(f"Failed to load {file_path}")

    # Save to output text file
    with open(output_path, "w", encoding="utf-8") as f_out:
        for line in combined_text:
            f_out.write(line.strip() + "\n")

    logger.info(f"Transcript saved to {output_path}")
