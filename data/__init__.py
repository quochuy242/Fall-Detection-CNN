import cv2
import numpy as np
import os
import os.path as osp
from pathlib import Path
from typing import Tuple, List


def preprocessing_image(
    image: np.ndarray, size: tuple = (32, 32), gray: bool = True
) -> np.ndarray:
    """Convert image to correct size and convert to gray scale (if needed)"""

    image = cv2.resize(src=image, dsize=size, interpolation=cv2.INTER_NEAREST)
    if gray:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def get_label(path: Path) -> int:
    return 1 if "fall" in path.parent.name.lower() else 0


def convert_video_to_frame(path: Path) -> List[Tuple]:
    """Convert video to list of frame in type: tuple(frame, label)"""

    vidcap = cv2.VideoCapture(path)
    success, frame = vidcap.read()

    images_w_labels = []
    while success:
        images_w_labels.append((preprocessing_image(frame), get_label(path)))
        success, frame = vidcap.read()

    vidcap.release()

    return images_w_labels


def get_all_file(directory: Path) -> List[Path]:
    return directory.glob("*.mp4") + directory.glob("*.MOV")


def get_dataset(path: Path):
    folders = [path / directory for directory in os.listdir(path)]

    dataset = []
    for folder in folders:
        files = get_all_file(directory=folder)
        for file in files:
            dataset.append(convert_video_to_frame(file))

    print(f"{len(dataset)=}")
