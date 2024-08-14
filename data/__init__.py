import cv2
import os
import os.path as osp
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List
from itertools import chain


def preprocessing_image(
    image: np.ndarray, size: tuple = (32, 32), gray: bool = True
) -> np.ndarray:
    '''Convert image to correct size and gray scale (if needed)'''

    image = cv2.resize(src=image, dsize=size, interpolation=cv2.INTER_NEAREST)
    if gray:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
    return image


def get_label(path: Path) -> int:
    '''Get label for each frame'''
    return 1 if "fall" in path.parent.name.lower() else 0


def convert_video_to_frame(path: Path) -> List[Tuple]:
    '''Convert video to list of frame in type: Tuple(frame, label)'''

    vidcap = cv2.VideoCapture(path)
    success, frame = vidcap.read()

    images_w_labels = []
    while success:
        images_w_labels.append((preprocessing_image(frame), get_label(path)))
        success, frame = vidcap.read()

    vidcap.release()

    return images_w_labels


def get_all_file(directory: Path) -> List[Path]:
    '''Convert the combined iterable of video '.mp4' and .'MOV' to the list'''
    
    return list(chain(directory.glob("*.mp4") + directory.glob("*.MOV")))


def get_dataset(path: Path) -> None:
    '''Get all data to the list Dataset'''
    
    folders = [path / directory for directory in os.listdir(path)]

    dataset = []
    for folder in folders:
        files = get_all_file(directory=folder)
        for file in files:
            dataset.append(convert_video_to_frame(file))
            
    print(f'{len(dataset)=}')


def train_valid_test_split(
    df: pd.DataFrame,
    destination: Path = Path('Dataset'),
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    random_state: int = 242,
) -> Tuple[pd.DataFrame]:
    '''
    Split data to train, validation and test dataframe.
    Args:
        - df (pd.DataFrame): the dataframe with 2 columns, include 'path' and 'label'.
        - train_ratio, val_ratio (float, optional): the ratio of train and validation dataframe.
    '''
    
    test_ratio = 1 - train_ratio - val_ratio
    
    # create train, val, test dataset corresponding train, val and test ratio
    train_df = df.sample(n=int(df.shape[0] * train_ratio), random_state=random_state)
    val_df = df.sample(n=int(df.shape[0] * val_ratio), random_state=random_state)
    test_df = df.sample(n=int(df.shape[0] * test_ratio), random_state=random_state)
    
    # make the directory to destination
    os.makedirs(name=str(destination), exist_ok=True)
    
    train_df.to_csv(str(destination / 'train.csv'), index=False)
    val_df.to_csv(str(destination / 'val.csv'), index=False)
    test_df.to_csv(str(destination / 'test.csv'), index=False)
    
    return train_df, val_df, test_df