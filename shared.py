from os import path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, RandomCrop, Normalize
from torchvision.transforms import RandomHorizontalFlip, CenterCrop, RandomEqualize

from utils import TrainingDataset

BASE_DIR = path.join("C:\\", "Users", "leomv", "Documents", "KneeXrayData", "ClsKLData", "kneeKL224")
CLASSES = ["0", "1", "2", "3", "4"]
dataset_types = ["train", "val"]

datasets = {
    "val": ImageFolder(path.join(BASE_DIR, "val"), transform=Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])),
    "train": TrainingDataset(path.join(BASE_DIR, "train"), transforms=[
        Compose([
            RandomCrop(224),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        Compose([
            RandomCrop(224),
            RandomHorizontalFlip(1.0),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        Compose([
            RandomCrop(224),
            RandomEqualize(1.0),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
    ])
}

dataloaders = {x: DataLoader(datasets[x], batch_size=12, shuffle=True, num_workers=8) for x in dataset_types}
dataset_sizes = {x: len(datasets[x]) for x in dataset_types}

device = "cuda" if (torch.cuda.is_available()) else "cpu"


def get_dataset(dataset_type: str) -> Tuple[list, list]:
    dataset_data, dataset_target = [], []
    for idx, (data, target) in enumerate(dataloaders[dataset_type]):
        dataset_data.append(data)
        dataset_target.append(target)
    return dataset_data, dataset_target
