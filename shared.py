import glob
import pickle
from enum import Enum
from os import path
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from numpy import array, ndarray
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from torch import Tensor
from torch.nn import Identity, Module, Linear
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassSpecificity, MulticlassAccuracy, MulticlassRecall
from torchvision.datasets import ImageFolder
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
from torchvision.models.resnet import resnet101, ResNet
from torchvision.transforms import Compose, Resize, ToTensor, RandomCrop, Normalize
from torchvision.transforms import RandomHorizontalFlip, CenterCrop, RandomEqualize
from xgboost import XGBClassifier

from training.utils import TrainingDataset, show_progress_bar

BASE_DIR = path.join("C:\\", "Users", "leomv", "Documents", "KneeXrayData", "ClsKLData", "kneeKL224")
dataset_types = ["train", "val"]

CLASSES = ["0", "1", "2", "3", "4"]
kl_level = [
    "Not arthrosis [KL = 0]",
    "Not arthrosis [KL = 1]",
    "Arthrosis [KL = 2]",
    "Arthrosis [KL = 3]",
    "Arthrosis [KL = 4]"
]

xgboost_params = {
    "random_state": 42,
    "objective": "multi:softmax",
    "num_class": len(CLASSES),
    "n_estimators": 200,
    "eval_metric": "mlogloss"
}

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

resnext_model = None


def get_resnext_model() -> ResNet:
    global resnext_model

    if resnext_model is not None:
        return resnext_model

    checkpoint = torch.load("../models/checkpoint-model")

    resnext_model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
    resnext_model.fc = Linear(resnext_model.fc.in_features, len(CLASSES))
    resnext_model.load_state_dict(checkpoint["model_state_dict"])
    resnext_model = resnext_model.to(training_device())
    resnext_model.eval()

    return resnext_model


def get_svm_model() -> SVC:
    return pickle.load(open(path.join(Path(__file__).parent.absolute(), "models", "svm-model"), "rb"))


def get_xgboost_model() -> XGBClassifier:
    model = XGBClassifier(**xgboost_params)
    model.load_model(path.join(Path(__file__).parent.absolute(), "models", "xgb_model_1.json"))

    return model


class DatasetType(Enum):
    Train = "train"
    Validation = "val"
    Test = "test"


def training_device() -> str:
    return "cuda" if (torch.cuda.is_available()) else "cpu"


def fix_cardinality_issues(*datasets: list) -> Tuple[list, ...]:
    fixed_datasets = ()

    for dataset in datasets:
        inconsistency = len(dataset[len(dataset) - 1]) != len(dataset[len(dataset) - 2])
        fixed_datasets += (dataset[:-1] if inconsistency else dataset,)

    return datasets


def load_dataset(dataset_type: DatasetType) -> Tuple[ndarray, ndarray]:
    images, labels = [], []

    for directory_path in glob.glob(path.join(BASE_DIR, str(dataset_type.value), "*")):
        label = directory_path.split("\\")[-1]

        for img_path in glob.glob(path.join(directory_path, "*.png")):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            images.append(img)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    encoder = LabelEncoder()
    encoder.fit(labels)

    return images / 255.0, encoder.transform(labels)


def get_extracted_data(data_loader: DataLoader, extractor: Module = None, save: bool = False):
    if extractor is None:
        extractor = resnet101(pretrained=True)
        try:
            extractor.load_state_dict("feature_extractor.pth")
            print("feature extractor loaded from file")
        except:
            extractor.fc = Identity()
            extractor.to(training_device())
        extractor.eval()

    total = len(data_loader)
    current = 0

    for index, (data, target) in enumerate(data_loader):
        show_progress_bar(current, total)
        img_tensor = data.to(training_device())
        target = target.numpy()

        with torch.no_grad():
            feature = extractor(img_tensor)

        feature = feature.cpu().detach().squeeze(0).numpy()

        if index == 0:
            features = feature
            targets = target
        else:
            features = np.concatenate([features, feature], axis=0)
            targets = np.concatenate([targets, target], axis=0)

        current += 1

    if save:
        torch.save(extractor.state_dict(), "./models/feature_extractor.pth")

    return features, targets


def get_dataset(dataset_type: str) -> Tuple[array, array]:
    dataset_data, dataset_target = [], []
    encoder = LabelEncoder()

    for idx, (data, target) in enumerate(dataloaders[dataset_type]):
        dataset_data.append(data)

        encoder.fit(target)
        dataset_target.append(encoder.transform(target))

    dataset_data, dataset_target = fix_cardinality_issues(dataset_data, dataset_target)

    return np.array(dataset_data, dtype=np.object), np.array(dataset_target, dtype=np.object)


def get_metrics(num_classes: int, preds: Tensor, target: Tensor) -> dict:
    # ["Sensitivity", "Specificity", "Precision", "Accuracy", "F1 Score"]
    metrics = {}

    sensitivity = MulticlassRecall(num_classes=num_classes)
    metrics["Sensitivity"] = sensitivity(preds, target).item()

    specificity = MulticlassSpecificity(num_classes=num_classes)
    metrics["Specificity"] = specificity(preds, target).item()

    precision = MulticlassPrecision(num_classes=num_classes)
    metrics["Precision"] = precision(preds, target).item()

    f1score = MulticlassF1Score(num_classes=num_classes)
    metrics["F1-Score"] = f1score(preds, target).item()

    accuracy = MulticlassAccuracy(num_classes=num_classes)
    metrics["Accuracy"] = accuracy(preds, target).item()

    return metrics
