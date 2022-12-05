import json
from dataclasses import dataclass
from enum import Enum
from time import time
from tkinter.filedialog import askopenfilename
from tkinter.ttk import Separator
from typing import Union, Tuple

import numpy as np
import scipy.special as sk
import torch
from PIL import Image
from customtkinter import CTkToplevel, CTkLabel, CTkImage, CTkEntry, CTkFrame
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pandas import DataFrame
from seaborn import heatmap
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from torchvision.datasets.folder import pil_loader
from torchvision.models import ResNet
from torchvision.transforms import Resize, CenterCrop, ToTensor, Compose, Normalize
from xgboost import XGBClassifier

from shared import kl_level, get_resnext_model, training_device, CLASSES, get_metrics


def is_integer(input):
    try:
        int(input)
        return True
    except:
        return False


@dataclass
class StatsImagesPaths:
    confusion_matrix: str
    metrics: str


@dataclass
class ImagePaths:
    resnext: StatsImagesPaths
    svm: StatsImagesPaths
    xgboost: StatsImagesPaths


class ClassifierOption(Enum):
    ResNext = 0
    SVM = 1
    XGBoost = 2


class MetricsTable(CTkFrame):
    def __init__(self, master: any, metrics: dict, **kwargs):
        super().__init__(master, **kwargs)

        count = 0
        for metric, value in metrics.items():
            CTkEntry(self, placeholder_text=metric, corner_radius=0).grid(row=0, column=count, sticky="ew")
            CTkEntry(self, placeholder_text=value, corner_radius=0).grid(row=1, column=count, sticky="ew")
            count += 1


class SectionTitle(CTkFrame):
    def __init__(self, master: any, title: str, **kwargs):
        super().__init__(master, **kwargs)

        self.start_separator = Separator(self, orient="horizontal")
        self.start_separator.pack(fill="x", expand=1)

        self.title_label = CTkLabel(master=self, text=title, font=(None, 32))
        self.title_label.pack(after=self.start_separator, fill=None, expand=0, anchor="center", pady=10)

        self.end_separator = Separator(self, orient="horizontal")
        self.end_separator.pack(after=self.title_label, fill="x", expand=1)


class ResponsiveImageWindow(CTkToplevel):
    def __init__(self, image: Image, window_title="Image", label=None, metrics: dict = None, padx=0, pady=0, *args):
        CTkToplevel.__init__(self, *args)

        self.title(window_title)

        self.image = image
        self.img_copy = self.image.copy()

        self.background_image = CTkImage(self.image)

        self.label_text = label
        if label is not None:
            self.image_label = CTkLabel(self, text=label)
            self.image_label.pack(fill="x", expand=True, padx=padx, pady=pady)

        self.metrics = metrics
        if metrics is not None:
            self.metrics_table = MetricsTable(self, metrics)
            self.metrics_table.pack(fill="x", expand=True, padx=padx, pady=pady, anchor="center")

        self.background = CTkLabel(self, image=self.background_image, text="")
        self.background.pack(fill="both", expand="true", padx=padx, pady=pady)
        self.background.bind("<Configure>", self._resize_image)

    def _resize_image(self, event):
        new_width = event.width
        new_height = event.height

        self.image = self.img_copy.resize((new_width, new_height))

        self.background_image = CTkImage(self.image, size=(new_width, new_height))
        self.background.configure(image=self.background_image)

        if self.label_text is not None:
            self.image_label = CTkLabel(self, text=self.label_text, font=(None, new_width * 0.8))


def transform_image(image_path: str):
    my_transforms = Compose([
        Resize(255),
        CenterCrop(224),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = pil_loader(image_path)
    return my_transforms(img).unsqueeze(0)


def get_image_path():
    return askopenfilename(title="Pick a knee image", filetypes=[("Images", "*.jpeg"), ("Images", "*.jpg"), ("Images", "*.png")])


def open_prediction_popup(image_path, confidence, found, elapsed_time):
    window = ResponsiveImageWindow(Image.open(image_path),
                                   window_title="Prediction",
                                   padx=20, pady=20,
                                   label=f"{kl_level[found]} at confidence score: {confidence:.2f}\nElapsed time: {elapsed_time:.2f} secs")
    window.geometry("500x500")


def open_metrics_popup_from_paths(stats_paths: StatsImagesPaths, title="Image"):
    with open(stats_paths.metrics) as json_file:
        metrics = json.load(json_file)

    window = ResponsiveImageWindow(Image.open(stats_paths.confusion_matrix), window_title=title, metrics=metrics, padx=20, pady=10)
    window.geometry("500x500")


def open_metrics_popup(confusion_matrix: Figure, metrics: dict, title="Metrics"):
    image = Image.frombytes("RGB", confusion_matrix.canvas.get_width_height(), confusion_matrix.canvas.tostring_rgb())
    window = ResponsiveImageWindow(image, window_title=title, metrics=metrics, padx=20, pady=10)
    window.geometry("500x500")


def sklearn_metrics(model: Union[XGBClassifier, SVC], test_loader: DataLoader) -> Tuple[list, list]:
    predictions = []
    corrects = []

    extractor = get_resnext_model()

    for inputs, labels in test_loader:
        output = extractor(inputs.to(training_device()))
        prediction = model.predict(output.detach().cpu())

        predictions.extend(prediction)

        labels = labels.data.cpu().numpy()
        corrects.extend(labels)

    return predictions, corrects


def resnext_metrics(model: ResNet, test_loader: DataLoader) -> Tuple[list, list]:
    predictions = []
    corrects = []

    for inputs, labels in test_loader:
        output = model(inputs.to(training_device()))
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        predictions.extend(output)

        labels = labels.data.cpu().numpy()
        corrects.extend(labels)

    return predictions, corrects


def model_metrics(model: Union[ResNet, XGBClassifier, SVC], test_loader: DataLoader) -> Tuple[dict, Figure]:
    corrects, predictions = resnext_metrics(model, test_loader) if isinstance(model, ResNet) else sklearn_metrics(model, test_loader)

    cm = confusion_matrix(corrects, predictions, normalize="true")
    cm_df = DataFrame(cm, index=[i for i in CLASSES], columns=[i for i in CLASSES])

    plt.figure()
    heatmap(cm_df, annot=True)
    cm = plt.gcf()

    metrics = get_metrics(len(CLASSES), torch.as_tensor(predictions), torch.as_tensor(corrects))

    return metrics, cm


def resnext_predict(model: ResNet, image_path: str) -> Tuple[float, int, float]:
    elapsed_time = time()

    tensor = transform_image(image_path)
    tensor = tensor.to(training_device())
    output = model.forward(tensor)

    probs = torch.softmax(output, dim=1)
    conf, classes = torch.max(probs, 1)
    confidence, found = conf.item(), classes.item()

    elapsed_time = time() - elapsed_time

    return confidence, found, elapsed_time


def sklearn_predict(model: Union[XGBClassifier, SVC], image_path: str) -> Tuple[float, int, float]:
    elapsed_time = time()

    tensor = transform_image(image_path)
    tensor = tensor.to(training_device())

    output = get_resnext_model().forward(tensor)

    found = model.predict(output.detach().cpu())[0]
    probs = model.predict_proba(output.detach().cpu())

    probs = sk.softmax(probs, axis=1)
    confidence = np.max(probs)

    elapsed_time = time() - elapsed_time

    return confidence, found, elapsed_time


def classifier_name(classifier: Union[ResNet, XGBClassifier, SVC]):
    return "ResNext" if isinstance(classifier, ResNet) else "XGBoost" if isinstance(classifier, XGBClassifier) else "SVM"
