from threading import Thread
from time import time
from tkinter.filedialog import askdirectory
from typing import Union

import customtkinter
from customtkinter import CTk, CTkLabel, CTkButton, CTkProgressBar, CTkToplevel, CTkEntry, CTkFont
from sklearn.svm import SVC
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from xgboost import XGBClassifier

from gui.train_custom_model import train_resnext_model, train_xgboost_model, train_svm_model
from gui.utils import get_image_path, open_prediction_popup, open_metrics_popup, ImagePaths, StatsImagesPaths, is_integer, model_metrics, resnext_predict, \
    open_metrics_popup_from_paths, ClassifierOption, sklearn_predict, classifier_name, SectionTitle
from shared import get_resnext_model, get_xgboost_model, get_svm_model

customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")

image_paths = ImagePaths(
    resnext=StatsImagesPaths(
        confusion_matrix="../stats/resnext_confusion_matrix.png",
        metrics="../stats/resnext_metrics.json"
    ),
    svm=StatsImagesPaths(
        confusion_matrix="../stats/svm_confusion_matrix.png",
        metrics="../stats/svm_metrics.json"
    ),
    xgboost=StatsImagesPaths(
        confusion_matrix="../stats/xgboost_confusion_matrix.png",
        metrics="../stats/xgboost_metrics.json"
    ),
)


def custom_trained_model_options(model: Union[ResNet, XGBClassifier, SVC], training_time: float, batch_size: int):
    window = CTkToplevel()
    window.geometry("400x400")
    window.title(f"Custom trained {classifier_name(model)} options")

    window.grid_rowconfigure((1, 2, 3), weight=1)
    window.grid_columnconfigure((1, 2), weight=1)

    def open_metrics():
        test_dataset_dir = askdirectory(title="Pick image test dataset")
        loading_bar.grid(row=3, columnspan=3, sticky="ew", padx=20, pady=10)
        loading_bar.start()

        dataset = ImageFolder(test_dataset_dir, transform=Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]))

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

        metrics, confusion_matrix = model_metrics(model, dataloader)

        loading_bar.stop()
        loading_bar.grid_forget()

        open_metrics_popup(confusion_matrix, metrics)

    def predict():
        image_path = get_image_path()
        loading_bar.grid(row=3, columnspan=3, sticky="ew", padx=20, pady=10)
        loading_bar.start()

        confidence, found, elapsed_time = resnext_predict(model, image_path) if isinstance(model, ResNet) else sklearn_predict(model, image_path)

        loading_bar.stop()
        loading_bar.grid_forget()

        open_prediction_popup(image_path, confidence, found, elapsed_time)

    time_elapsed = CTkLabel(master=window, text=f"Model training done within {training_time:.2f} secs")
    time_elapsed.grid(row=1, columnspan=3, padx=20, pady=20, sticky="ew")

    metrics_button = CTkButton(master=window, text="Metrics", command=lambda: Thread(target=open_metrics).start())
    metrics_button.grid(row=2, column=1, sticky="ew", padx=20, pady=10)

    predict_button = CTkButton(master=window, text="Predict image", command=lambda: Thread(target=predict).start())
    predict_button.grid(row=2, column=2, sticky="ew", padx=20, pady=10)

    loading_bar = CTkProgressBar(master=window, mode="indeterminate")


def custom_model_training(classifier: ClassifierOption):
    window = CTkToplevel()
    window.title(f"Train {classifier.name} model")
    window.geometry("900x500")

    directory = None

    def get_dir():
        nonlocal directory
        directory = askdirectory(title="Pick image dataset")
        pick_dir_button.configure(text=directory if directory is not None else "Pick dataset")
        train_button.configure(state="enabled" if directory is not None else "disabled")

    def train():
        pick_dir_button.configure(state="disable")
        train_button.configure(state="disable")

        epochs = 1
        if classifier is ClassifierOption.ResNext:
            epochs = int(num_epochs.get())
        batches = int(batch_size.get())

        dataset = ImageFolder(directory, transform=Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]))

        total_steps = epochs * len(dataset)
        progress_bar.configure(width=total_steps)
        progress_bar.pack(fill="both", padx=20, pady=20)

        train_ds, val_ds = random_split(dataset, [0.7, 0.3])

        dataloaders = {
            "train": DataLoader(train_ds, batch_size=batches, shuffle=True, num_workers=8),
            "val": DataLoader(val_ds, batch_size=batches, shuffle=True, num_workers=8)
        }
        dataset_sizes = {"train": len(train_ds), "val": len(val_ds)}

        elapsed_time = time()

        if classifier is ClassifierOption.ResNext:
            model = train_resnext_model(dataloaders, dataset_sizes, epochs, len(dataset.classes), on_step=lambda value: progress_bar.set(value))
        elif classifier is ClassifierOption.SVM:
            model = train_svm_model(dataloaders)
        else:
            model = train_xgboost_model(dataloaders)

        elapsed_time = time() - elapsed_time

        window.destroy()
        custom_trained_model_options(model, elapsed_time, batches)

    pick_dir_button = CTkButton(master=window, text="Pick dataset", command=get_dir)
    pick_dir_button.pack(anchor="center", fill="x", padx=20, pady=20)

    if classifier is ClassifierOption.ResNext:
        num_epochs_label = CTkLabel(master=window, text="Number of epochs")
        num_epochs_label.pack(padx=20, pady=(10, 0), anchor="w")
        num_epochs = CTkEntry(master=window, placeholder_text="Number of epochs", validatecommand=is_integer)
        num_epochs.pack(padx=20, pady=(0, 10), fill="x")

    batch_size_label = CTkLabel(master=window, text="Batch size")
    batch_size_label.pack(padx=20, pady=(10, 0), anchor="w")
    batch_size = CTkEntry(master=window, placeholder_text="Batch size", validatecommand=is_integer)
    batch_size.pack(padx=20, pady=(0, 10), fill="x")

    train_button = CTkButton(master=window, text="Train", command=lambda: Thread(target=train).start())
    train_button.pack(anchor="e", side="bottom", padx=20, pady=20)
    train_button.configure(state="disabled")

    progress_bar = CTkProgressBar(master=window)
    progress_bar.set(0)


class App(CTk):
    def __init__(self):
        super().__init__()
        self.title("Arthrosis detection")
        self.geometry(f"{1100}x{580}")
        self.state("zoomed")

        self.grid_columnconfigure((1, 2, 3), weight=1)
        self.grid_rowconfigure((1, 2, 3, 4, 5, 6, 7), weight=1)

        self.main_title = CTkLabel(master=self, text="Arthrosis detection", font=(None, 48), text_color="#FFD700")
        self.main_title.grid(row=1, columnspan=5, sticky="ew")

        # pretrained models label
        self.pretrained_label = SectionTitle(master=self, title="Pre trained models")
        self.pretrained_label.grid(row=2, columnspan=5, sticky="ew")

        # resnext gui components
        self.pretrained_resnext_button = CTkButton(master=self,
                                                   text="ResNext",
                                                   fg_color="transparent",
                                                   border_width=2,
                                                   font=(None, 20),
                                                   text_color=("gray10", "#DCE4EE"),
                                                   command=lambda: Thread(target=self.predict_image_resnext).start())
        self.pretrained_resnext_button.grid(row=3, column=1, padx=20, pady=20, sticky="ew")

        self.resnext_metrics = CTkButton(self,
                                         text="ResNext Metrics",
                                         fg_color="transparent",
                                         font=CTkFont(size=16, underline=True),
                                         command=lambda: open_metrics_popup_from_paths(image_paths.resnext, title="ResNext metrics"))

        self.resnext_metrics.grid(row=4, column=1, sticky="nsew")

        # svm gui components
        self.pretrained_svm_button = CTkButton(master=self,
                                               text="SVM",
                                               fg_color="transparent",
                                               border_width=2,
                                               font=(None, 20),
                                               text_color=("gray10", "#DCE4EE"),
                                               command=lambda: Thread(target=self.predict_svm).start())
        self.pretrained_svm_button.grid(row=3, column=2, padx=20, pady=20, sticky="ew")

        self.svm_metrics = CTkButton(self,
                                     text="SVM Metrics",
                                     fg_color="transparent",
                                     font=CTkFont(size=16, underline=True),
                                     command=lambda: open_metrics_popup_from_paths(image_paths.svm, title="SVM metrics"))

        self.svm_metrics.grid(row=4, column=2, sticky="nsew")

        # xgboost gui components
        self.pretrained_xgboost_button = CTkButton(master=self,
                                                   text="XGBoost",
                                                   fg_color="transparent",
                                                   border_width=2,
                                                   font=(None, 20),
                                                   text_color=("gray10", "#DCE4EE"),
                                                   command=lambda: Thread(target=self.predict_xgboost).start())
        self.pretrained_xgboost_button.grid(row=3, column=3, padx=20, pady=20, sticky="ew")

        self.xgboost_metrics = CTkButton(self,
                                         text="XGBoost Metrics",
                                         fg_color="transparent",
                                         font=CTkFont(size=16, underline=True),
                                         command=lambda: open_metrics_popup_from_paths(image_paths.xgboost, title="XGBoost metrics"))
        self.xgboost_metrics.grid(row=4, column=3, sticky="nsew")

        # customized train model label
        self.pretrained_label = SectionTitle(master=self, title="Train custom model")
        self.pretrained_label.grid(row=5, columnspan=5, sticky="ew")

        self.train_custom_resnext_button = CTkButton(master=self,
                                                     text="ResNext",
                                                     fg_color="transparent",
                                                     border_width=2,
                                                     font=(None, 20),
                                                     text_color=("gray10", "#DCE4EE"),
                                                     command=lambda: custom_model_training(ClassifierOption.ResNext))
        self.train_custom_resnext_button.grid(row=6, column=1, padx=20, pady=20, sticky="ew")

        self.train_custom_svm_button = CTkButton(master=self,
                                                 text="SVM",
                                                 fg_color="transparent",
                                                 border_width=2,
                                                 font=(None, 20),
                                                 text_color=("gray10", "#DCE4EE"),
                                                 command=lambda: custom_model_training(ClassifierOption.SVM))
        self.train_custom_svm_button.grid(row=6, column=2, padx=20, pady=20, sticky="ew")

        self.train_custom_xgboost_button = CTkButton(master=self,
                                                     text="XGBoost",
                                                     fg_color="transparent",
                                                     border_width=2,
                                                     font=(None, 20),
                                                     text_color=("gray10", "#DCE4EE"),
                                                     command=lambda: custom_model_training(ClassifierOption.XGBoost))
        self.train_custom_xgboost_button.grid(row=6, column=3, padx=20, pady=20, sticky="ew")

        self.loading_bar = CTkProgressBar(self, mode="indeterminate")

    def predict_image_resnext(self, model=None):
        image_path = get_image_path()

        self.loading_bar.grid(row=7, sticky="ew", columnspan=4)
        self.loading_bar.start()

        if model is None:
            model = get_resnext_model()

        confidence, found, elapsed_time = resnext_predict(model, image_path)

        self.loading_bar.grid_forget()
        open_prediction_popup(image_path, confidence, found, elapsed_time)

    def predict_svm(self):
        image_path = get_image_path()

        self.loading_bar.grid(row=7, sticky="ew", columnspan=4)
        self.loading_bar.start()

        svm = get_svm_model()
        confidence, found, elapsed_time = sklearn_predict(svm, image_path)

        self.loading_bar.grid_forget()
        open_prediction_popup(image_path, confidence, found, elapsed_time)

    def predict_xgboost(self):
        image_path = get_image_path()

        self.loading_bar.grid(row=7, sticky="ew", columnspan=4)
        self.loading_bar.start()

        model = get_xgboost_model()
        confidence, found, elapsed_time = sklearn_predict(model, image_path)

        self.loading_bar.grid_forget()
        open_prediction_popup(image_path, confidence, found, elapsed_time)


if __name__ == "__main__":
    app = App()
    app.mainloop()
