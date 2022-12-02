from tkinter.filedialog import askopenfilename

import torch
from PIL import Image
from customtkinter import CTkToplevel, CTkLabel, CTkImage
from torch.nn import Linear
from torchvision.datasets.folder import pil_loader
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
from torchvision.transforms import Resize, CenterCrop, ToTensor, Compose, Normalize

CLASSES = ["0", "1", "2", "3", "4"]

resnext_model = None
training_device = "cuda" if (torch.cuda.is_available()) else "cpu"


def get_resnext_model():
    global resnext_model

    if resnext_model is not None:
        return resnext_model

    checkpoint = torch.load("./checkpoint-model")

    resnext_model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
    resnext_model.fc = Linear(resnext_model.fc.in_features, len(CLASSES))
    resnext_model.load_state_dict(checkpoint["model_state_dict"])
    model = resnext_model.to(training_device)
    model.eval()

    return resnext_model


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
    return askopenfilename(title="Pick a knee image",
                           filetypes=[("Images", "*.jpeg"), ("Images", "*.jpg"), ("Images", "*.png")])


def open_popup(image_path, confidence, found, elapsed_time):
    window = CTkToplevel()
    window.geometry("500x500")

    prediction = CTkLabel(window, text=f"{found} at confidence score: {confidence:.2f}\n"
                                       f"Elapsed time: {elapsed_time:.2f} secs")
    prediction.pack(fill="x", expand=True)

    image_width = window.winfo_width()
    image_height = window.winfo_height()

    image = CTkLabel(window, text="", image=CTkImage(Image.open(image_path), size=(image_width, image_height)))
    image.pack(fill="x", expand=True)
