from tkinter.filedialog import askopenfilename

import customtkinter
import torch
from PIL import Image
from PIL.ImageTk import PhotoImage
from torch.nn import Linear
from torch.nn.functional import softmax
from torchvision.datasets.folder import pil_loader
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
from torchvision.transforms import Resize, CenterCrop, ToTensor, Compose, Normalize

customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

kl_level = [
    "Not arthrosis [KL = 0]",
    "Not arthrosis [KL = 1]",
    "Arthrosis [KL = 2]",
    "Arthrosis [KL = 3]",
    "Arthrosis [KL = 4]"
]


def transform_image(image_path: str):
    my_transforms = Compose([
        Resize(255),
        CenterCrop(224),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = pil_loader(image_path)
    return my_transforms(img).unsqueeze(0)


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("Arthrosis detection")
        self.geometry(f"{1100}x{580}")

        self.grid_columnconfigure((1, 2, 3), weight=1)
        self.grid_rowconfigure((1, 2), weight=1)

        self.main_title = customtkinter.CTkLabel(master=self, text="Arthrosis detection", font=(None, 36))
        self.main_title.grid(row=1, column=2)

        self.resnext_button = customtkinter.CTkButton(master=self,
                                                      text="ResNext",
                                                      fg_color="transparent",
                                                      border_width=2,
                                                      text_color=("gray10", "#DCE4EE"),
                                                      command=self.predict_resnext)
        self.resnext_button.grid(row=2, column=1, padx=(20, 20), pady=(20, 20), sticky="ew")

        self.tree_classifier_button = customtkinter.CTkButton(master=self,
                                                              text="Tree Classifier",
                                                              fg_color="transparent",
                                                              border_width=2,
                                                              text_color=("gray10", "#DCE4EE"))
        self.tree_classifier_button.grid(row=2, column=2, padx=(20, 20), pady=(20, 20), sticky="ew")

        self.xgboost_button = customtkinter.CTkButton(master=self,
                                                      text="XGBoost",
                                                      fg_color="transparent",
                                                      border_width=2,
                                                      text_color=("gray10", "#DCE4EE"))
        self.xgboost_button.grid(row=2, column=3, padx=(20, 20), pady=(20, 20), sticky="ew")

    def get_image_path(self):
        return askopenfilename(title="Pick a knee image",
                               filetypes=[("Images", "*.jpeg"), ("Images", "*.jpg"), ("Images", "*.png")])

    def predict_resnext(self):
        image_path = self.get_image_path()

        classes = ["0", "1", "2", "3", "4"]

        device = "cuda" if (torch.cuda.is_available()) else "cpu"
        checkpoint = torch.load("./checkpoint-model")

        model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
        model.fc = Linear(model.fc.in_features, len(classes))
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        tensor = transform_image(image_path)
        tensor = tensor.to(device)
        output = model.forward(tensor)

        probs = softmax(output, dim=1)
        conf, classes = torch.max(probs, 1)
        confidence, found = conf.item(), kl_level[classes.item()]

        self.open_popup(image_path, confidence, found)

    def open_popup(self, image_path, confidence, found):
        window = customtkinter.CTkToplevel()
        window.geometry("500x500")

        prediction = customtkinter.CTkLabel(window, text=f"{found} at confidence score: {confidence:.2f}")
        prediction.grid(row=1, column=1, padx=50, pady=5, sticky="ew")

        image = customtkinter.CTkLabel(window, image=PhotoImage(Image.open(image_path)))
        image.grid(row=2, column=1, padx=50, pady=5, sticky="ew")


if __name__ == "__main__":
    app = App()
    app.mainloop()
