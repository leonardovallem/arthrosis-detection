from time import time

import customtkinter
import torch
from PIL import Image
from customtkinter import CTk, CTkLabel, CTkButton, CTkProgressBar, CTkImage
from torch.nn.functional import softmax

from gui.utils import get_image_path, transform_image, open_popup, get_resnext_model, training_device

customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")

kl_level = [
    "Not arthrosis [KL = 0]",
    "Not arthrosis [KL = 1]",
    "Arthrosis [KL = 2]",
    "Arthrosis [KL = 3]",
    "Arthrosis [KL = 4]"
]


class App(CTk):
    def __init__(self):
        super().__init__()
        self.title("Arthrosis detection")
        self.geometry(f"{1100}x{580}")
        self.state("zoomed")

        self.grid_columnconfigure((1, 2, 3), weight=1)
        self.grid_rowconfigure((1, 2, 3, 4), weight=1)

        self.main_title = CTkLabel(master=self, text="Arthrosis detection", font=(None, 36))
        self.main_title.grid(row=1, column=2)

        self.resnext_button = CTkButton(master=self,
                                        text="ResNext",
                                        fg_color="transparent",
                                        border_width=2,
                                        text_color=("gray10", "#DCE4EE"),
                                        command=self.predict_resnext)
        self.resnext_button.grid(row=2, column=1, padx=(20, 20), pady=(20, 20), sticky="ew")

        self.resnext_confusion_matrix = CTkLabel(self,
                                                 text="",
                                                 image=CTkImage(Image.open("./stats/resnext_confusion_matrix.png"), size=(300, 300)))

        self.resnext_confusion_matrix.grid(row=3, column=1, sticky="nsew")

        self.tree_classifier_button = CTkButton(master=self,
                                                text="Tree Classifier",
                                                fg_color="transparent",
                                                border_width=2,
                                                text_color=("gray10", "#DCE4EE"))
        self.tree_classifier_button.grid(row=2, column=2, padx=(20, 20), pady=(20, 20), sticky="ew")

        self.xgboost_button = CTkButton(master=self,
                                        text="XGBoost",
                                        fg_color="transparent",
                                        border_width=2,
                                        text_color=("gray10", "#DCE4EE"))
        self.xgboost_button.grid(row=2, column=3, padx=(20, 20), pady=(20, 20), sticky="ew")

        self.loading_bar = CTkProgressBar(self, mode="indeterminate")

    def predict_resnext(self):
        self.loading_bar.grid(row=4, sticky="ew", columnspan=4)
        self.loading_bar.start()

        image_path = get_image_path()

        model = get_resnext_model()

        elapsed_time = time()
        tensor = transform_image(image_path)
        tensor = tensor.to(training_device)
        output = model.forward(tensor)

        probs = softmax(output, dim=1)
        conf, classes = torch.max(probs, 1)
        confidence, found = conf.item(), kl_level[classes.item()]

        print("classes", classes)

        elapsed_time = time() - elapsed_time

        self.loading_bar.grid_forget()
        open_popup(image_path, confidence, found, elapsed_time)


if __name__ == "__main__":
    app = App()
    app.mainloop()
