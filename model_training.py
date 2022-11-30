import copy
import time
from os import path

import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
from torch.nn import Linear, CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.datasets import ImageFolder
from torchvision.models import ResNeXt50_32X4D_Weights
from torchvision.transforms import Compose, Resize, ToTensor, RandomCrop, Normalize
from torchvision.transforms import RandomHorizontalFlip, CenterCrop, RandomEqualize

from dataset_utils import TrainingDataset

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


def print_percent_done(current, total, bar_len=100, title="Please wait"):
    percent_done = (current + 1) / total * 100
    percent_done = round(percent_done, 1)

    done = round(percent_done / (100 / bar_len))
    togo = bar_len - done

    done_str = "█" * int(done)
    togo_str = "░" * int(togo)

    print(f"\t⏳{title}: [{done_str}{togo_str}] {percent_done}% done")


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, checkpoint: object = None):
    start_epoch = 0

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    num_epochs += start_epoch

    torch.cuda.empty_cache()
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(start_epoch, num_epochs):
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # progress display
            total = len(dataloaders[phase])
            current = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                print_percent_done(current, total, title=f"Epoch {epoch}/{num_epochs - 1} [{phase.capitalize()}]")
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                current += 1
            print()
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save({
        "epoch": num_epochs - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, "./checkpoint-model")

    return model


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {CLASSES[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


if __name__ == "__main__":
    dataloaders = {
        x: DataLoader(datasets[x], batch_size=12, shuffle=True, num_workers=8) for x in ["train", "val"]
    }
    dataset_sizes = {x: len(datasets[x]) for x in ["train", "val"]}

    device = "cuda" if (torch.cuda.is_available()) else "cpu"
    print(f"using {device}")

    model_ft = models.resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
    model_ft.fc = Linear(model_ft.fc.in_features, len(CLASSES))

    model_ft = model_ft.to(device)
    criterion = CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = StepLR(optimizer_ft, step_size=7, gamma=0.1)

    checkpoint = torch.load("./checkpoint-model")
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10, checkpoint=checkpoint)
    visualize_model(model_ft)
