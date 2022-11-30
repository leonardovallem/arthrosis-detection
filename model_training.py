import copy
import time

import torch
from torch.nn import Linear
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchvision import models
from torchvision.models import ResNeXt50_32X4D_Weights

from utils import custom_loss_function, print_percent_done
from shared import dataloaders, device, dataset_sizes, CLASSES


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

            print(f"Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n")

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


if __name__ == "__main__":
    print(f"using {device}")

    model_ft = models.resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
    model_ft.fc = Linear(model_ft.fc.in_features, len(CLASSES))

    model_ft = model_ft.to(device)

    # Observe that all parameters are being optimized
    optimizer_ft = SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = StepLR(optimizer_ft, step_size=7, gamma=0.1)

    checkpoint = torch.load("./checkpoint-model")
    model_ft = train_model(model_ft, custom_loss_function, optimizer_ft, exp_lr_scheduler, num_epochs=10)
