import copy
from typing import Callable

import torch
from sklearn.svm import SVC
from torch.nn import Linear
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
from xgboost import XGBClassifier

from shared import training_device, get_resnext_model, xgboost_params, get_extracted_data
from training.custom_cross_entropy_loss import cross_entropy


def train_resnext_model(dataloaders, dataset_sizes, num_epochs, num_classes, on_step: Callable):
    model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
    model.fc = Linear(model.fc.in_features, num_classes)
    model = model.to(training_device())
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    torch.cuda.empty_cache()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    count = 0

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(training_device())
                labels = labels.to(training_device())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = cross_entropy(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                count += 1
                on_step(count)
            if phase == "train":
                scheduler.step()

            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model


def train_svm_model(dataloaders):
    extractor = get_resnext_model()
    features, targets = get_extracted_data(dataloaders["train"], extractor=extractor)

    svm = SVC(kernel="linear", probability=True)
    svm.fit(features, targets.reshape(-1))

    return svm


def train_xgboost_model(dataloaders):
    extractor = get_resnext_model()

    features_train, targets_train = get_extracted_data(dataloaders["train"], extractor=extractor)
    features_eval, targets_eval = get_extracted_data(dataloaders["val"], extractor=extractor)

    model = XGBClassifier(**xgboost_params)
    model = model.fit(features_train, targets_train.reshape(-1),
                      eval_set=[(features_eval, targets_eval.reshape(-1))],
                      early_stopping_rounds=20,
                      verbose=False)

    return model
