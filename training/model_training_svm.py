import pickle

from sklearn.svm import SVC

from shared import dataloaders, get_extracted_data, get_resnext_model


def train_model():
    extractor = get_resnext_model()
    features, targets = get_extracted_data(dataloaders["train"], extractor=extractor)

    svm = SVC(kernel="linear", probability=True)
    svm.fit(features, targets.reshape(-1))

    pickle.dump(svm, open("../models/svm-model", "wb"))


if __name__ == "__main__":
    train_model()
