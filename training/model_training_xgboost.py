import pickle

import xgboost as xgb

from shared import dataloaders, get_extracted_data, get_resnext_model, xgboost_params


def train_model():
    extractor = get_resnext_model()

    features_train, targets_train = get_extracted_data(dataloaders["train"], extractor=extractor)
    features_eval, targets_eval = get_extracted_data(dataloaders["val"], extractor=extractor)

    xgb_model_1 = xgb.XGBClassifier(**xgboost_params)
    xgb_model_1 = xgb_model_1.fit(features_train, targets_train.reshape(-1),
                                  eval_set=[(features_eval, targets_eval.reshape(-1))],
                                  early_stopping_rounds=20,
                                  verbose=False)
    xgb_model_1.save_model("../models/xgb_model_1.json")
    pickle.dump(xgb_model_1, open("../models/checkpoint-xgb-model-1", "wb"))

    xgb_model_2 = xgb.XGBClassifier(**xgboost_params)
    xgb_model_2 = xgb_model_2.fit(features_eval, targets_eval.reshape(-1),
                                  eval_set=[(features_train, targets_train.reshape(-1))],
                                  early_stopping_rounds=20,
                                  verbose=False)
    xgb_model_2.save_model("../models/xgb_model_2.json")
    pickle.dump(xgb_model_2, open("../models/checkpoint-xgb-model-2", "wb"))


if __name__ == "__main__":
    train_model()
