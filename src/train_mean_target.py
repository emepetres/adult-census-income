import pandas as pd

import config
from common.encoding import fill_cat_with_none, mean_target_encoding
from model_dispatcher import XGBoostEncoded


if __name__ == "__main__":
    # load the full training data with folds
    df = pd.read_csv(config.TRAINING_FILE)

    # map targets to 0s and 1s
    target_mapping = {"<=50K": 0, ">50K": 1}
    df.loc[:, config.TARGET] = df.loc[:, config.TARGET].map(target_mapping)

    # FIXME: map introduces a new row filled with nan values¿? #122
    df = df.iloc[:-1]

    ord_features = ["age", "fnlwgt", "capital.gain", "capital.loss", "hours.per.week"]
    # # # Drop numerical columns for simplicity
    # # df.drop(ord_features, inplace=True, axis=1)

    # all columns are features except target and kfold columns
    features = [f for f in df.columns if f not in (config.TARGET, "kfold")]
    cat_features = [col for col in features if col not in ord_features]

    fill_cat_with_none(df, cat_features)

    df = mean_target_encoding(df, ord_features, cat_features, folds=5)
    features = [f for f in df.columns if f not in (config.TARGET, "kfold")]

    for fold_ in range(5):
        # initialize model
        custom_model = XGBoostEncoded(df, fold_, config.TARGET, features)

        # fit model on training data
        custom_model.encode()  # dummy function that only prepares data
        custom_model.fit()

        # predict on validation data and get roc auc score
        auc = custom_model.predict_and_score()

        # print auc
        print(f"Fold = {fold_}, AUC = {auc}")
