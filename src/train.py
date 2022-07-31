import argparse
import itertools
import pandas as pd

from sklearn import metrics

import config
from model_dispatcher import (
    DecisionTreeModelSVD,
    ModelInterface,
    LogisticRegressionModel,
    DecisionTreeModel,
    XGBoost,
)


def feature_engineering(df, cat_cols) -> pd.DataFrame:
    """
    This function is used for feature engineering
    :param df: the pandas dataframe with train/test data
    :param cat_cols: list of categorical columns
    :return: dataframe with new features
    """
    # this will create all 2-combinations of values in this list
    # for example:
    # list(itertools.combinations([1,2,3], 2)) will return
    # [(1, 2), (1, 3), (2, 3)]
    combi = list(itertools.combinations(cat_cols, 2))
    for c1, c2 in combi:
        df.loc[:, c1 + "_" + c2] = df[c1].astype(str) + "_" + df[c2].astype(str)

    return df


def run(fold: int, model: ModelInterface):
    # load the full training data with folds
    df = pd.read_csv(config.TRAINING_FILE)

    # map targets to 0s and 1s
    target_mapping = {"<=50K": 0, ">50K": 1}
    df.loc[config.TARGET, :] = df.income.map(target_mapping)

    # FIXME: map introduces a new row filled with nan values¿? #122
    df = df.iloc[:-1]

    ord_features = ["age", "fnlwgt", "capital.gain", "capital.loss", "hours.per.week"]
    # # # Drop numerical columns for simplicity
    # # df.drop(ord_features, inplace=True, axis=1)

    # all columns are features except target and kfold columns
    features = [f for f in df.columns if f not in (config.TARGET, "kfold")]
    cat_features = [col for col in features if col not in ord_features]

    # add new features
    df = feature_engineering(df, cat_features)
    features = [f for f in df.columns if f not in (config.TARGET, "kfold")]
    cat_features = [col for col in features if col not in ord_features]

    # fill all NaN values with NONE
    # note that I am converting all columns to "strings"
    # it doesn't matter because all are categories
    for col in cat_features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # initialize model
    lr_model = model(df_train, df_valid, config.TARGET, cat_features, ord_features)

    # encode all features (they are all categorical)
    lr_model.encode()

    # fit model on training data
    lr_model.fit()

    # predict on validation data
    valid_preds = lr_model.predict()

    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)

    # print auc
    print(f"Fold = {fold}, AUC = {auc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="lr")

    args = parser.parse_args()

    model = None
    if args.model == "lr":
        model = LogisticRegressionModel
    elif args.model == "rf":
        model = DecisionTreeModel
    elif args.model == "svd":
        model = DecisionTreeModelSVD
    elif args.model == "xgb":
        model = XGBoost
    else:
        raise argparse.ArgumentError(
            "Only 'lr' (logistic regression)"
            ", 'rf' (random forest)"
            ", 'svd' (random forest with truncate svd)"
            ", 'xgb' (XGBoost)"
            " models are supported"
        )

    for fold_ in range(5):
        run(fold_, model)
