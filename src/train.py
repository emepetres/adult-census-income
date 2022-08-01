import argparse
import itertools
import pandas as pd

import config
from common.encoding import fill_cat_with_none
from model_dispatcher import (
    CustomModel,
    LogisticRegressionModel,
    DecisionTreeModel,
    DecisionTreeModelSVD,
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


def run(fold: int, model: CustomModel):
    # load the full training data with folds
    df = pd.read_csv(config.TRAINING_FILE)

    # map targets to 0s and 1s
    target_mapping = {"<=50K": 0, ">50K": 1}
    df.loc[config.TARGET, :] = df.loc[:, config.TARGET].map(target_mapping)

    # FIXME: map introduces a new row filled with nan values¿? #122
    df = df.iloc[:-1]

    ord_features = ["age", "fnlwgt", "capital.gain", "capital.loss", "hours.per.week"]
    # # # Drop numerical columns for simplicity
    # # df.drop(ord_features, inplace=True, axis=1)

    # all columns are features except target and kfold columns
    features = [f for f in df.columns if f not in (config.TARGET, "kfold")]
    cat_features = [col for col in features if col not in ord_features]

    # # # add new features
    df = feature_engineering(df, cat_features)
    features = [f for f in df.columns if f not in (config.TARGET, "kfold")]
    cat_features = [col for col in features if col not in ord_features]

    fill_cat_with_none(df, cat_features)

    # initialize model
    custom_model = model(df, fold, config.TARGET, cat_features, ord_features)

    # encode all features
    custom_model.encode()

    # fit model on training data
    custom_model.fit()

    # predict on validation data and get roc auc score
    auc = custom_model.predict_and_score()

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
