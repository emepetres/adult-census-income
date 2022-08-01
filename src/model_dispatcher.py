from typing import List
import pandas as pd

from sklearn import ensemble, linear_model, metrics
import xgboost as xgb

from common.encoding import (
    encode_to_onehot,
    reduce_dimensions_svd,
    encode_to_values,
)


class CustomModel:
    def __init__(
        self,
        data: pd.DataFrame,
        fold: int,
        target: str,
        cat_features: List[str],
        ord_features: List[str],
    ):
        self.data = data
        self.fold = fold
        self.target = target
        self.cat_features = cat_features
        self.ord_features = ord_features

        self.features = cat_features + ord_features

    def encode(self):
        """Transforms data into x_train & x_valid"""
        pass

    def fit(self):
        """Fits the model on x_valid and train target"""
        pass

    def predict_and_score(self) -> float:
        """Predicts on x_valid data and score using AUC"""
        # predict on validation data
        # we need the probability values as we are calculating AUC
        # we will use the probability of 1s
        valid_preds = self.model.predict_proba(self.x_valid)[:, 1]

        return metrics.roc_auc_score(self.df_valid.income.values, valid_preds)


class LogisticRegressionModel(CustomModel):
    def encode(self):
        # get training & validation data using folds
        self.df_train = self.data[self.data.kfold != self.fold].reset_index(drop=True)
        self.df_valid = self.data[self.data.kfold == self.fold].reset_index(drop=True)

        # get encoded dataframes with new categorical features
        df_cat_train, df_cat_valid = encode_to_onehot(
            self.df_train, self.df_valid, self.cat_features
        )

        # we have a new set of categorical features
        encoded_features = df_cat_train.columns.to_list() + self.ord_features

        # TODO: normalize ordinal features!

        dfx_train = pd.concat([df_cat_train, self.df_train[self.ord_features]], axis=1)
        dfx_valid = pd.concat([df_cat_valid, self.df_valid[self.ord_features]], axis=1)

        self.x_train = dfx_train[encoded_features].values
        self.x_valid = dfx_valid[encoded_features].values

    def fit(self) -> pd.DataFrame:
        self.model = linear_model.LogisticRegression()

        # fit model on training data
        self.model.fit(self.x_train, self.df_train.loc[:, self.target].values)


class DecisionTreeModel(CustomModel):
    def encode(self):
        encode_to_values(self.data, self.cat_features)

        # get training & validation data using folds
        self.df_train = self.data[self.data.kfold != self.fold].reset_index(drop=True)
        self.df_valid = self.data[self.data.kfold == self.fold].reset_index(drop=True)

        self.x_train = self.df_train[self.features].values
        self.x_valid = self.df_valid[self.features].values

    def fit(self) -> pd.DataFrame:
        self.model = ensemble.RandomForestClassifier(n_jobs=-1)

        # fit model on training data
        self.model.fit(self.x_train, self.df_train.loc[:, self.target].values)


class DecisionTreeModelSVD(DecisionTreeModel):
    def encode(self):
        super().encode()

        self.x_train, self.x_valid = reduce_dimensions_svd(
            self.x_train, self.x_valid, 120
        )


class XGBoost(DecisionTreeModel):
    def fit(self) -> pd.DataFrame:
        self.model = xgb.XGBClassifier(
            n_jobs=-1, verbosity=0, max_depth=5  # , n_estimators=200
        )

        # fit model on training data
        self.model.fit(self.x_train, self.df_train.loc[:, self.target].values)


class XGBoostEncoded(XGBoost):
    def __init__(self, data: pd.DataFrame, fold: int, target: str, features: List[str]):
        super().__init__(data, fold, target, [], [])

        self.features = features

    def encode(self):
        # get training & validation data using folds
        self.df_train = self.data[self.data.kfold != self.fold].reset_index(drop=True)
        self.df_valid = self.data[self.data.kfold == self.fold].reset_index(drop=True)

        self.x_train = self.df_train[self.features].values
        self.x_valid = self.df_valid[self.features].values
