from typing import List
import pandas as pd

from sklearn import ensemble, linear_model
import xgboost as xgb

from common.cat_encoding import (
    encode_to_onehot,
    reduce_dimensions_svd,
    encode_to_values,
)


class ModelInterface:
    def encode(self):
        """Transforms data into x_train & x_valid"""
        pass

    def fit(self):
        """Fits the model on x_valid and train target"""
        pass

    def predict(self):
        """Predicts on x_valid data"""
        pass


class LogisticRegressionModel(ModelInterface):
    def __init__(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        target: str,
        cat_features: List[str],
        ord_features: List[str],
    ):
        self.df_train = train
        self.df_valid = valid
        self.target = target
        self.cat_features = cat_features
        self.ord_features = ord_features
        self.features = cat_features + ord_features

    def encode(self):
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

    def predict(self) -> pd.DataFrame:
        # predict on validation data
        # we need the probability values as we are calculating AUC
        # we will use the probability of 1s
        return self.model.predict_proba(self.x_valid)[:, 1]


class DecisionTreeModel(ModelInterface):
    def __init__(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        target: str,
        cat_features: List[str],
        ord_features: List[str],
    ):
        self.df_train = train
        self.df_valid = valid
        self.target = target
        self.cat_features = cat_features
        self.ord_features = ord_features
        self.features = cat_features + ord_features

    def encode(self):
        encode_to_values(self.df_train, self.df_valid, self.cat_features)

        self.x_train = self.df_train[self.features].values
        self.x_valid = self.df_valid[self.features].values

    def fit(self) -> pd.DataFrame:
        self.model = ensemble.RandomForestClassifier(n_jobs=-1)

        # fit model on training data
        self.model.fit(self.x_train, self.df_train.loc[:, self.target].values)

    def predict(self) -> pd.DataFrame:
        # predict on validation data
        # we need the probability values as we are calculating AUC
        # we will use the probability of 1s
        return self.model.predict_proba(self.x_valid)[:, 1]


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
