import copy
import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import make_column_transformer
from scipy import sparse

import config


def fill_cat_with_none(data: pd.DataFrame, cat_features: List[str]):
    """
    fill all NaN values with NONE
    # note that I am converting all columns to "strings"
    # it doesn't matter because all are categories
    """
    for col in cat_features:
        data.loc[:, col] = data[col].astype(str).fillna("NONE")


def encode_to_onehot(  # one-hot encoding
    df_train: pd.DataFrame, df_valid: pd.DataFrame, features: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Best sparse optimization, but slow on trees algorithms

    Returns dataframes with features transformed to one-hot features,
    and the new created features
    """

    # initialize OneHotEncoder from scikit-learn
    transformer = make_column_transformer(
        (OneHotEncoder(), features), remainder="passthrough"
    )

    # fit ohe on training + validation features
    # (do this way as it would be with training + testing data)
    full_data = pd.concat([df_train[features], df_valid[features]], axis=0)
    transformer.fit(full_data[features])

    # transform training & validation data
    tdf_train = pd.DataFrame.sparse.from_spmatrix(
        transformer.transform(df_train[features]),
        columns=transformer.get_feature_names_out(),
    )
    tdf_valid = pd.DataFrame.sparse.from_spmatrix(
        transformer.transform(df_valid[features]),
        columns=transformer.get_feature_names_out(),
    )

    # return training & validation features
    return (tdf_train, tdf_valid)


def encode_to_values(data: pd.DataFrame, features: List[str]):  # Label Encoding
    """
    Encode target labels with value between 0 and n_classes-1.
    Transforms inline.

    Used only on tree-based algorithms
    """
    # fit LabelEncoder on training + validation features
    # (do this way as it would be with training + testing data)

    for col in features:
        # initialize LabelEncoder for each feature column
        lbl = LabelEncoder()

        # fit the label encoder on all data
        lbl.fit(data[col])

        # transform all the data
        data.loc[:, col] = lbl.transform(data[col])


def reduce_dimensions_svd(
    x_train: sparse.csr_matrix, x_valid: sparse.csr_matrix, n_components: int
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Used over a OneHotEnconding to reduce its size"""
    # initialize TruncatedSVD
    # we are reducing the data to 120 components
    svd = TruncatedSVD(n_components=n_components)

    # fit svd on full sparse training data
    full_sparse = sparse.vstack((x_train, x_valid))
    svd.fit(full_sparse)

    # transform sparse training data
    x_train = svd.transform(x_train)

    # transform sparse valid data
    x_valid = svd.transform(x_valid)

    return (x_train, x_valid)


def mean_target_encoding(
    data: pd.DataFrame, num_cols: List[str], cat_cols: List[str], folds: int = 5
) -> pd.DataFrame:
    """
    Preprocess all data using mean target encoding

    Use before folds separation
    """

    # make a copy of dataframe
    df = copy.deepcopy(data)

    # label encode the features
    encode_to_values(df, cat_cols)

    # a list to store 5 validation dataframes
    encoded_dfs = []

    # go over all folds
    for fold_ in range(folds):
        # fetch training and validation data
        df_train = df[df.kfold != fold_].reset_index(drop=True)
        df_valid = df[df.kfold == fold_].reset_index(drop=True)
        # for all feature columns
        for column in cat_cols + num_cols:
            # create dict of category:mean target
            mapping_dict = dict(df_train.groupby(column)[config.TARGET].mean())
            # column_enc is the new column we have with mean encoding
            df_valid.loc[:, column + "_enc"] = df_valid[column].map(mapping_dict)
        # append to our list of encoded validation dataframes
        encoded_dfs.append(df_valid)

    # create full data frame again and return
    encoded_df = pd.concat(encoded_dfs, axis=0)
    return encoded_df
