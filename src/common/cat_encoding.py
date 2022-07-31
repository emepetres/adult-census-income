import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import make_column_transformer
from scipy import sparse


def encode_to_onehot(
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


def encode_to_values(
    df_train: pd.DataFrame, df_valid: pd.DataFrame, features: List[str]
):
    """
    Encode target labels with value between 0 and n_classes-1.
    Transforms inline.

    Used only on tree-based algorithms
    """
    # fit LabelEncoder on training + validation features
    # (do this way as it would be with training + testing data)
    full_data = pd.concat([df_train[features], df_valid[features]], axis=0)

    for col in features:
        # initialize LabelEncoder for each feature column
        lbl = LabelEncoder()

        # fit the label encoder on all data
        lbl.fit(full_data[col])

        # transform all the data
        df_train.loc[:, col] = lbl.transform(df_train[col])
        df_valid.loc[:, col] = lbl.transform(df_valid[col])
