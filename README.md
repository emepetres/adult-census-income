# Categorical Feature Encoding Challenge II

The prediction task is to determine whether a person makes over $50K a year.
Dataset [adult-census-income](https://www.kaggle.com/datasets/uciml/adult-census-income) from kaggle.

NOTES:

* Logistic regression performs well on categorical variables only. XGBoost doesn't really improve here.
* Without removing ordinal features, logistics regresion performs very badly because they need to be normalized first.
* XGBoost performns very well with ordinal variables, as tree models don't need ordinal normalization.
* Naive feature engineering combining categories in two improved xgb a little bit. Decreasing to 5 max depth improved a little bit extra.

## Train

```bash
cd src
conda activate ml
python create_folds.py
python -W ignore train.py --model=[xgb]  # [lr|rf|svd|xgb]
```
