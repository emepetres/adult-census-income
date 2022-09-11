# Adult Census Income Dataset

Dataset where the prediction task is to determine whether a person makes over $50K a year.
[adult-census-income](https://www.kaggle.com/datasets/uciml/adult-census-income) from kaggle.

NOTES:

TL;DR; XGBoost works best, and improves a little bit with target encoding

* Logistic regression performs well on categorical variables only. XGBoost doesn't really improve here.
* Without removing ordinal features, logistics regresion performs very badly because they need to be normalized first.
* XGBoost performns very well with ordinal variables, as tree models don't need ordinal normalization.
* Naive feature engineering combining categories in two improved xgb a little bit. Decreasing to 5 max depth improved a little bit extra.
* Aplying target encoding over XGBoost improves the model a little bit further.

## Train

```bash
cd src
conda activate ml
python create_folds.py
python -W ignore train.py [--model=xgb]  # [lr|rf|svd|xgb]
python -W ignore train_mean_target.py
```
