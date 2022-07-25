# Categorical Feature Encoding Challenge II

The prediction task is to determine whether a person makes over $50K a year.
Dataset [adult-census-income](https://www.kaggle.com/datasets/uciml/adult-census-income) from kaggle.

NOTES:

## Train

```bash
cd src
conda activate ml
python create_folds.py
python -W ignore train.py --model=[lr|rf|svd|xgb]
```
