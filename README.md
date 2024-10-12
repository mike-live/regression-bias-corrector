# Regression bias corrector

Package is aimed to correct regression fit by any machine learning model that underestimate high values and overestimate low values of outcome variable (1D case). The problem usually arises in case when metrics like MSE and MAE shows better results in biased predictions.

# Algorithm

Now is implemeneted only linear correction for 1D case by the Linear Regression approach.

The idea is to linearly transform the `y_pred` to change the coefficient of linear regression between `y_true` and `y_pred` to 1. It is corresponds to the diagonal on plane `y_true` and `y_pred` that means equal pace of increasing `y_pred` with increasing `y_true`. 
- `y_true` -- actual values
- `y_pred` -- predicted values


# How to install
```bash
pip install regression-bias-corrector
```

# How to use
```python
import 
```