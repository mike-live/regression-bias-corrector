# Regression bias corrector

Package is aimed to correct regression fit by any machine learning model that underestimate high values and overestimate low values of outcome variable (1D case). The problem usually arises in case when metrics like MSE and MAE shows better results in biased predictions.

# Algorithm

Now is implemeneted only linear correction for 1D case by the Linear Regression approach.

The idea is to linearly transform the `y_pred` to change the coefficient of linear regression between `y_true` and `y_pred` to 1. It is corresponds to the diagonal on plane `y_true` and `y_pred` that means equal pace of increasing `y_pred` with increasing `y_true`. 
- `y_true` - actual values
- `y_pred` - predicted values


# How to install

### PyPI
```bash
pip install regression-bias-corrector
```

### From source
```bash
git clone https://github.com/mike-live/regression-bias-corrector.git
pip install .
```

# How to use

### Fit and predict
```python
# Import Linear Bias corrector
from regression-bias-corrector import LinearBiasCorrector

# Create instance of corrector
corrector = LinearBiasCorrector()

# Fit corrector on actual outcome variable and on predicted values of the model for train dataset
# y_train - actual values
# y_train_pred - predicted values
corrector.fit(y_train, y_train_pred)

# correct predicted values on test dataset
# y_test_pred - predicted values by model on test dataset
y_test_pred_unbiased = corrector.predict(y_test_pred)

# y_test_pred_unbiased - unbiased predicted values on test
```

### Save and load

```python
corrector.save('linear_corrector.yml')

loaded_corrector = LinearBiasCorrector.load('linear_corrector.yml')
```

# Authors
&copy; 2024 Lobachevsky University

* **Krivonosov Mikhail** - *Idea inventor* - [mike-live](https://github.com/mike-live)
* **Khabarova Tatiana** - *Algorithm implementation* - [mike-live](https://github.com/enot9910)
