import numpy as np
from sklearn.linear_model import LinearRegression

class LinearBiasCorrector:
    def __init__(self):
        self.rotation_lr = None
        self.intercept_lr = None

    def fit(self, true_train, pred_train):
        ln = LinearRegression()
        ln.fit(np.array(true_train).reshape(-1, 1), np.array(pred_train))
        slope, intercept = ln.coef_[0], ln.intercept_
        self.rotation_lr, self.intercept_lr = self.transform_line_linreg(
            slope, 1, intercept
        )

    def predict(self, pred_train):
        if self.rotation_lr is None or self.intercept_lr is None:
            raise ValueError("Model must be fitted before predicting.")
        return pred_train * self.rotation_lr + self.intercept_lr

    def transform_line_linreg(self, slope1, slope2, intercept):
        rotation_factor = slope2 / slope1
        d = -intercept / slope1
        return rotation_factor, d