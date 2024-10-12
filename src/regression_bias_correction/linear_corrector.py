import numpy as np
from sklearn.linear_model import LinearRegression

class LinearBiasCorrection:
    def __init__(self):
        self.rotation_lr = None
        self.intercept_lr = None

    def fit(self, true_train, pred_train):
        ln = LinearRegression()
        ln.fit(np.array(true_train).reshape(-1, 1), pred_train)
        slope, intercept = ln.coef_[0], ln.intercept_
        self.rotation_lr, self.intercept_lr = self.rotate_line_lin_regr(
            slope, 1, intercept)

    def predict(self, pred_train):
        if self.rotation_lr is None or self.intercept_lr is None:
            raise ValueError("Model must be fitted before predicting.")
        return pred_train * self.rotation_lr + self.intercept_lr

    def rotate_line_lin_regr(self, slop1, slop2, intercept):
        rotation_factor = slop2 / slop1
        d = -intercept / slop1
        return rotation_factor, d