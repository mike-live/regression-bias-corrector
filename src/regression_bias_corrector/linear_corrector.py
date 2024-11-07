import numpy as np
from sklearn.linear_model import LinearRegression
from ._version import __version_tuple__, __version__
import yaml
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
    
    @classmethod
    def load(cls, path_to_yml):
        with open(path_to_yml, 'r') as fin:
            model_dict = yaml.full_load(fin)
        corrector = LinearBiasCorrector()
        corrector.rotation_lr = model_dict['rotation_lr']
        corrector.intercept_lr = model_dict['intercept_lr']
        return corrector
        
    def save(self, path_to_yml):
        model_dict = {
            'model_name': 'linear_bias_correction',
            'version': __version__,
            'version_tuple': __version_tuple__,
            'rotation_lr': self.rotation_lr.item(),
            'intercept_lr': self.intercept_lr.item()
        }
        with open(path_to_yml, 'w') as fout:
            yaml.dump(model_dict, fout, default_flow_style=False)
    
    def __str__(self):
        return f'Linear bias corrector: a={self.rotation_lr} b={self.intercept_lr}'