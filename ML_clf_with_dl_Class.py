import numpy as np


class MlClfWithDl:

    def __init__(self, clf, dl_model, ml_opt):
        self.ml_clf = clf
        self.dl_model = dl_model
        self.ml_opt = ml_opt

    def predict(self, features_test):
        if type(features_test) is not np.ndarray:
            features_test = np.array(features_test)
        middle_output = self.dl_model.predict(features_test)
        result = self.ml_clf.predict(middle_output)
        return result
