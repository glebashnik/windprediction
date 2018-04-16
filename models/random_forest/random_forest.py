from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt


class RandomForest:

    def __init__(self, features):
        self.features = features

    def train(self, x, y):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error
        import numpy as np
        import matplotlib.pyplot as plt

        print('Fitting random forest')

        self.regressor = RandomForestRegressor(bootstrap=True, criterion='mae', max_depth=20,
                                               max_features='auto', max_leaf_nodes=None, n_estimators=500, n_jobs=2,
                                               oob_score=False, random_state=0, verbose=1, warm_start=False)
        self.regressor.fit(x, y)

    def test(self, x, y):
        return mean_absolute_error(y, self.regressor.predict(x))
