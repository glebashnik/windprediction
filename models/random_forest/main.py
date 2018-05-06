from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error  


class RandomForest_featureimportance:

    def __init__(self):
        pass

    def train(self, x, y):
        self.regressor = RandomForestRegressor(bootstrap=True, criterion='mae', max_depth=20,
           max_features='auto', max_leaf_nodes=None, n_estimators=500, n_jobs=4,
           oob_score=False, random_state=0, verbose=1, warm_start=False)
        self.regressor.fit(x, y)

    def predict(self, x, y):
        return mean_absolute_error(y, self.regressor.predict(x))