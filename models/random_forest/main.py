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

        print(self.regressor.feature_importances_)

        importances = self.regressor.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.regressor.estimators_],
                     axis=0)

        # Extra
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")

        for f in range(x.shape[1]):
            print("%d. feature %d (%f)" %
                  (f + 1, indices[f], importances[indices[f]]))

        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(x.shape[1]), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(x.shape[1]),
                   self.features[indices], rotation='vertical')
        plt.xlim([-1, x.shape[1]])
        plt.tight_layout()
        plt.show()

    def predict(self, x, y):
        return mean_absolute_error(y, self.regressor.predict(x))
