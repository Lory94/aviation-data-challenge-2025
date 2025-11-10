from sklearn.dummy import DummyRegressor


class TrainSetMean_v0_0_0(object):

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super().__init__()

    def fit(self, X, y, column_functions):

        self.model = DummyRegressor(strategy="mean")
        self.model.fit(X=X, y=y)

    def predict(self, X):
        return self.model.predict(X)
