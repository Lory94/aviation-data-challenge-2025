from sklearn.pipeline import Pipeline

class FeatureEngineering(Pipeline):

    def fit(self, X):
        pass

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
