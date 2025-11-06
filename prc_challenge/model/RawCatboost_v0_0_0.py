from catboost import CatBoostRegressor


class RawCatboost_v0_0_0():

    def fit(self, train_X, train_Y):

        self.model = CatBoostRegressor(
            iterations=2,
            learning_rate=1,
            depth=2,
        )

        self.model.fit(train_X, train_Y)
    
    def predict(self, X):
        return self.model.predict(X)
