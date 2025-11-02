from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

class LinearRegression_v0_0_0(BaseEstimator, RegressorMixin):

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super().__init__()

    def fit(self, X, y):

        print(self.kwargs)

        preprocessing = ColumnTransformer(
            [
                (
                    'imputation', 
                    SimpleImputer(strategy='mean'), 
                    self.kwargs["interval_numeric"]+self.kwargs["ratio_numeric"],
                ),
                (
                    'category_encoding', 
                    OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
                    self.kwargs["nominal_category"]+self.kwargs["ordinal_category"],
                ),
            ],
            remainder='drop',
        ).set_output(transform='pandas')
        model = SklearnLinearRegression()

        self.pipeline = Pipeline(steps=[
            ("preprocessing", preprocessing),
            ("model", model),
        ])

        self.pipeline = self.pipeline.fit(X=X, y=y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)
