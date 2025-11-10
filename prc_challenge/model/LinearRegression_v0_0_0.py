from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


class LinearRegression_v0_0_0(object):

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super().__init__()

    def fit(self, X, y, column_functions):

        preprocessing = ColumnTransformer(
            [
                (
                    'imputation', 
                    SimpleImputer(strategy='mean'), 
                    column_functions["numerical"],
                ),
                (
                    'category_encoding', 
                    OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
                    column_functions["categorical"],
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

    def predict(self, X):
        return self.pipeline.predict(X)
