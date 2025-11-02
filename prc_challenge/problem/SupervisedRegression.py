from sklearn.metrics import mean_squared_error
from ..data.SupervisedDataset import SupervisedDataset
from ..utils import load_objects_from_config
from sklearn.pipeline import Pipeline


class SupervisedRegression(object):
    """A classical ML supervised regression problem

    Args:
        train_X (pandas.DataFrame): TODO
        train_Y (pandas.Series): TODO
    """

    def __init__(
        self,
        supervised_dataset:SupervisedDataset,
    ):
        self.train_X = supervised_dataset.train_X
        self.train_Y = supervised_dataset.train_Y
        self.__test_X = supervised_dataset.test_X
        self.__test_Y = supervised_dataset.test_Y

    def evaluate(
        self,
        inference_pipeline,
    ):
        
        metrics = {}
        
        # Train MSE
        y_pred = inference_pipeline.predict(self.train_X)
        y_true = self.train_Y
        metrics["mse(train)"] = mean_squared_error(y_pred=y_pred, y_true=y_true)

        # Test MSE
        y_pred = inference_pipeline.predict(self.__test_X)
        y_true = self.__test_Y
        metrics["mse(train)"] = mean_squared_error(y_pred=y_pred, y_true=y_true)
        
        print(metrics)

    def solve_using(self, config):

        train_X, train_Y = self.train_X, self.train_Y
        objects = load_objects_from_config(config)
        feature_engineering = objects["feature_engineering"](**objects["feature_engineering_kwargs"])
        train_X = feature_engineering.fit_transform(X=train_X)

        model = objects["model"](**objects["model_kwargs"])
        model.fit(X=train_X, y=train_Y)

        inference_pipeline = Pipeline(steps=[
            objects["feature_engineering"],
            model,
        ])

        return inference_pipeline
