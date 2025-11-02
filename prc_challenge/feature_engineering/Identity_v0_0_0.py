from .FeatureEngineering import FeatureEngineering


class Identity_v0_0_0(FeatureEngineering):
    
    def transform(self, X):
        if not self.__sklearn_is_fitted__:
            raise Exception("FeatureEngineering has to be fit before using transform.")
        return X
