class SupervisedDataset(object):

    def __init__(
        self,
        train_X, train_Y,
        test_X, test_Y,
        features,
    ):
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y

        self.features = features
