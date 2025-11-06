from .TabularDataset import TabularDataset

class SupervisedTabularDataset(TabularDataset):

    def __init__(
        self,
        data,
        y_column,
    ):
        super().__init__(
            data=data,
        )
        self.y_column = y_column
