class BasePostCleaning(object):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, FuelSegment_train, FlightList_train, Predictions):
        raise NotImplementedError