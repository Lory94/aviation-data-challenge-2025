class BaseFeatureEngineering(object):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, FuelSegment_X, FuelSegment_Y, FlightList, Airport, Flight):
        pass

    def __call__(self, FuelSegment_X, FuelSegment_Y, FlightList, Airport, Flight, column_functions):
        raise NotImplementedError
