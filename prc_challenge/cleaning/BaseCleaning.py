class BaseCleaning(object):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, FuelSegment_X, FuelSegment_Y, FlightList, Airport, Flight):
        raise NotImplementedError