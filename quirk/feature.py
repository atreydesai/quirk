from abc import ABC, abstractmethod
import numpy as np

class Feature(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def cardinality(self, k):
        pass

    @abstractmethod
    def evaluate(self, ts):
        pass


class Point(Feature):

    def cardinality(self, k):
        return k

    def evaluate(self, ts):
        vector = np.empty((len(ts.datapoints),))
        vector.fill(np.nan)

        for i in range(len(ts.datapoints)):
            vector[i] = ts.datapoints[i].value

        return vector


class Mean(Feature):

    def cardinality(self, k):
        return 1

    def evaluate(self, ts):
        vector = np.empty((len(ts.datapoints),))
        vector.fill(np.nan)

        values = np.array([dp.value for dp in ts.datapoints])
        vector[0] = np.mean(values)

        return vector