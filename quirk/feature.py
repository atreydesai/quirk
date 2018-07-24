from abc import ABC, abstractmethod

class Feature(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def evaluate(self, ts):
        pass


class Point(Feature):

    def evaluate(self, ts):
        return [dp.value for dp in ts.datapoints]


class Mean(Feature):

    def evaluate(self, ts):
        values = [dp.value for dp in ts.datapoints]
        return [sum(values) / len(values)]