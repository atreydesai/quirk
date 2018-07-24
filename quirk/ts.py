import dateutil.parser

def parse_timestamp(timestamp):
    return dateutil.parser.parse(timestamp, ignoretz=False)


def parse_training(path):
    training = []

    try:
        f = open(path)
        lines = [line for line in f.read().split('\n') if len(line)]
        for line in lines:
            start, end = line.split(',')
            start = parse_timestamp(start)
            end = parse_timestamp(end)
            training.append(Interval(start, end))
    except FileNotFoundError:
        return []

    return training


class Interval:

    def __init__(self, start, end):
        self.start = start
        self.end = end


class Datapoint:

    def __init__(self, index, value):
        self.index = index
        self.value = value

    def __repr__(self):
        return 'Datapoint<{},{}>'.format(
            self.index.__repr__(), self.value.__repr__()
        )


class TimeSeries:

    def __init__(self, datapoints=None):
        self.datapoints = datapoints

    def index(self):
        if self.datapoints is None:
            return []
        return [dp.index for dp in self.datapoints]

    def values(self):
        if self.datapoints is None:
            return []
        return [dp.value for dp in self.datapoints]

    def series(self):
        return (self.index(), self.values())

    def import_csv(self, path):
        try:
            f = open(path)
            lines = [line for line in f.read().split('\n') if len(line)]
            datapoints = []
            for line in lines:
                index, value = line.split(',')
                timestamp = parse_timestamp(index)
                datapoints.append(Datapoint(timestamp, float(value)))
            self.datapoints = datapoints
        except:
            return False
        return True
