from quirk.quirk import Quirk
from quirk.feature import Point
from quirk.ts import TimeSeries, parse_training

import matplotlib.pyplot as plt

timeseries = TimeSeries()
timeseries.import_csv('dataset.csv')

q = Quirk(timeseries, 3)
q.add_feature(Point)

for item in parse_training('training.csv'):
    q.add_training(item)

report = q.classify()

intervals = report.intervals
for interval in intervals:
    print(interval.start, interval.end)