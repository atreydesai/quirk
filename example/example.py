from quirk.quirk import Quirk
from quirk.feature import Point
from quirk.ts import TimeSeries, parse_training

import matplotlib.pyplot as plt

timeseries = TimeSeries()
timeseries.import_csv('../tests/dataset.csv')

q = Quirk(timeseries, 3)
q.add_feature(Point())

for item in parse_training('../tests/training.csv'):
    q.add_training(item)

report = q.classify()

intervals = report.intervals
for interval in intervals:
    print(interval.start, interval.end)

plt.figure(figsize=(20, 8))

plt.plot(report.X[100:250], report.y[100:250], color='#5ACAE4')

for i, interval in enumerate(intervals):
    if 2 <= i < 8:
        a, b = interval.start, interval.end
        plt.axvspan(a, b, alpha=0.1, color='#c6c6c6', linewidth=0)

plt.savefig('test.png')
