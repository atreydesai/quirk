import unittest

from quirk.quirk import Quirk
from quirk.feature import Point
from quirk.ts import TimeSeries, parse_training

class TestQuirk(unittest.TestCase):

    def test_integration(self):
        timeseries = TimeSeries()
        timeseries.import_csv('dataset.csv')

        q = Quirk(timeseries, 3)
        q.add_feature(Point)

        training = parse_training('training.csv')
        for item in training:
            q.add_training(item)

        report = q.classify()
        intervals = report.intervals

        for expected in training:
            a, b = expected.start, expected.end

            found = False

            for actual in intervals:
                c, d = actual.start, actual.end
                start, end = min(a, c), max(b, d)

                if start <= a and b <= end:
                    found = True
                    break

            self.assertTrue(found)

if __name__ == '__main__':
    unittest.main()