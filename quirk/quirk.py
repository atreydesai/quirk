from quirk.ts import TimeSeries, Interval

from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def format_datetime(dt):
    return dt.strftime('%Y-%m-%d %H:%M:%S')


class Report:

    def __init__(self, X, y, intervals):
        self.X = X
        self.y = y
        self.intervals = intervals


class Quirk:

    def __init__(self, ts, k):
        self.ts = ts
        self.k = k
        self.training = []
        self.features = []

    def add_training(self, interval):
        self.training.append(interval)

    def add_feature(self, feature):
        self.features.append(feature)

    def classify(self):
        dp = self.ts.datapoints
        X, y = [], []

        # build training X
        for i in range(0, len(dp) - self.k + 1):
            window = TimeSeries(dp[i:i + self.k])

            # get feature
            vector = []
            for feature in self.features:
                vector.extend(feature().evaluate(window))

            X.append(vector)

        # build training Y
        for i in range(0, len(dp) - self.k + 1):
            window = TimeSeries(dp[i:i + self.k])

            # label training datapoints
            found = False
            for wdp in window.datapoints:
                dt = wdp.index
                for training_interval in self.training:
                    if training_interval.start <= dt and dt <= training_interval.end:
                        found = True

            y.append(int(found))

        # scale data
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

        # classify
        model = SVC()
        model.fit(X, y)
        predicted = model.predict(X)

        # cross-validation
        scores = cross_val_score(model, X, y, cv=10)

        windows = [i for (i, item) in enumerate(predicted) if item == 1]
        actuals = []
        for i in range(0, len(dp) - self.k + 1):
            window = TimeSeries(dp[i:i + self.k])
            if i in windows:
                start, end = window.datapoints[0].index, window.datapoints[-1].index
                actuals.append(Interval(start, end))

        merged = self._merge_intervals(actuals)

        return Report(X, y, merged)

    def _merge_intervals(self, intervals):
        if len(intervals) <= 1:
            return intervals

        merged = []
        a, b = intervals[0].start, intervals[0].end

        for i in range(1, len(intervals)):
            c, d = intervals[i].start, intervals[i].end
            if c <= b:
                b = max(b, d)
            else:
                merged.append(Interval(a, b))
                a, b = c, d

        merged.append(Interval(a, b))

        return merged