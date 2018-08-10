from quirk.ts import TimeSeries, Interval

import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
        feature_cardinality = self._feature_cardinality()
        rolling_windows = self._rolling_windows()

        X = np.zeros((rolling_windows, feature_cardinality))
        y = np.zeros((rolling_windows,))

        # build training X
        for i in range(rolling_windows):
            window = TimeSeries(dp[i:i + self.k])

            # build individual feature vector
            vector = np.zeros((len(self.features), self.k))
            for j, feature in enumerate(self.features):
                vector[j] = feature.evaluate(window)

            vector = vector.reshape((1, self.k * len(self.features)))
            vector = vector[~np.isnan(vector)]
            X[i] = vector

        # build training Y
        for i in range(rolling_windows):
            window = TimeSeries(dp[i:i + self.k])

            # label training datapoints
            found = 0
            for wdp in window.datapoints:
                dt = wdp.index
                for training_interval in self.training:
                    if training_interval.start <= dt and dt <= training_interval.end:
                        found = 1

            y[i] = found

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

        # identify anomalous intervals
        actuals = []
        for i in range(rolling_windows):
            if predicted[i]:
                start, end = dp[i].index, dp[i + self.k - 1].index
                actuals.append(Interval(start, end))

        # merge overlapping intervals
        merged = self._merge_intervals(actuals)

        return Report(self.ts.index(), self.ts.values(), merged)

    def _rolling_windows(self):
        return len(self.ts.datapoints) - self.k + 1

    def _feature_cardinality(self):
        features = 0
        for feature in self.features:
            features += feature.cardinality(self.k)
        return features

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
