import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import cross_validate
import sklearn.feature_selection
from numpy import mean


class AlgorithmRunner:
    def __init__(self, name, k=15):
        if name == 'KNN':
            self.algorithm = KNeighborsClassifier(k)
        elif name == 'Rocchio':
            self.algorithm = NearestCentroid()

        self.name = name

    def run(self, data, folds=5):
        """
        run 'self.algorithm' with cross validation. print precision, recall and accuracy.
        :param data: Data object to run the algorithm on
        :param folds: number of folds for the cross validation
        :return: None
        """
        cv = data.split_to_k_folds(folds)
        results = cross_validate(self.algorithm, data.data.drop(columns='salary'), data.data['salary'].ravel(),
                                 scoring=('precision', 'recall', 'accuracy'), cv=cv)

        print(f"{self.name} classifier: {mean(results['test_precision'])},"
              f" {mean(results['test_recall'])},"
              f" {mean(results['test_accuracy'])}")

    def q2_run(self, data, folds=5, num=20):
        """
        run method with more free parameters for question 2.
        :param data: Data object
        :param folds: number of fold in the cross validation
        :param num: k best features to select form the data frame
        :return: None
        """
        cv = data.split_to_k_folds(folds)
        df, sal_series = self.better(data=data, num=num)
        results = cross_validate(self.algorithm, df, sal_series, scoring=('precision', 'recall', 'accuracy'), cv=cv)

        print(f"{self.name} classifier: {round(mean(results['test_precision']), 3)},"
              f" {round(mean(results['test_recall']), 3)},"
              f" {round(mean(results['test_accuracy']), 3)}")

    def better(self, data, num):
        """
        perform the following processes to tune the data to produce better accuracy measure:
        - select best 'num' features from the data set with sklearn.feature_selection.SelectKBest with parameter k=num
        - normalize/scale the data with data.normalizer
        :return: processed data frame without 'salary' column and 'salary' column
        """
        salary_column = data.data['salary'].copy()
        df = data.data.drop(columns='salary')
        selector = sklearn.feature_selection.SelectKBest(k=num)
        df = data.normalizer.fit_transform(pd.DataFrame(selector.fit_transform(df, y=salary_column)), y=salary_column)
        return df, salary_column
