from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import cross_validate
from numpy import mean


class AlgorithmRunner:
    def __init__(self, name, k=5):
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
        results = cross_validate(self.algorithm, data.data, data.data['salary'],
                                 scoring=('precision', 'recall', 'accuracy'), cv=cv)

        print(f"{self.name} classifier: {mean(results['test_precision'])}, {mean(results['test_recall'])},"
              f" {mean(results['test_accuracy'])}")
