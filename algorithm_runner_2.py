from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import DistanceMetric
from numpy import mean


class AlgorithmRunner2:
    def __init__(self, name, k=15, metric=DistanceMetric.get_metric('euclidean'), p=2):
        if name == 'KNN':
            self.algorithm = KNeighborsClassifier(k, p=p)
        elif name == 'Rocchio':
            self.algorithm = NearestCentroid(metric=metric)

        self.name = name

    def run(self, data, folds=5, selection_type='None'):
        """
        run 'self.algorithm' with cross validation. print precision, recall and accuracy.
        :param data: Data object to run the algorithm on
        :param folds: number of folds for the cross validation
        :return: None
        """
        cv = data.split_to_k_folds(folds)
        results = cross_validate(self.algorithm, self.feature_selection(data, selection_type), data.data['salary'],
                                 scoring=('precision', 'recall', 'accuracy'), cv=cv)

        print(f"{self.name}_2 classifier: {mean(results['test_precision'])}, {mean(results['test_recall'])},"
              f" {mean(results['test_accuracy'])}")

    def feature_selection(self, d, type):
        if type == 'None':
            return d.data.drop('salary', axis='columns')
        if type == 'VarianceThreshold':
            select = VarianceThreshold(0.05)
        elif type == 'SelectKBest':
            select = SelectKBest(score_func=mutual_info_classif, k=10)
        elif type == 'GenericUnivariateSelect':
            select = GenericUnivariateSelect(mutual_info_classif)

        select.fit(d.data.drop('salary', axis='columns'), d.data['salary'])

        # support = select.get_support(indices=True)
        # index_list = []
        # for i in range(len(support)):
        #     index_list.append(support[i])
        #
        # for index in index_list:
        #     print(d.data.columns[index])

        return select.fit_transform(d.data.drop('salary', axis='columns'), d.data['salary'])
