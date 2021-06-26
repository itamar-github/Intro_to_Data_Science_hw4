import sys
import data_2
import algorithm_runner_2
import data
import algorithm_runner


def main(argv):
    d = data.Data(argv[1])
    d2 = data_2.Data2(argv[1])

    d.preprocess()
    d2.preprocess()

    print("base knn:\nKNN classifier: 0.9618016626972935, 0.9881731479599181, 0.961639189585485")
    # knn_5 = algorithm_runner.AlgorithmRunner(name='KNN', k=15)
    # knn_5.run(d)

    # print("with variance threshold 0.05")
    # knn_5_2 = algorithm_runner_2.AlgorithmRunner2(name='KNN', k=15)
    # knn_5_2.run(d2)
    #
    # print("with select k best, k = 10")
    # knn_5_2.run(d2, selection_type="SelectKBest")
    #
    # print("with GenericUnivariateSelect:")
    # knn_5_2.run(d2, selection_type='GenericUnivariateSelect')

    print("euclidean: ")
    rocchio = algorithm_runner.AlgorithmRunner(name='Rocchio')
    rocchio.run(d)
    print("manhattan: ")
    rocchio_2 = algorithm_runner_2.AlgorithmRunner2(name='Rocchio', metric='manhattan')
    rocchio_2.run(d2)
    print("chebyshev: ")
    rocchio_2 = algorithm_runner_2.AlgorithmRunner2(name='Rocchio', metric='chebyshev')
    rocchio_2.run(d2)
    print("minkowski: ")
    rocchio_2 = algorithm_runner_2.AlgorithmRunner2(name='Rocchio', metric='minkowski')
    rocchio_2.run(d2)


if __name__ == '__main__':
    main(sys.argv)
