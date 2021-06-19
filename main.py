import sys
import data
import algorithm_runner


def main(argv):
    d = data.Data(argv[1])

    print(f"united capital: ")
    d.custom_preprocess(inplace=True, unite_capital=True, pivot_cat=True)

    knn_5 = algorithm_runner.AlgorithmRunner(name='KNN', k=15)
    knn_5.run(d)

    rocchio = algorithm_runner.AlgorithmRunner(name='Rocchio')
    rocchio.run(d)

    print(f"separated capital: ")
    d.custom_preprocess(inplace=True, unite_capital=False, pivot_cat=True)

    knn_5 = algorithm_runner.AlgorithmRunner(name='KNN', k=15)
    knn_5.run(d)

    rocchio = algorithm_runner.AlgorithmRunner(name='Rocchio')
    rocchio.run(d)


if __name__ == '__main__':
    main(sys.argv)
