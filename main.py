import sys
import data
import algorithm_runner


def main(argv):

    print("Question 1:")

    d = data.Data(argv[1])
    d.preprocess()

    knn = algorithm_runner.AlgorithmRunner('KNN')
    knn.run(d)

    rocchio = algorithm_runner.AlgorithmRunner('Rocchio')
    rocchio.run(d)

    print("Question 2:")
    d = data.Data(argv[1])
    d.custom_preprocess(inplace=True, pivot_cat=True, norm='standard')

    knn = algorithm_runner.AlgorithmRunner('KNN', k=25)
    knn.q2_run(d, num=10)

    d.custom_preprocess(inplace=True, pivot_cat=True, norm='l1')

    rocchio = algorithm_runner.AlgorithmRunner('Rocchio')
    rocchio.q2_run(d, num=20)


if __name__ == '__main__':
    main(sys.argv)
