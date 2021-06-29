import sys
import data_2
import algorithm_runner_2
import data
import algorithm_runner


def main(argv):

    print("Question 1:\nKNN classifier: 0.8696313977533052, 0.9102524346412153, 0.8301114812386867\n"
          "Rocchio classifier: 0.9116050283247954, 0.6941287277776247, 0.7197042921755279")
    # print("Question 1:")
    #
    # d = data.Data(argv[1])
    # d.preprocess()
    # d.data.to_csv()
    #
    # knn = algorithm_runner.AlgorithmRunner('KNN')
    # knn.run(d)
    #
    # rocchio = algorithm_runner.AlgorithmRunner('Rocchio')
    # rocchio.run(d)

    print("Question 2:")
    d = data.Data(argv[1])
    d.preprocess()
    d.data.to_csv()

    d.better()

    knn = algorithm_runner.AlgorithmRunner('KNN')
    knn.run(d)

    rocchio = algorithm_runner.AlgorithmRunner('Rocchio')
    rocchio.run(d)


if __name__ == '__main__':
    main(sys.argv)
