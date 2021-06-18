import sys
import data
import algorithm_runner


def main(argv):
    d = data.Data(argv[1])
    d.preprocess()

    algo = algorithm_runner.AlgorithmRunner(name='KNN', k=5)
    algo.run(d)


if __name__ == '__main__':
    main(sys.argv)