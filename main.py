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

    for i in range(3, 11):
        for j in range(1, 11):
            print(f"k = {i}, p = {j}")
            knn = algorithm_runner_2.AlgorithmRunner2('KNN', k=i, p=j)
            knn.run(d2)


if __name__ == '__main__':
    main(sys.argv)
