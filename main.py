import sys
import data
import algorithm_runner


def main(argv):

    # print("Question 1:")
    #
    # d = data.Data(argv[1])
    # d.preprocess()
    #
    # knn = algorithm_runner.AlgorithmRunner('KNN')
    # knn.run(d)
    #
    # rocchio = algorithm_runner.AlgorithmRunner('Rocchio')
    # rocchio.run(d)
    #
    # print()
    #
    # print("Question 2:")
    # d = data.Data(argv[1])
    # d.custom_preprocess(inplace=True, pivot_cat=True, norm='standard')
    #
    # knn = algorithm_runner.AlgorithmRunner('KNN', k=25)
    # knn.q2_run(d, num=10)
    #
    # d.custom_preprocess(inplace=True, pivot_cat=True, norm='l1')
    #
    # rocchio = algorithm_runner.AlgorithmRunner('Rocchio')
    # rocchio.q2_run(d, num=20)

    d = data.Data(argv[1])
    for l in ['l1', 'l2', 'max', 'standard', 'minmax']:
        print(f"------------ {l} ------------")
        d.custom_preprocess(inplace=True, pivot_cat=True, norm=l)
        for i in range(5, 51, 5):
            print(f"select {i} best features:")
            for j in range(10, 31):
                print(f"K = {j}")
                knn = algorithm_runner.AlgorithmRunner('KNN', k=j)
                knn.q2_run(d, num=i)
            rocchio = algorithm_runner.AlgorithmRunner('Rocchio')
            rocchio.q2_run(d, num=i)




if __name__ == '__main__':
    main(sys.argv)
