import pandas as pd
import numpy as np
import sklearn as sk


class Data:
    def __init__(self, path):
        """
        declare 'data' data member
        """
        self.data = None
        self.path = path

    def preprocess(self):
        """
        preprocess data as follows:
        - read all features except "fnlwgt"
        - filter records with "?" values
        - transform categorical features to indicator sets
        - transform "salary"
        - normalize continuous features with MinMax normalizer
        :return:
        """
        # read csv file into 'data' data frame
        # strip data of leading or tailing whitespaces
        self.data = pd.read_csv(self.path, skipinitialspace=True)
        # drop "fnlwgt" column
        self.data.drop(columns="fnlwgt", inplace=True)
        # replace all "?" with None
        self.data.replace('?', np.nan, inplace=True)
        # filter rows with NaNs
        self.data.dropna(inplace=True)

        categorical_features = ['workclass', 'education', 'martial-status', 'occupation', 'relationship', 'race',
                                     'sex', 'native-country']
        continuous_features = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

        # transform categorical features into dummies
        self.data = pd.get_dummies(self.data, columns=categorical_features)

        # transform salary values: '<=50K' -> 1, '>50K' -> 0
        self.data['salary'] = self.data['salary'].apply(lambda s: 1 if s == '<=50K' else 0)

        # normalize continuous features via MinMax normalization
        for feature in continuous_features:
            min_value = self.data[feature].min()
            max_value = self.data[feature].max()
            self.data[feature] = (self.data[feature] - min_value) / (max_value - min_value)

    @staticmethod
    def split_to_k_folds(k):
        """
        return sklearn KFold object with k folds
        :param k: number of folds
        :return: sklearn KFold object
        """
        kf = sk.model_selection.KFold(n_splits=k, shuffle=True, random_state=10)

    def head(self):
        print(self.data.head(30))
