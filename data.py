import pandas as pd
import numpy as np
import sklearn.feature_selection
import sklearn.model_selection
import seaborn as sn
import matplotlib.pyplot as plt


class Data:

    categorical_features = ['workclass', 'education', 'martial-status', 'occupation', 'relationship', 'race',
                            'sex', 'native-country']
    continuous_features = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    def __init__(self, path):
        """
        declare 'data' data member
        """
        self.data = None
        self.path = path
        self.categorical_features = ['workclass', 'education', 'martial-status', 'occupation', 'relationship', 'race',
                                     'sex', 'native-country']
        self.continuous_features = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

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
        self.custom_preprocess(drop_cat=False, inplace=True, unite_capital=False, flip_salary_index=False,
                               pivot_cat=True)
        # # read csv file into 'data' data frame
        # # strip data of leading or tailing whitespaces
        # self.data = pd.read_csv(self.path, skipinitialspace=True)
        # # drop "fnlwgt" column
        # self.data.drop(columns="fnlwgt", inplace=True)
        # # replace all "?" with None
        # self.data.replace('?', np.nan, inplace=True)
        # # filter rows with NaNs
        # self.data.dropna(inplace=True)
        #
        # # transform categorical features into dummies
        # self.data = pd.get_dummies(self.data, columns=self.categorical_features)
        #
        # # transform salary values: '<=50K' -> 1, '>50K' -> 0
        # self.data['salary'] = self.data['salary'].apply(lambda s: 1 if s == '<=50K' else 0)
        #
        # # normalize continuous features via MinMax normalization
        # for feature in self.continuous_features:
        #     min_value = self.data[feature].min()
        #     max_value = self.data[feature].max()
        #     self.data[feature] = (self.data[feature] - min_value) / (max_value - min_value)

    def sk_preprocess(self):
        df = self.custom_preprocess(flip_salary_index=True, pivot_cat=True)

        salary_column = df['salary'].copy()
        selector = sklearn.feature_selection.VarianceThreshold(0.20)
        df = pd.DataFrame(selector.fit_transform(df))
        df.append(salary_column)

        print(df.head(30))

        self.data = df

    @staticmethod
    def split_to_k_folds(k):
        """
        return sklearn KFold object with k folds
        :param k: number of folds
        :return: sklearn KFold object
        """
        return sklearn.model_selection.KFold(n_splits=k, shuffle=True, random_state=10)

    def head(self):
        print(self.data.head(30))

    def cont_corr_matrix(self, unite_capital=True):
        """
        print correlation matrix between the continuous features in the dataset
        :param unite_capital: boolean. if true unite capital change. otherwise don't.
        :return: show PNG of correlation matrix
        """
        cont_df = self.custom_preprocess(unite_capital=unite_capital)
        corr_matrix = cont_df.corr()
        sn.heatmap(corr_matrix, annot=True)
        plt.show()

    def custom_preprocess(self, drop_cat=False, drop_cont=False, inplace=False, unite_capital=False,
                          flip_salary_index=False, pivot_cat=False):
        """
        return a preprocessed data frame only with 'salary' and continuous features
        :param drop_cat: boolean. if true, drop all categorical features. otherwise, don't.
        :param drop_cont: boolean. if true, drop all continuous features. otherwise, don't.
        :param inplace: boolean. if true set self.data equal to df before return
        :param unite_capital: boolean. if true unite capital gain and capital loss to one column. otherwise, don't
        :param flip_salary_index: boolean. if true, transform salary values: '<=50K' -> 0, '>50K' -> 1. otherwise, do
        the opposite.
        :param pivot_cat: boolean. if true get dummies for categorical features.
        :return: pd.DataFrame object
        """
        self.reset_features_list()

        df = self.custom_read_csv()

        self.transform_salary_values(df=df, reverse=flip_salary_index)

        if unite_capital:
            self.unite_capital_change(df=df)

        self.normalize_cont_features(df=df)

        if drop_cat:
            self.drop_cat_features(df=df)

        if drop_cont:
            self.drop_cont_features(df=df)

        if pivot_cat:
            df = self.pivot_cat_features(df=df)

        if inplace:
            self.data = df

        return df

    def custom_read_csv(self):
        """
        read form csv to data frame. remove rows with NaN and drop 'fnlwgt' column.
        :return: pd.DataFrame object
        """
        # read csv file into 'data' data frame
        # strip data of leading or tailing whitespaces
        df = pd.read_csv(self.path, skipinitialspace=True)
        # drop "fnlwgt" column
        df.drop(columns="fnlwgt", inplace=True)
        # replace all "?" with None
        df.replace('?', np.nan, inplace=True)
        # filter rows with NaNs
        df.dropna(inplace=True)

        return df

    def transform_salary_values(self, df, reverse=False):
        """
        transform salary values from strings to 0/1.
        :param df: data frame to perform transformation on.
        :param reverse: boolean. if true, transform salary values: '<=50K' -> 0, '>50K' -> 1. otherwise do the opposite.
        :return: pd.DataFrame object
        """
        if reverse:
            # transform salary values: '<=50K' -> 0, '>50K' -> 1
            df['salary'] = df['salary'].apply(lambda s: 0 if s == '<=50K' else 1)
        else:
            # transform salary values: '<=50K' -> 1, '>50K' -> 0
            df['salary'] = df['salary'].apply(lambda s: 1 if s == '<=50K' else 0)

        return df

    def unite_capital_change(self, df):
        """
        unite capital loss and capital gain under one column by subtracting losses from gains column.
        :param df: data frame to perform transformation on.
        :return: pd.DataFrame object
        """
        df['capital-gain'] = df['capital-gain'] - df['capital-loss']
        df.drop(columns=['capital-loss'], inplace=True)
        df.rename(columns={'capital-gain': 'capital-change'}, inplace=True)
        self.continuous_features.append('capital-change')
        self.continuous_features.remove('capital-gain')
        self.continuous_features.remove('capital-loss')

        return df

    def normalize_cont_features(self, df):
        """
        normalize continuous features via MinMax normalization
        :param df: data frame to perform transformation on.
        :return: pd.DataFrame object
        """
        for feature in self.continuous_features:
            min_value = df[feature].min()
            max_value = df[feature].max()
            df[feature] = (df[feature] - min_value) / (max_value - min_value)

        return df

    def drop_cat_features(self, df):
        """
        drop categorical features
        :param df: data frame to perform transformation on.
        :return: pd.DataFrame object
        """
        df.drop(columns=self.categorical_features, inplace=True)

        return df

    def drop_cont_features(self, df):
        """
        drop continuous features
        :param df: data frame to perform transformation on.
        :return: pd.DataFrame object
        """
        df.drop(columns=self.continuous_features, inplace=True)

    def pivot_cat_features(self, df):
        """
        transform categorical features into dummies.
        :param df: data frame to perform transformation on.
        :return: pd.DataFrame object
        """
        return pd.get_dummies(df, columns=self.categorical_features)

    def reset_features_list(self):
        """
        reset continuous and categorical features' list to static list state
        """
        self.categorical_features = Data.categorical_features.copy()
        self.continuous_features = Data.continuous_features.copy()

    def cat_feat_conditional_probability(self, feature):
        """
        calculate probability for a sample to have salary >50K given it is of some category:
        - get dummies for the feature
        - for each indicator:
          - sum the number of matching pairs of the feature indicator and salary
          - subtract the number of mismatching pairs
          - divide by number of samples in the category
        print bar graph of the calculated ratios.
        :param feature: String object. the title of the feature to examine
        :return: show PNG of bar plot of conditional probability for a high salary given a sample is part of a category
        of feature
        """

        """bad comments"""

        # isolate feature and salary series
        df = self.custom_preprocess(flip_salary_index=True)
        feature_frame = df[feature]
        salary_series = df['salary']

        # pivot the feature series to get dummies
        feature_frame = pd.get_dummies(feature_frame, columns=[feature])

        # find probability for each dummy and store in a data frame
        corr_frame = pd.DataFrame(index=list(feature_frame.keys()), columns=['"correlation"'])
        for dummy in feature_frame.keys():
            # store multiplied columns in the dummy column.
            # if there is 1 in both columns the it will stay in the new Series
            pos_corr = (feature_frame[dummy] & salary_series).sum()
            total_of_dummy = feature_frame[dummy].sum()
            corr_frame['"correlation"'][dummy] = pos_corr / total_of_dummy
            print(dummy, total_of_dummy, pos_corr)

        corr_frame['positive'] = corr_frame['"correlation"'] > 0
        # plot results
        corr_frame['"correlation"'].plot(kind='barh')
        plt.show()

    def cat_conditional_probability(self):
        """
        plot bar graph of conditional probabilities for a high salary given that a subject is part of some category.
        :return: show PNG of bar plot of conditional probability for a high salary given a sample is part of a category
        """
        # get dummies and isolate salary series
        df = self.custom_preprocess(flip_salary_index=True, drop_cont=True, pivot_cat=True)
        salary_series = df['salary']
        df.drop(columns=['salary'], inplace=True)

        corr_frame = pd.DataFrame(index=list(df.keys()), columns=['P(>50K | is of category: _)'])

        # find probability for each dummy and store in a data frame
        for dummy in df.keys():
            # store multiplied columns in the dummy column.
            # if there is 1 in both columns the it will stay in the new Series
            pos_corr = (df[dummy] & salary_series).sum()
            total_of_dummy = df[dummy].sum()
            corr_frame['P(>50K | is of category: _)'][dummy] = pos_corr / total_of_dummy
            print(dummy, total_of_dummy, pos_corr)

        # plot results
        corr_frame['P(>50K | is of category: _)'].plot(kind='barh')
        plt.show()
