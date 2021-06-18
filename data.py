import pandas as pd
import sklearn as sk

class Data:
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

