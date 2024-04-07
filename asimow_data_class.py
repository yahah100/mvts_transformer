import pandas as pd
from src.datasets.data import BaseData
import os
import re
import glob
import numpy as np


class ArcWeldingData(BaseData):
    """
    """

    def __init__(self, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None, config=None):

        self.set_num_processes(n_proc=n_proc)

        self.all_df, self.labels_df = self.load_all(root_dir, file_list=file_list, pattern=pattern)
        self.all_df = self.all_df.sort_values(by=['ID'])  # datasets is presorted
        self.all_df = self.all_df.set_index('ID')
        self.all_IDs = self.all_df.index.unique()  # all sample (session) IDs
        self.max_seq_len = 200
        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        self.feature_names = ['current', 'voltage']
        self.feature_df = self.all_df[self.feature_names]


    def load_all(self, root_dir, file_list=None, pattern=None):
        """
        Loads datasets from csv files contained in `root_dir` into a dataframe, optionally choosing from `pattern`
        Args:
            root_dir: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_dir` to consider.
                Otherwise, entire `root_dir` contents will be used.
            pattern: optionally, apply regex string to select subset of files
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
        """
        # each file name corresponds to another date. Also tools (A, B) and others.

        if file_list is None:
            data_paths = glob.glob(os.path.join(root_dir, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_dir, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_dir, '*')))
        print(f"Data paths: {data_paths}")
        if pattern is None:
            # by default evaluate on
            selected_paths = data_paths
        else:
            selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))
        input_paths = [p for p in selected_paths if os.path.isfile(p) and p.endswith('.csv')]
        # Select paths for training and evaluation
        all_df = self.read_data(f'{root_dir}/asimow_dataset.csv')
        if len(input_paths) == 0:
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df


    def load_single(self, filepath):
        df = pd.read_csv(filepath)

        labels = pd.Series(df['label'], dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes, dtype=np.int8) 
        return df, labels_df

    @staticmethod
    def read_data(filepath):
        """Reads a single .csv, which typically contains a day of datasets of various machine sessions.
        """
        df = pd.read_csv(filepath)
        return df

    @staticmethod
    def select_columns(df):
        return df


def test():
    root_dir = 'data'
    machineData = ArcWeldingData(root_dir, pattern="val")
    print(machineData.all_df.head())
    # print(machineData.feature_df.head())
    # print(machineData.feature_names)
    # print(machineData.all_IDs)
    # print(machineData.max_seq_len)

if __name__ == '__main__':
    test()