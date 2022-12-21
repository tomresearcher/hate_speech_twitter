import argparse
from common.load_data import load_data
from model.beto.beto_model import train_predict_model
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from src.common.score import scorePredict


def main(parser):
    args = parser.parse_args()
    training_set = args.training_set
    use_cuda = args.use_cuda
    is_cross_validation = args.is_cross_validation
    is_evaluate = args.is_evaluate
    exec_model(training_set, is_cross_validation, is_evaluate, use_cuda)
    

def exec_model(training_set, is_cross_validation, is_evaluate, use_cuda):

    df = load_data(os.getcwd() + training_set, True)

    count_feature = len(df['features'].iloc[0])

    results = None
    f1s = None
    if is_cross_validation:
        n = 5
        kf = KFold(n_splits=n, random_state=3, shuffle=True)
        results = []
        f1s = []
        for train_index, val_index in kf.split(df):
            train_df = df.iloc[train_index]
            val_df = df.iloc[val_index]
            acc, f1, model_outputs_test = train_predict_model(train_df, val_df, False, use_cuda, count_feature, 'multi')
            results.append(acc)
            f1s.append(f1)
    else:
        df_train, df_test = train_test_split(df, test_size=0.20,  random_state=1)
        df_test.to_json('test', lines=True)
        _, f1, model_outputs_test = train_predict_model(df_train, df_test, True, use_cuda, count_feature, 'multi', is_evaluate, True)

    if results:
        print(np.mean(results))
        print(np.mean(f1s))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--training_set",
                        default="/data/Dataset_VIL.json",
                        type=str,
                        help="This parameter is the relative dir of training set.")

    parser.add_argument("--is_cross_validation",
                        default=False,
                        action='store_true',
                        help="This parameter should be True if cross-validation is a requirement.")

    parser.add_argument("--use_cuda",
                        default=True,
                        action='store_true',
                        help="This parameter should be True if cuda is present.")

    parser.add_argument("--is_evaluate",
                        default=True,
                        action='store_true',
                        help="This parameter should be True if you want to split train in train and dev.")

    main(parser)