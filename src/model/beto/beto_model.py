import os
from src.common.score import scorePredict
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from src.model.out_simple_transformer.ClassificationModel import ClassificationModel


def train_predict_model(df_train, df_test, is_predict, use_cuda, value_head, type_class, is_evaluate, is_features):
    labels_test = pd.Series(df_test['labels']).to_numpy()
    labels = list(df_train['labels'].unique())
    labels.sort()

    if type_class == 'bi':
        num_labels = 1
    else:
        num_labels = len(labels)

    manual_seed = 4260

    model = ClassificationModel('beto', 'dccuchile/bert-base-spanish-wwm-cased',
                                 num_labels=num_labels, use_cuda=use_cuda, args={
                                'learning_rate':2e-5,
                                'num_train_epochs': 3,
                                'reprocess_input_data': True,
                                'overwrite_output_dir': True,
                                'evaluate_during_training': is_evaluate,
                                'process_count': 10,
                                'train_batch_size': 2,
                                'eval_batch_size': 2,
                                'max_seq_length': 400,
                                'multiprocessing_chunksize': 500,
                                'fp16': True,
                                'fp16_opt_level': 'O1',
                                'value_head': value_head,
                                'regression': False,
                                'tensorboard_dir': 'tensorboard',
                                'manual_seed': manual_seed,
                                'wandb_project': 'Tuits'})

    df_eval = None
    if is_evaluate:
        df_train, df_eval = train_test_split(df_train, test_size=0.2, train_size=0.8, random_state=1)
        print(df_train['labels'].value_counts())
        print(df_eval['labels'].value_counts())
    model.train_model(df_train, eval_df=df_eval, f1=sklearn.metrics.f1_score)

    results = ''
    if is_predict:
        if is_features:
            text_a = df_test['text_a']
            text_b = df_test['text_b']
            features = df_test['features']
            df_result = pd.concat([text_a, text_b, features], axis=1)
        else:
            text = df_test['text']
            features = df_test['features']
            df_result = pd.concat([text, features], axis=1)
        value_in = df_result.values.tolist()
        _, model_outputs_test = model.predict(value_in)
    else:
        result, model_outputs_test, wrong_predictions = model.eval_model(df_test, acc=accuracy_score)
        results = result['acc']
    if type_class == 'bi':
        y_predict = np.round(model_outputs_test)
    else:
        y_predict = np.argmax(model_outputs_test, axis=1)

    values, f1 = scorePredict(y_predict, labels_test, labels)
    print(values)
    return results, f1, model_outputs_test


def predict(df_test, use_cuda, model_dir):
    model = ClassificationModel(model_type='bert', model_name=os.getcwd() + model_dir, use_cuda=use_cuda)
    labels_test = pd.Series(df_test['labels']).to_numpy()
    labels = list(df_test['labels'].unique())
    labels.sort()
    text = df_test['texts']
    features = df_test['features']
    df_result = pd.concat([text, features], axis=1)
    value_in = df_result.values.tolist()
    _, model_outputs_test = model.predict(value_in)
    y_predict = np.argmax(model_outputs_test, axis=1)
    print(scorePredict(y_predict, labels_test, labels))