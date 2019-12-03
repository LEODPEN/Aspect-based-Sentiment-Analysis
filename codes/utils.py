# -*- coding: utf-8 -*-

import pickle
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import os
import csv
import glob
import pandas as pd


def pickle_load(file_path):
    return pickle.load(open(file_path, 'rb'))


def pickle_dump(obj, file_path):
    pickle.dump(obj, open(file_path, 'wb'))



def concat(predict, real, model_name, type):
    """
        return score for predictions made by sentiment analysis model
        :param predict : list []
        :param real : list []
        :param model_name: atae, tsa
        :param type: 1: dev/ 2: test
        :return:
        """
    result = {}
    result["predict"] = predict
    result["real"] = real
    df = pd.DataFrame.from_dict(result)

    name = ""
    if type is 1:
        name = '{}_predict_result_{}.csv'.format(model_name, 'dev')
    else:
    # type = 2
        name = '{}_predict_result_{}.csv'.format(model_name, 'test')
    path = os.getcwd()
    print("the csv file of results of comparing is at " + path + '/' + name)
    # 保存列index, 便于寻找不同行
    df.to_csv(name, index=1)


def get_score_senti(y_true, y_pred):
    """
    return score for predictions made by sentiment analysis model
    :param y_true: array shaped [batch_size, 3]
    :param y_pred: array shaped [batch_size, 3]
    :return:
    """
    y_true = np.argmax(y_true, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro') # 宏f1 每个类的f1求均值

    print('acc:', acc)
    print('macro_f1:', f1)
    return acc, f1
