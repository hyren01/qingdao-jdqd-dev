# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import model_evalution as mu
import preprocess as pp
from datetime import timedelta
import obtain_data as od


class Session_result:
    def __init__(self, model_dir, load_model=True):
        """

        :param model_dir: 存放模型的目录名
        :param load_model: 是否加载已有模型
        """
        if load_model:
            self.cnn_model = mu.load_cnn_model(model_dir)

    def predict_sample(self, input_sample, n_classes, output_len):
        in_encoder = np.array([input_sample])
        in_decoder = np.zeros((len(in_encoder), output_len, n_classes), dtype='float32')
        in_decoder[:, 0, 0] = 1
        predict = np.zeros((len(in_encoder), output_len), dtype='float32')
        for i in range(output_len - 1):
            predict = self.cnn_model.predict([in_encoder, in_decoder])
            predict_ = predict.argmax(axis=-1)
            predict_ = predict_[:, i].tolist()
            for j, x in enumerate(predict_):
                in_decoder[j, i + 1, x] = 1
        output_seq = predict[0].tolist()
        return output_seq

    def pred_test_reload(self, inputs_test, n_classes, output_len):
        preds = [self.predict_sample(input_sample, n_classes, output_len) for input_sample in inputs_test]
        return preds


def execute2(model_dir, pca_dir, n_classes, cv_row):
    """
    执行页面的预测请求
    :param cv_row:
    :param values_pca:
    :param n_classes:
    :param model_dir: 模型存放的目录
    :return: 预测结果拼接成的字符串
    """
    sub_model_dir = model_dir.split('/')[-1]
    input_len, output_len, n_pca = sub_model_dir.split('-')[:3]
    input_len, output_len, n_pca = int(input_len), int(output_len), int(n_pca)
    dates, data = od.combine_data(['data_xlshuju_1', 'data_xlshuju_2', 'data_xlshuju_3'], True)
    values_pca = pp.apply_pca(n_pca, pca_dir, data, True)
    inputs_test = pp.gen_test_inputs(values_pca, input_len, cv_row)
    session_result = Session_result(model_dir)
    preds = session_result.pred_test_reload(inputs_test, n_classes, output_len)
    return preds