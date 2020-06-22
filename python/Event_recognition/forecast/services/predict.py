# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 10:48:56 2019

@author: 12894
"""
# --------------------------------原始数据进行未来n_out天的预测----------------------------------
import sys
import os
import pandas as pd
import numpy as np
from numpy import argmax
from sklearn.externals import joblib
from keras.models import load_model


class Session_result:

    def __init__(self, model_path, data_file, n_in, n_out, pca_n):
        """
        n_in:  被使用的滞后期数据行数
        n_out: 要预测的未来天数
        pca_n: 降维到多少
        """
        # print(f"model_path={model_path}, datafile={datafile}, n_in={n_in}, n_out={n_out}, pca_n={pca_n}")
        self.can_predict = True
        self.model_path = model_path
        self.datafile = data_file
        self.n_in = n_in
        self.n_out = n_out
        self.pca_n = pca_n

        encode_model = f"{self.model_path}/{self.n_in}-{self.n_out}-{self.pca_n}encoder.h5"
        if not os.path.exists(encode_model):
            print("不存在该模型 {}，退出该次预测".format(encode_model))
            self.can_predict = False
            return

        with open(self.model_path+'/'+str(self.n_in)+'-'+str(self.n_out)+'-'+str(self.pca_n)+'dig_lable.txt', 'r') as f:
            self.dig_lable = eval(f.read())
        self.num_classes = len(self.dig_lable)
        self.pca = joblib.load(f"{self.model_path}/{self.n_in}-{self.n_out}-{self.pca_n}fit_pca")
        self.encoder_model = load_model(f"{self.model_path}/{self.n_in}-{self.n_out}-{self.pca_n}encoder.h5", compile=False)
        self.decoder_model = load_model(f"{self.model_path}/{self.n_in}-{self.n_out}-{self.pca_n}decoder.h5", compile=False)
        
    def predict_sequence(self, infenc, infdec, source):
        # 输入序列编码得到编码状态向量
        state = infenc.predict(source)
        # 初始目标序列输入：通过开始字符计算目标序列第一个字符，这里是0
        target_seq = np.array([0.0 for _ in range(self.num_classes)]).reshape(1, 1, self.num_classes)
        # 输出序列列表
        output = list()
        for t in range(self.n_out):
            # predict next char
            yhat, h, c = infdec.predict([target_seq] + state)
            # 截取输出序列，取后三个
            output.append(yhat[0, 0, :])
            # 更新状态
            state = [h, c]
            # 更新目标序列(用于下一个词预测的输入)
            target_seq = yhat
        return np.array(output) 

    def load_data(self):
        dataset = pd.read_csv(self.datafile, header=None, index_col=0)
        values = dataset.values
        values_pca = self.pca.transform(values)
        values_pca = values_pca[-self.n_in:, :]
        input_seq = np.array([values_pca])
        target = self.predict_sequence(self.encoder_model, self.decoder_model, input_seq)
        result = one_hot_decode(target)
        pred = []
        value = 0
        for i in result:
            pred.append(self.dig_lable[i])
            value = value + i
        if value != 0:
            return pred


def one_hot_decode(encoded_seq):
    # one_hot解码
    return [argmax(vector) for vector in encoded_seq]


def execute(model_path, data_file, muti_days, min_dim=5, max_dim=10, min_lag=10, max_lag=61):
    """
    datafile: 从生产数据中转成2000维的csv文件，第一列必须是时间列
    """
    # 为了演示效果，可以把原始数据从292行进行人为切分

    # 调用模型，得到预测结果。1行days列的数组，每列代表1天发生的事件
    all_preds = []
    for i in [min_dim, max_dim]:           # 各种降维选择
        for j in range(min_lag, max_lag, 5):           # 滞后期的选择
            # 获得了一次预测结果：[1, days]
            session_result = Session_result(model_path, data_file, j, muti_days, i)
            if not session_result.can_predict:
                continue
            pred = session_result.load_data()
            # 清理到 全0的 pred 结果
            print(f"pca={i:2d} window={j:3d} : pred={pred}")
            all_preds.append(pred) if pred is not None else None
    return all_preds


def transform_result(preds):

    new_result = {}
    if len(preds) < 1:
        return new_result

    frist_pred = preds[0]
    for index, element in enumerate(frist_pred):
        new_result[str(index + 1)] = []

    for pred_element in preds:
        for index, events_id in enumerate(pred_element):
            if events_id is None or events_id == 0:
                continue
            new_result_key = str(index + 1)
            new_result_value = new_result.get(new_result_key)
            if new_result_value.__contains__(events_id):
                continue
            new_result_value.append(events_id)

    return new_result


if __name__ == '__main__':
    print("\n============================================")
    if len(sys.argv)-1 != 3:
        exit("missing cmd args : model_path datafile days")
    model_path = sys.argv[1]
    datafile = sys.argv[2]
    days = int(sys.argv[3])
    print("Start ... ...\n")
    print(f"using model_path={model_path}, datafile={datafile}, days={days} for predict.")
    all_preds = execute(model_path, datafile, days)
    print("\n============================================")
    preds_result = transform_result(all_preds)
    print(preds_result)
