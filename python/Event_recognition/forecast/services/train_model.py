# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 17:51:43 2019

@author: 12894
"""
import os
import sys
import pandas as pd
import numpy as np
from numpy import argmax
from numpy import array_equal
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.utils import to_categorical
from sklearn.decomposition import PCA
from sklearn.externals import joblib


# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'


class Session:
    def __init__(self, data_file, n_in, n_out, latent_dim, batch_size, epochs, pca_n):
        self.datafile = data_file
        # 滞后期
        self.n_in = n_in
        # 预测未来天数
        self.n_out = n_out
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.pca_n = pca_n
        # dot = data_file.find('.')
        self.model_path = os.path.dirname(data_file)
        # self.model_path = "mt" + self.model_path
        # os.mkdir(self.model_path) if not os.path.exists(self.model_path) else None
        self.num_classes = None
        # self.one_hot_decode = None

    def deposition(self, values):
        pca = PCA(n_components=self.pca_n)
        values_pca = pca.fit_transform(values)
        joblib.dump(pca,
                    self.model_path + '/' + str(self.n_in) + '-' + str(self.n_out) + '-' + str(self.pca_n) + 'fit_pca')
        return values_pca

    def load_data(self):
        dataset = pd.read_csv(self.datafile, header=None, index_col=0)
        # dataset=dataset.iloc[:292,:]
        target_categorys = sorted(list(set(dataset.iloc[:, -1])))
        self.num_classes = len(target_categorys)
        dig_lable = dict(enumerate(target_categorys))
        lable_dig = dict((lable, dig) for dig, lable in dig_lable.items())
        # 把事件映射成标签
        with open(self.model_path + '/' + str(self.n_in) + '-' + str(self.n_out) + '-' + str(
                self.pca_n) + 'dig_lable.txt', 'w') as f:
            f.write(str(dig_lable))
        length = len(dataset.columns)
        dataset['szbq'] = dataset[length].apply(lambda x: lable_dig.get(x))
        values = dataset.values
        values_pca = self.deposition(values[:, :-2])
        return values, values_pca

    def split_sequences(self):
        values, values_pca = self.load_data()
        x, y, yin = list(), list(), list()
        for i in range(len(values)):
            end_ix = i + self.n_in
            out_end_ix = end_ix + self.n_out
            if out_end_ix > len(values):
                break
            seq_x, seq_y, seq_yin = values_pca[i:end_ix, :], values[end_ix:out_end_ix, -1], \
                                    np.insert(values[end_ix:out_end_ix - 1, -1], 0, 0, axis=0)
            x.append(seq_x)
            y.append(seq_y)
            yin.append(seq_yin)
        return np.array(x), np.array(y), np.array(yin)

    def define_models(self, n_input, n_output):
        # 训练模型中的encoder
        encoder_inputs = Input(shape=(None, n_input))
        encoder = LSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]  # 仅保留编码状态向量
        # 训练模型中的decoder
        decoder_inputs = Input(shape=(None, n_output))
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(n_output, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        # 新序列预测时需要的encoder
        encoder_model = Model(encoder_inputs, encoder_states)
        # 新序列预测时需要的decoder
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        return model, encoder_model, decoder_model

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

    def accuracy(self):
        print("======================================================")
        print(f"Current value: 滞后期={self.n_in}, pca={self.pca_n}")
        print("======================================================")
        array_x, array_y, array_yin = self.split_sequences()
        if len(array_x) == 0 or len(array_y) == 0:
            print("len(array_x)或len(array_y)为0不再计算")
            return
        src_encoded = to_categorical(array_yin, num_classes=self.num_classes)
        src_decoded = to_categorical(array_y, num_classes=self.num_classes)
        n_input = array_x.shape[2]
        n_output = src_decoded.shape[2]
        model, encoder_model, decoder_model = self.define_models(n_input, n_output)
        model.fit([array_x, src_encoded], src_decoded, batch_size=self.batch_size, epochs=self.epochs)
        # Save model
        encoder_model.save(
            self.model_path + '/' + str(self.n_in) + '-' + str(self.n_out) + '-' + str(self.pca_n) + 'encoder.h5')
        decoder_model.save(
            self.model_path + '/' + str(self.n_in) + '-' + str(self.n_out) + '-' + str(self.pca_n) + 'decoder.h5')
        total = len(src_decoded)
        correct = 0
        rmse = 0.0
        for seq_index in range(total):
            input_seq = array_x[seq_index: seq_index + 1]
            target = self.predict_sequence(encoder_model, decoder_model, input_seq)
            if array_equal(one_hot_decode(src_decoded[seq_index: seq_index + 1][0]), one_hot_decode(target)):
                correct += 1
            rmse = float(correct) / float(total) * 100.0
        #        for seq_index in range(total):
        #            input_seq = array_X[seq_index: seq_index + 1]
        #            target = self.predict_sequence(encoder_model, decoder_model, input_seq)
        #            with open("./true_pred.txt",'a') as f:
        #                f.write(str(self.one_hot_decode(src_decoded[seq_index: seq_index + 1][0]))+
        #                        str(self.one_hot_decode(target))+'\n')
        return round(rmse)


def one_hot_decode(encoded_seq):
    # one_hot解码
    return [argmax(vector) for vector in encoded_seq]


def execute(data_file, muti_day, min_dim=5, max_dim=10, min_lag=10, max_lag=61, unit=128, batch=64, epoch=150):
    """
    datafile:   包含2000维数据，并且，把事件表数据按照日期横向合并到最后一列，且数据按照日期排序
    unit:       编码解码里面的神经元个数
    batch:      每次喂给网络的小批量数据个数
    epoch:      训练次数

    """
    # print("loop: 5, 10 | 10, 61, 5")
    model_path = None
    for i in [min_dim, max_dim]:  # 各种降维选择
        for j in range(min_lag, max_lag, 5):  # 滞后期的选择
            sess = Session(data_file, j, muti_day, unit, batch, epoch, i)
            rmse = sess.accuracy()
            model_path = sess.model_path

    return model_path


if __name__ == '__main__':
    print("\n============================================")
    if len(sys.argv) - 1 != 2:
        exit("missing cmd args : datafile days")
    print("Start training ... ...\n")
    datafile = sys.argv[1]
    days = int(sys.argv[2])
    print(f"using datafile={datafile}, days={days} for training model!")
    execute(datafile, days)
