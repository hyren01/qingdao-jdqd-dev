# -*- coding: utf-8 -*-
import os
import numpy as np
from keras.layers import Input, LSTM, Dense
from keras.models import Model
import sys

module_dir = os.path.dirname(__file__)
import_dir = os.path.join(module_dir, '../..')
log_path = os.path.join(import_dir, 'log/logs')
sys.path.append(import_dir)

import utils.logger as logger
from algorithm.event_forecast import preprocess as pp

LOG = logger.Logger("debug", log_path=log_path)


class Session:
    def __init__(self, n_in, n_out, latent_dim, batch_size, epochs, pca_n):
        """
        初始化模型训练session
        :param n_in: 输入序列长度
        :param n_out: 输出序列长度
        :param latent_dim: RNN hidden units 个数
        :param batch_size:
        :param epochs:
        :param pca_n: 降维维度
        """
        self.n_in = n_in
        self.n_out = n_out
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.pca_n = pca_n

    def split_sequences(self, values_pca, events_p_oh, input_len, output_len, dates, start_date, end_date):
        """
        数据集划分
        :param end_date:
        :param start_date:
        :param dates:
        :param output_len:
        :param input_len:
        :param events_p_oh:
        :param values_pca: 降维操作后的输入数据
        :return: 整个seq2seq模型, encoder模型, decoder模型
        """
        # inputs_train, outputs_train = pp.gen_train_samples(values_pca, events_p_oh, self.n_in, self.n_out, cv_row)
        inputs_train, outputs_train = pp.gen_samples_by_date(values_pca, events_p_oh, input_len, output_len,
                                                             dates, start_date, end_date)
        outputs_train_inf = np.insert(outputs_train, 0, 0, axis=-2)[:, :-1, :]
        return inputs_train, outputs_train, outputs_train_inf

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

    def accuracy(self, array_x, array_y, array_yin, model_sub_dir):
        """
        训练模型并对模型进行评估
        :param array_yin:
        :param array_y:
        :param array_x:
        :param model_sub_dir: 存放此次训练所生成的模型的目录
        :return: 模型的召回率, 误报率
        """
        LOG.info(f"Current value: 滞后期={self.n_in}, pca={self.pca_n}")

        n_output = array_y.shape[2]
        n_input = array_x.shape[2]
        model, encoder, decoder = self.define_models(n_input, n_output)
        model.fit([array_x, array_yin], array_y, batch_size=self.batch_size, epochs=self.epochs, verbose=2)
        encoder.save(f'{model_sub_dir}/encoder.h5')
        decoder.save(f'{model_sub_dir}/decoder.h5')


def execute(model_name, data, events_p_oh, dates, start_date, end_date, output_len, min_pca_dim, max_pca_dim,
            min_input_len, max_input_len, step, num_units=128, batch=64, epoch=150):
    """
    执行页面的训练模型请求, 遍历不同超参数的组合来训练模型
    :param step:
    :param end_date:
    :param start_date:
    :param dates:
    :param data:
    :param events_p_oh:
    :param model_name:
    :param output_len: 输出序列长度(预测天数)
    :param min_pca_dim: 遍历pca维度起始值
    :param max_pca_dim: 遍历pca维度结束值
    :param min_input_len: 遍历输入序列长度最小值
    :param max_input_len: 遍历输入序列长度最大值
    :param num_units: RNN单元隐藏节点个数
    :param batch:
    :param epoch:
    :return: None
    """
    module_path = os.path.dirname(__file__)
    models_dir = f'{module_path}/resources/models'
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    pca_dir = models_dir
    sub_model_dirs = []
    sub_model_names = []
    outputs_list = []
    params_list = []

    for i in range(min_pca_dim, max_pca_dim, step):  # 各种降维选择
        values_pca = pp.apply_pca(i, pca_dir, data)
        for j in range(min_input_len, max_input_len, 5):  # 滞后期的选择
            sess = Session(j, output_len, num_units, batch, epoch, i)
            sub_model_name = f'{model_name}-{sess.n_in}-{sess.n_out}-{sess.pca_n}'
            sub_model_names.append(sub_model_name)
            sub_model_dir = models_dir + '/' + sub_model_name
            if not os.path.exists(sub_model_dir):
                os.mkdir(sub_model_dir)
            sub_model_dirs.append(sub_model_dir)
            array_x, array_y, array_yin = sess.split_sequences(values_pca, events_p_oh, j, output_len, dates,
                                                               start_date, end_date)
            outputs_list.append(array_y)
            params_list.append([j, output_len, i])
            sess.accuracy(array_x, array_y, array_yin, sub_model_dir)
    return sub_model_dirs, sub_model_names, outputs_list, params_list


if __name__ == '__main__':
    pass
