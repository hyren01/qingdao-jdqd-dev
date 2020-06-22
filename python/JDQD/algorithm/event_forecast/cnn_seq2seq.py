# -*- coding: utf-8 -*-
import numpy as np
import os
from datetime import date
from keras.models import Model
from keras.layers import Input, Convolution1D, Dot, Dense, Activation, Concatenate
import obtain_data as od
import preprocess as pp


class Session:
    def __init__(self, n_in, n_out, batch_size, epochs, pca_n):
        self.n_in = n_in
        self.n_out = n_out  
        self.batch_size = batch_size
        self.epochs = epochs
        self.pca_n = pca_n


    def split_sequences(self, values_pca, events_p_oh, split_event_row):
        """
        数据集划分
        :param split_event_row:
        :param events_p_oh:
        :param values_pca: 降维操作后的输入数据
        :param dates: 输入数据日期列表
        :return: 整个seq2seq模型, encoder模型, decoder模型
        """
        inputs_train, outputs_train = pp.gen_train_samples(values_pca, events_p_oh, self.n_in, self.n_out, split_event_row)
        outputs_train_inf = np.insert(outputs_train, 0, 0, axis=-2)[:, :-1, :]
        return inputs_train, outputs_train, outputs_train_inf

    def define_model(self, n_input, n_output):
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, n_input))
        # Encoder
        x_encoder = Convolution1D(256, kernel_size=3, activation='relu',
                                  padding='causal')(encoder_inputs)
        x_encoder = Convolution1D(256, kernel_size=3, activation='relu',
                                  padding='causal', dilation_rate=2)(x_encoder)
        x_encoder = Convolution1D(256, kernel_size=3, activation='relu',
                                  padding='causal', dilation_rate=4)(x_encoder)
        
        decoder_inputs = Input(shape=(None, n_output))
        # Decoder
        x_decoder = Convolution1D(256, kernel_size=3, activation='relu',
                                  padding='causal')(decoder_inputs)
        x_decoder = Convolution1D(256, kernel_size=3, activation='relu',
                                  padding='causal', dilation_rate=2)(x_decoder)
        x_decoder = Convolution1D(256, kernel_size=3, activation='relu',
                                  padding='causal', dilation_rate=4)(x_decoder)
        # Attention
        attention = Dot(axes=[2, 2])([x_decoder, x_encoder])
        attention = Activation('softmax')(attention)
        
        context = Dot(axes=[2, 1])([attention, x_encoder])
        decoder_combined_context = Concatenate(axis=-1)([context, x_decoder])
        
        decoder_outputs = Convolution1D(64, kernel_size=3, activation='relu',
                                        padding='causal')(decoder_combined_context)
        decoder_outputs = Convolution1D(64, kernel_size=3, activation='relu',
                                        padding='causal')(decoder_outputs)
        # Output
        decoder_dense = Dense(n_output, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        
        return model 
    
    def Accuracy(self, values_pca, events_p_oh, model_sub_dir, split_event_row):
        array_X, array_y, array_yin = self.split_sequences(values_pca, events_p_oh, split_event_row)
        n_input = array_X.shape[2]
        n_output = array_y.shape[2]
        model = self.define_model(n_input, n_output)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        model.fit([array_X, array_yin], array_y,batch_size=self.batch_size,epochs=self.epochs)
        model.save(model_sub_dir + '/' + 'cnn_model.h5')


def execute(days, event_priority, pca_dim_min=4, pca_dim_max=65, input_len_min=5,
            input_len_max=61, batch=64, epoch=150):
    """
    执行页面的训练模型请求, 遍历不同超参数的组合来训练模型
    :param days: 输出序列长度(预测天数)
    :param pca_dim_min: 遍历pca维度起始值
    :param pca_dim_max: 遍历pca维度结束值
    :param input_len_min: 遍历输入序列长度最小值
    :param input_len_max: 遍历输入序列长度最大值
    :param unit: RNN单元隐藏节点个数
    :param batch:
    :param epoch:
    :return: None
    """
    models_dir = f'resources/models'
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    pca_dir = models_dir
    models_sub_dir = []
    dates, data = od.combine_data(['data_xlshuju_1', 'data_xlshuju_2', 'data_xlshuju_3'], True)
    events_p = od.get_events(dates, event_priority)
    events_set = pp.get_events_set(events_p)
    events_p_oh = pp.events_one_hot(events_p, events_set)
    recur_events_rows = pp.get_recur_events_rows(events_p_oh)
    split_event_row = recur_events_rows[5][-3]
    for i in range(pca_dim_min, pca_dim_max, 4):  # 各种降维选择
        values_pca = pp.apply_pca(i, pca_dir, data)
        for j in range(input_len_min, input_len_max, 5):  # 滞后期的选择
            print('运行到第pca_n:{} n_in:{}的模型'.format(i, j))
            sess = Session(j, days, batch, epoch, i)
            model_sub_dir_ = f'{sess.n_in}-{sess.n_out}-{sess.pca_n}-{date.today()}'
            model_sub_dir = models_dir + '/' + model_sub_dir_
            if not os.path.exists(model_sub_dir):
                os.mkdir(model_sub_dir)
            models_sub_dir.append(model_sub_dir)
            sess.Accuracy(values_pca, events_p_oh, model_sub_dir, split_event_row)



if __name__ == '__main__':
    days = 3
    epoch = 100
    models_dir = 'resources/models'
    pca_dir = 'resources/models'

    execute(days, batch=64, epoch=epoch)
