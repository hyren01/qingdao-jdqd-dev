# coding: utf-8

import os
import sys

file = os.path.abspath(__file__)
dir_path = os.path.dirname(file)
dir_path = os.path.join(dir_path, "../")
sys.path.append(dir_path)

import re
import numpy as np
from resources.bert4keras_v01.utils import  data_generator
from resources.bert4keras_v01.backend import keras
from resources.bert4keras_v01.tokenizer import Tokenizer
from resources.bert4keras_v01.bert import build_bert_model
from keras.layers import Dense, Dropout

from predict import logger
from predict import config
from predict import data_util
from predict import get_triples
from predict import get_abstract

# 设置日志类
LOG = logger.Logger('info')
# 加载参数类
Config = config.Config()

# bert模型参数
config_path = os.path.join(dir_path, Config.config_path)
# bert字典
vocab_path = os.path.join(dir_path, Config.vocab_path)
# 事件匹配模型路径
match_model_path = os.path.join(dir_path, Config.match_model_path)
# 事件列表路径
allevent_path = os.path.join(dir_path, Config.allevent_path)
# 字符串编码最大长度
maxlen = Config.max_length
# 预测批量大小
batch_size = Config.batch_size

# 加载bert字典，构建token类
tokenizer = Tokenizer(vocab_path)


def load_match_model():
    '''
    搭建匹配模型框架，构建模型
    :return:匹配模型对象
    '''
    # 构建bert模型主体
    bert = build_bert_model(
        config_path=config_path,
        with_pool=True,
        return_keras_model=False,
    )
    # 搭建分类模型框架
    output = Dropout(rate=0.1)(bert.model.output)
    output = Dense(units=2,
                   activation='softmax',
                   kernel_initializer=bert.initializer)(output)

    model = keras.models.Model(bert.model.input, output)
    # 加载模型参数
    model.load_weights(match_model_path)

    return model


def valid_file(file_path):
    '''
    传入文件路径，验证文件是否存在
    :param file_path: 文件路径
    :return: True, False
    '''

    if os.path.isfile(file_path):
        return True
    elif os.path.isdir(file_path):
        LOG.info('The path {} is a directory.'.format(file_path))
        return False
    elif not os.path.exists(file_path):
        LOG.info('The file {} is missed.'.format(file_path))
        return False


def valid_dir(dir_path):
    '''
    传入文件夹路径，验证文件夹是否存在
    :param dir_path: 文件夹路径
    :return: True, False
    '''
    if os.path.isfile(dir_path):
        LOG.info('The path {} is a file.'.format(dir_path))
        return False
    elif os.path.isdir(dir_path):
        return True
    elif not os.path.exists(dir_path):
        LOG.info('The directory {} is missed.'.format(dir_path))


def file_reader(file_path):
    '''
    传入文件路径，读取文件内容
    :param file_path: 文件路径
    :return: 文件内容--str
    '''
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except Exception as e:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content


def generate_samples(event_list, sentences):
    '''
    传入事件列表和语句列表，生成匹配对进行预测
    :param event_list: 事件列表--list
    :param sentences: 语句列表--list
    :return: 样本[(event, sentence, 0), ]格式为元组列表
    '''
    samples = []
    for once in sentences:
        if once:
            for event in event_list:
                if event:
                    samples.append((event[-1], once, str(0)))
    return samples


def get_events():
    '''
    传入事件文件路径，返回事件列表
    :param event_path: 保存事件的文件路径
    :return: 事件列表--list
    '''
    if valid_file(allevent_path):
        content = file_reader(allevent_path)
        events = content.split('\n')
        event_list = []
        for once in events:
            if once:
                event_ID = once.split('`')[0]
                event = once.split('`')[1].replace(' ', '')
                event_list.append((event_ID, event))
        return event_list
    else:
        exit(1)


def cut_sentence(content):
    '''
    传入文本字符串，进行分句
    :param content: 文本字符串
    :return: 句子列表--list
    '''
    return re.split('[。?？!！]', content)


def get_parts(content):
    '''
    传入文本字符串，获取文本的前后各128个字符
    :param content:出入的文章字符串
    :return:过去的文章各个部分--list
    '''
    parts = []

    if len(content) > 128:
        parts.append(content[0:128])
        parts.append(content[-128:])
    else:
        parts.append(content)

    return parts


def sort_socres(event_list, pred):
    '''
    对预测得到的结果按照置信度进行排序
    :param event_list: 事件列表--list
    :param pred: 预测结果
    :return:按照相似度排序后的列表 [{event_id : str, ratio: float}]
    '''

    predicted_event = {}
    event_scores = {}
    for key in event_list:
        predicted_event[key[1]] = [key[0], []]
    for once in pred:
        predicted_event[once[0]][1].append(once[-1])
    for i in predicted_event:
        event_scores[predicted_event[i][0]] = max(predicted_event[i][1])
    event_sorted = list(sorted(event_scores.items(), key=lambda e: e[1], reverse=True))
    event_sorted = [{'event_id': elem[0], 'ratio': elem[1]} for elem in event_sorted]

    return event_sorted


def get_predict_result(model, event_list, title, content, sample_type):
    '''
    传入模型对象以及事件列表和待匹配的文本，返回匹配后的事件id及对应的相似度
    :param model: 匹配模型对象
    :param event_list: 事件列表
    :param title: 文章标题
    :param content: 文章内容
    :param sample_type: 匹配类型
    :return: {事件ID:score}
    '''
    def evaluate(data):
        '''
        传入经过ids化的数据，进行预测
        :param data: ids化后的数据
        :return: results--相似度列表
        '''
        results = []
        for x_true, y_true in data:
            y_pred = model.predict(x_true)
            results.extend(np.reshape(y_pred[:, 1], (-1,)).tolist())

        return results

    title = data_util.process_data(title)
    content = data_util.process_data(content)
    title_samples = generate_samples(event_list, [title])

    # 判断文章样本类型，根据类型生成样本
    if sample_type == 'triples':
        # 调用三元组模块，抽取文章中所有的三元组
        triples_content = get_triples.get_triples(content)
        triples_sentences = []
        for once in triples_content:
            if once:
                triples_sentences.append(''.join(once))
        content_samples = generate_samples(event_list, triples_sentences)

    elif sample_type == 'abstract':
        # 抽取文章摘要句子
        summary_sentences = get_abstract.get_abstract(content)
        content_samples = generate_samples(event_list, summary_sentences)

    elif sample_type == 'parts':
        # 获取文章的部分片段作为样本
        parts_sentences = get_parts(content)
        content_samples = generate_samples(event_list, parts_sentences)

    else:
        LOG.info('Sample_type is wrong, please choose triples, abstract or parts. The defalult is parts.')
        exit(1)

    # 标题样本生成对象
    title_generator = data_generator(title_samples, tokenizer, max_length=maxlen, batch_size=batch_size)
    # 获取文章标题匹配结果
    title_results = evaluate(title_generator)
    # 文章内容样本生成对象
    content_generator = data_generator(content_samples, tokenizer, max_length=maxlen, batch_size=batch_size)
    # 获取文章内容匹配结果
    content_results = evaluate(content_generator)

    # 整理匹配结果
    # [[event1, title, score], ]
    title_predicted = []
    for elem, pred in zip(title_generator.data, title_results):
        title_predicted.append(list(elem[0:2]) + [pred])

    # [[event1, sentence, score], ]
    content_predicted = []
    for elem, pred in zip(content_generator.data, content_results):
        content_predicted.append(list(elem[0:2]) + [pred])

    # 标题匹配结果排序
    title_pred = sort_socres(event_list, title_predicted)
    # 文章内容匹配结果排序
    content_pred = sort_socres(event_list, content_predicted)

    return title_pred, content_pred


if __name__ == '__main__':
    content = '美韩决定举行大规模联合军演'
    title = '韩美取消联合军演'
    data = execute_predict(title, content, sample_type='triples')

    print(data)
