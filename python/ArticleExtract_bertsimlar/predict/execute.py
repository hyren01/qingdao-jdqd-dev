# coding: utf-8

#from .BertSimlar import bert_predict
from .data_util import process_data
from .logger import Logger
import os
import re

LOG = Logger('info')

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
        return False

def file_reader(file_path):

    '''
    传入文件路径，读取文件内容
    :param file_path: 文件路径
    :return: 文件内容--str
    '''

    try:
        with open(file_path,'r', encoding = 'gbk') as file:
            content = file.read()
        return content
    except Exception as e:
        with open(file_path,'r',encoding='utf-8') as file:
            content = file.read()
        return content


def generate_samples(query, sentences):

    '''
    传入事件列表和语句列表，生成匹配对进行预测
    :param event_list: 事件列表--list
    :param sentences: 语句列表--list
    :return: 样本[(query, sentence, 0), ]格式为元组列表
    '''
    samples = []
    for once in sentences:
        if once:
            samples.append((query,once,str(0)))
    return samples


def cut_sentence(content):

    '''
    传入文本字符串，进行分句
    :param content: 文本字符串
    :return: 句子列表--list
    '''
    sentences = re.split('[。?？!！]', content)

    return sentences


def sort_socres(results):
    '''
    传入预测结果，对结果进行排序
    :param results: 预测结果{article_id:score}--dict
    :return: sorted_results [{'article_id':'', 'score':}, ] --list
    '''
    sorted_results = list(sorted(results.items(), key=lambda e: e[1], reverse=True))
    sorted_results = [{'article_id': elem[0], 'score': elem[1]} for elem in sorted_results]

    return sorted_results


#def execute_predict(query, file_dir = '../resources/allfile'):
#    '''
#    传入待搜索语句，文件夹路径以及文章片段
#    :param query: 待搜索语句--str
#    :param file_dir: 文本存储文件夹路径
#    :return: results 预测得到的结果 [{title:score}, ]  字典列表
#    '''
#    assert type(query) == str
#
#    # 获取文件列表读取文件内容
#    file_list = os.listdir(file_dir)
#    results = {}
#    for file in file_list:
#        file_path = os.path.join(file_dir, file)
#        content = file_reader(file_path)
#        content = process_data(content)
#        sentences = cut_sentence(content)
#        sentences.append(file.replace('.txt', ''))
#        samples = generate_samples(query, sentences)
#        pred = bert_predict.get_result(samples)
#        results[file.replace('.txt', '')] = max(pred)
#
#    results = sort_socres(results)
#
#    return results




if __name__ == '__main__':

    content = '美韩决定举行大规模联合军演'
    data = execute_predict(content, file_dir = '../resources/allfile')


    print(data)