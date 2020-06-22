#!/usr/bin/env python
# -*- coding:utf-8 -*-

from config.config import Config
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter

__allennlp_constituency_parser_model_path = Config().allennlp_constituency_parser_model_path
__constituency_predictor = Predictor.from_path(__allennlp_constituency_parser_model_path)
__allennlp_dependency_model_path = Config().allennlp_dependency_parser_model_path
__dependency_predictor = Predictor.from_path(__allennlp_dependency_model_path)
__allennlp_openie_model_path = Config().allennlp_openie_model_path
__openie_predictor = Predictor.from_path(__allennlp_openie_model_path)
__allennlp_namedentity_model_path = Config().allennlp_namedentity_model_path
__namedentity_predictor = Predictor.from_path(__allennlp_namedentity_model_path)
__allennlp_coref_model_path = Config().allennlp_coref_model_path
__coref_predictor = Predictor.from_path(__allennlp_coref_model_path)
# __allennlp_sentiment_analysis_model_path = Config().sentiment_analysis_model_path
# __sentiment_analysis_predictor = Predictor.from_path(__allennlp_sentiment_analysis_model_path)


def article_to_sentences(article):
    """
    将一篇文章转换为句子，这些句子以list返回

    """
    if article is None or article == '':
        return []

    splitter = SpacySentenceSplitter(language='en_core_web_sm', rule_based=False)
    sentences = splitter.split_sentences(article)

    return sentences


def constituency_parse_with_sentence(sentence):
    """
    分析句子的组成成份，以文本方式返回

    """
    if sentence is None or sentence == '':
        return ''

    parse_json = __constituency_predictor.predict(sentence=sentence)
    parse_text = parse_json['trees']  # 按句子的级别进行成份分析

    return parse_text


def denpendency_parse_with_sentence(sentence):
    """
    分析句子的依赖成份，以文本方式返回

    """
    if sentence is None or sentence == '':
        return ''

    parse_json = __dependency_predictor.predict(sentence=sentence)  # 按句子的级别进行成份分析
    parse_text = parse_json['hierplane_tree']

    return parse_text


def openie_with_sentence(sentence):
    """
    抽取句子的主谓宾，主谓宾以json方式返回

    """
    if sentence is None or sentence == '':
        return []

    spo_json = __openie_predictor.predict(sentence=sentence)

    return spo_json


def ner_with_sentence(sentence):
    """
    句子的命名实体识别，实体数据以数组方式返回

    """
    if sentence is None or sentence == '':
        return []

    ner_json = __namedentity_predictor.predict(sentence=sentence)

    return ner_json['words'], ner_json['tags']


def coref_with_article(article):
    """
    文章/文本的代词消解，返回消解后的文章/文本

    """
    if article is None or article == '':
        return ''
    article = __coref_predictor.coref_resolved(article)

    return article


def coref_with_sentence(sentence):
    """
    句子的代词消解，实体数据以元组方式返回

    """
    if sentence is None or sentence == '':
        return []

    coref_json = __coref_predictor.predict(document=sentence)
    '''
    描述以下代码数据处理所根据的含义。
    原始数据：
    {'top_spans': [[0, 1], [2, 2], [3, 4], [6, 6], [8, 8], [9, 11], [15, 16]], 'predicted_antecedents': 
    [-1, -1, -1, 2, -1, -1, -1], 'document': ['North', 'Korea', 'announced', 'this', 'morning', 'that', 'it', 'would', 'suspend', 'the', 'military', 'exercises', 'to', 'be', 'held', 'next', 'Friday', '.'], 
    'clusters': [[[0, 1], [6, 6]]]}
    clusters的第一层数组为多个cluster的含义，第二层数组表示为指代关系组，该组下标为0是指代关系的主体，下标除0以外的一般指代词；
    [0, 1]及[6, 6]指的是document对象中下标为0和1的（即North Korea）为指代关系的主体，下标为6和6的（即it）为代词；
    [9, 11]指的是document对象中下标为9、10、11的单词所组合成的一个词语。
    '''
    document = coref_json['document']   # 表示句子被分割成数组形式的单词
    clusters = coref_json['clusters']   # 表示对不同的指代关系进行分类的数组
    pronoun_array_map = []  # 对分析后的结果进行格式化，按照不同的指代关系中的主体进行分组
    for cluster in clusters:    # cluster表示不同的指代关系中的主体的一个组
        pronoun_map = {'subject': None, 'pronoun': []}
        for index, value in enumerate(cluster):
            index_range = range(value[0], value[1] + 1)     # 因为range(0, 1)只能得到0，而我想要的是0、1，所以+1
            word = []
            for word_index in index_range:
                word.append(document[word_index])
            word = ' '.join(word)
            if index == 0:
                pronoun_map['subject'] = word
            else:
                pronoun_map['pronoun'].append({'word': word, 'index': value})
        pronoun_array_map.append(pronoun_map)

    for pronoun_map in pronoun_array_map:
        subject = pronoun_map['subject']
        pronoun = pronoun_map['pronoun']
        for value in pronoun:
            index = value['index']
            index_range = range(index[0], index[1] + 1)
            for word_index in index_range:
                document[word_index] = subject

    return pronoun_array_map, ' '.join(document)


# def __sentiment_analysis(sentence):
#
#     if sentence is None or sentence == '':
#         return []
#
#     sentiment_analysis_json = __sentiment_analysis_predictor.predict(sentence=sentence)
#
#     return sentiment_analysis_json



def close():
    pass


if __name__ == '__main__':

    sentence = "North Korea notified the International Maritime Organization (IMO) on the morning of the 6th London time to adjust the launch time of the \"Earth Observation Satellite\" to between February 7 and 14."
    print(openie_with_sentence(sentence))
