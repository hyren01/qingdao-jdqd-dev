#!/usr/bin/env python
# -*- coding:utf-8 -*-

from stanfordcorenlp import StanfordCoreNLP
from config.config import Config
import json
import logging
import re

regular_pattern = r"Sentence\s#\d+\s\(\d+\stokens\):\r\n(.+)\r\n\r\nTokens:"
stanfordnlp_path = Config().stanfordnlp_path + "/"
__nlp = StanfordCoreNLP(stanfordnlp_path, lang='en', logging_level=logging.DEBUG)


def article_to_sentences(article):
    """
    将一篇文章转换为句子，这些句子以list返回

    """
    if article is None or article == '':
        return []

    props = {'annotators': 'ssplit', 'pipelineLanguage': 'en', 'outputFormat': 'text'}
    ssplit_text = __nlp.annotate(article, properties=props)
    sentences = re.findall(regular_pattern, ssplit_text)

    return sentences


def constituency_parse_with_sentence(sentence):
    """
    分析句子的组成成份，以文本方式返回

    """
    if sentence is None or sentence == '':
        return ''

    props = {'annotators': 'parse', 'pipelineLanguage': 'en', 'outputFormat': 'json'}
    parse_json = __nlp.annotate(sentence, properties=props)
    parse_jsonarray = json.loads(parse_json)['sentences']
    parse_text = str(parse_jsonarray[0]['parse'])  # 按句子的级别进行成份分析，所有下标永远是0

    return parse_text


def openie_with_sentence(sentence):
    """
    抽取句子的主谓宾，主谓宾以数组方式返回

    """
    if sentence is None or sentence == '':
        return []

    props = {'annotators': 'openie', 'pipelineLanguage': 'en', 'outputFormat': 'json'}
    ie_json = __nlp.annotate(sentence, properties=props)
    ie_jsonarray = json.loads(ie_json)['sentences']
    spo_text = str(ie_jsonarray[0]['openie'])  # 按句子的级别抽取主谓宾，所有下标永远是0

    return spo_text


def ner_with_sentence(sentence):
    """
    句子的命名实体识别，实体数据以数组方式返回

    """
    if sentence is None or sentence == '':
        return []

    props = {'annotators': 'ner', 'pipelineLanguage': 'en', 'outputFormat': 'json'}
    ner_json = __nlp.annotate(sentence, properties=props)
    ner_json = json.loads(ner_json)['sentences']
    namedentity_text = str(ner_json[0]['entitymentions'])

    return namedentity_text


def coref_with_sentence(sentence):
    """
    句子的代词消解，实体数据以元组方式返回

    """
    if sentence is None or sentence == '':
        return []

    props = {'annotators': 'coref', 'pipelineLanguage': 'en', 'outputFormat': 'json'}
    coref_json = __nlp.annotate(sentence, properties=props)
    coref_json = json.loads(coref_json)['corefs']

    return coref_json


def close():
    __nlp.close()


if __name__ == '__main__':

    sentence = 'North Korea announced this morning that it would suspend the military exercises to be held next Friday.'
    print(openie_with_sentence(sentence))
    print(ner_with_sentence(sentence))
    print(coref_with_sentence(sentence))
    close()
