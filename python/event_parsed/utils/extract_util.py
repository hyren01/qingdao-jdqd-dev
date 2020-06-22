#!/usr/bin/env python
# -*- coding:utf-8 -*-

import re
from utils.translate_util import translate_any_2_anyone


def extractor_word(sentence):
    """
    从特定格式的英文句子中抽取元素，如：[ARG0: South] [BV: Korea] [V: will] [ARG1: face] [ARG2: China] on the 15th .
    :param sentence: String.特定格式的英文句子。
    :return 主语数组、谓语动词数组、宾语数组。
    """
    sentence = str(sentence).replace('??', ' ')
    pattern = r"[\[]([\w,\(]+:\s.*?)[\]]"
    result = re.compile(pattern).findall(sentence)
    subject_array = []
    relation_array = []     # 谓语
    object_array = []
    for word in result:
        word_split = str(word).split(": ")
        constituency = word_split[0]
        word = word_split[1]
        # allennlp的主谓宾抽取会出现特殊情况：[BV(ARG0: South] [BV: Korea] [V: will] [ARG1: face] [ARG2: China] on the 15th .\n
        # 可以看到BV(ARG0: South是有问题，下面代码解决这个问题
        constituency = constituency if constituency.__contains__("(") is not True else constituency.split("(")[1]
        if constituency == 'ARG0':
            subject_array.append(word)
        elif constituency == 'V':
            relation_array.append(word)
        elif constituency == 'BV':
            pass
        else:
            object_array.append(word)

    return subject_array, relation_array, object_array


def extract_2_chinese(subject, predicate, object):
    """
    对传入英语的主语、谓语、宾语按固定格式组合，经过翻译后按照固定格式抽取出中文元素。
    :param subject: array.英文主语。
    :param predicate: array.英文谓语。
    :param object: array.英文宾语。
    :return 主语、谓语动词、宾语。
    """
    subject = "[" + " ".join(subject) + "]"
    predicate = "[" + predicate[0] + "]"
    object = "[" + " ".join(object) + "]"

    sentence = subject + predicate + object
    sentence = sentence.replace("-LRB-", "（").replace("-RRB-", "）")
    sentence = translate_any_2_anyone(sentence, target="zh")

    object_pattern = r"[\[,`](.*?)[\],`|'$]"
    result = re.compile(object_pattern).findall(sentence)

    if len(result) != 3:
        return False, '', '', ''

    return True, result[0], result[1], result[2]