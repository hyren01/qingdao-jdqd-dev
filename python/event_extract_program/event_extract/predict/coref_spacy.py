#!/usr/bin/env python
# coding : utf-8
import spacy
import neuralcoref
from feedwork.utils import logger as LOG

def get_spacy():
    """
    加载指代消解模型，返回模型对象。
    :return: nlp指代模型对象
    """
    # 加载spacy模型
    LOG.info("开始加载spacy模型。。。")
    nlp = spacy.load('en_core_web_sm')
    coref = neuralcoref.NeuralCoref(nlp.vocab)
    nlp.add_pipe(coref, name='neuralcoref')
    LOG.info("spacy模型加载完成!")

    return nlp


def coref_data(nlp, line):
    """
    传入nlp模型和待消解的英文字符串，对英文文本进行指代消解，返回消解后的字符串。
    :param nlp: 指代消解模型对象
    :param line: (str)待消解的文本
    :return: res(str)消解后的文本
    """
    if not isinstance(line, str):
        LOG.error("The type of content for coreference must be string!")
        raise TypeError
    doc = nlp(line)
    res = doc._.coref_resolved

    return res


if __name__ == "__main__":

    data = "今晚国足进行比赛，他们信心满满，但是他们不一定能赢。"
    # res = execute(data)
    # print(res)



