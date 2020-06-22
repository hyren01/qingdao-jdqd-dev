#!/usr/bin/env python
# -*- coding:utf-8 -*-

from utils.extract_util import extractor_word, extract_2_chinese
from utils.translate_util import translate_any_2_anyone

import logging
import services.allen_nlp as nlp

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s -%(message)s")


def event_extract_helper(sentence):
    """
    事件抽取接口的实现代码。
    :param sentence: String.句子。
    :return [{"subject":"", "verb":"", "object":"", "short_sentence":"",
                    "namedentity":{"person":"", "location":"", "organization":"", "miscellaneous":""},
                    "sentiment_analysis":"", event_datetime":"", "event_location":"", "negative_word":"",
                    "state":"", "event_type":""}]。
    """
    # 1、将传入的文本翻译为英文。
    sentence_txt = translate_any_2_anyone(sentence, target="en")
    # 2、去除处理因翻译问题，导致的错误字符问题后，调用allennlp进行主谓宾抽取，allennlp返回固定格式的英文文本，如：
    # [ARG0: South] [BV: Korea] [V: will] [ARG1: face] [ARG2: China] on the 15th .
    sentence_txt = sentence_txt.replace("`` ", "' ").replace(" ''", " '")
    sentence_txt = sentence_txt.replace("\"", "'").replace(",'", "',").replace("`s", "'s")
    sentence_txt = sentence_txt.replace("\\'", "'")
    content_spo = nlp.openie_with_sentence(sentence_txt)
    verbs = content_spo['verbs']

    if len(verbs) < 1:
        return []

    event_result = []
    for spo_object in verbs:
        # 3、对allennlp返回的固定格式的英文文本进行元素提取，提取出文本中的主、谓、宾。
        subject_array, relation_array, object_array = extractor_word(spo_object['description'])
        if len(subject_array) < 1 or len(relation_array) < 1 or len(object_array) < 1:
            logging.info("抽取主谓宾时，主语、谓语、宾语其中一项缺失：{msg}".format(msg=sentence_txt))
            continue
        # 4、对英文元素按固定格式拼接成字符串，将该字符串翻译为中文，最后再提取出中文元素。
        success, zh_subject, zh_verb, zh_object = extract_2_chinese(subject_array, relation_array, object_array)
        if success is not True:
            zh_subject = ",".join(subject_array)
            zh_verb = ",".join(relation_array)
            zh_object = ",".join(object_array)

        event = {"subject": zh_subject, "verb": zh_verb, "object": zh_object}

        sorten_sentence = " ".join(subject_array) + " " + " ".join(relation_array) + " " + " ".join(object_array)
        event["short_sentence"] = translate_any_2_anyone(sorten_sentence, target="zh")

        word_array, namedentity_array = nlp.ner_with_sentence(sorten_sentence)
        # ["B-LOC", "L-LOC", "O", "B-LOC", "L-LOC", "O", "U-LOC", "O", "O", "O", "O", "O", "O", "O", "U-LOC"]
        namedentity_word = []
        namedentity_tag = []
        temp_word = ''
        # 识别固定格式的命名实体标签
        for index, namedentity in enumerate(namedentity_array):
            if namedentity == 'O':
                continue
            tags = namedentity.split('-')
            profix = tags[0]
            tag = tags[1]
            word = word_array[index]

            if profix == 'U' or profix == 'S':  # 表示该单词是一个实体
                namedentity_word.append(word)
                namedentity_tag.append(tag)
            elif profix == 'B':  # 表示该单词是一个实体的开始
                temp_word = word
            elif profix == 'L':  # 表示该单词是一个实体的结束
                temp_word = temp_word + " " + word
                namedentity_word.append(temp_word)
                namedentity_tag.append(tag)
                temp_word = ''
            elif profix == 'I':   # 表示该单词在一个实体的内部
                temp_word = temp_word + " " + word
            else:
                pass

        person = []
        location = []
        organization = []
        miscellaneous = []

        if len(namedentity_word) == len(namedentity_tag):
            for index, tag in enumerate(namedentity_tag):
                word = namedentity_word[index]
                word = translate_any_2_anyone(word, source="en", target='zh')
                if tag == 'LOC':
                    location.append(word)
                elif tag == 'PER':
                    person.append(word)
                elif tag == 'MISC':
                    miscellaneous.append(word)
                elif tag == 'ORG':
                    organization.append(word)
                else:
                    pass
        # 'nameentity':{'person':['特朗普'], 'tag':'person'}]
        event["namedentity"] = {'person': ",".join(person), 'location': ",".join(location),
                                'organization': ",".join(organization), 'miscellaneous': ",".join(miscellaneous)}

        # 缺情感分析
        event["sentiment_analysis"] = ""
        event["event_datetime"] = ""
        event["event_location"] = ""  # 英文逗号分隔
        event["negative_word"] = ""  # 英文逗号分隔
        event["state"] = ""
        event["event_type"] = ""  # CAMEO CODE

        event_result.append(event)

    return event_result


def constituency_parsed_helper(sentence):
    """
    组成成份分析接口的实现代码。
    :param sentence: String.句子。
    :return 组成成份，如：(S (NP (JJ High) (HYPH -) (NN level) (NN security) (NNS consultations) (PP (IN between)
                         (NP (NP (NNP South) (NNP Korea)) (, ,) (NP (DT the) (NNP United) (NNP States)) (CC and)
                         (NP (NNP Japan))))) (VP (VBP have) (VP (VBN been) (VP (VBN held) (PP (IN in)
                         (NP (NNP Washington)))))) (. .))。
    """
    sentence_txt = translate_any_2_anyone(sentence, target="en")
    sentence_txt = str(sentence_txt)
    sentence_txt = sentence_txt.replace("(", " -LRB- ").replace(")", " -RRB- ")
    sentence_txt = sentence_txt.replace("（", " -LRB- ").replace("）", " -RRB- ")
    constituency_parse_text = nlp.constituency_parse_with_sentence(sentence_txt)
    constituency_parse_text = str(constituency_parse_text).replace("NNP -LRB-", "-LRB- -LRB-").replace("NNP -RRB-",
                                                                                                       "-RRB- -RRB-")

    return constituency_parse_text


def coref_with_article_helper(content):
    """
    指代消解接口的实现代码。
    :param content: String.文本。
    :return 组成成份，如：指代消解后的英文文章。
    """
    # 遗留问题，翻译只支持5000个字符。不能使用固定字符分句，因为：1、要想办法知道文章所属语言；2、句子存在很多复杂形式，如:"xxxx。"
    content = translate_any_2_anyone(content, target="en")
    if content == '' or type(content) is bytes:     # 类型为bytes表示翻译发生错误
        return False, "翻译时超过5000字符"
    content = nlp.coref_with_article(content)

    return True, content


if __name__ == '__main__':

    sentence = "North Korea notified the International Maritime Organization (IMO) on the morning of the 6th London " \
               "time to adjust the launch time of the \"Earth Observation Satellite\" to between February 7 and 14."
    print(event_extract_helper(sentence))
