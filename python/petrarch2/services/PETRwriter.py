# -*- coding: utf-8 -*-
# SYSTEM REQUIREMENTS
# This program has been successfully run under Mac OS 10.10; it is standard Python 2.7
# so it should also run in Unix or Windows.
#
# INITIAL PROVENANCE:
# Programmer: Philip A. Schrodt
#			  Parus Analytics
#			  Charlottesville, VA, 22901 U.S.A.
#			  http://eventdata.parusanalytics.com
#
# GitHub repository: https://github.com/openeventdata/petrarch
#
# Copyright (c) 2014	Philip A. Schrodt.	All rights reserved.
#
# This project is part of the Open Event Data Alliance tool set; earlier developments
# were funded in part by National Science Foundation grant SES-1259190
#
# This code is covered under the MIT license
#
# Report bugs to: schrodt735@gmail.com
#
# REVISION HISTORY:
# 22-Nov-13:	Initial version
# ------------------------------------------------------------------------

from __future__ import print_function
from __future__ import unicode_literals
from config.config import Config
# from googletrans import Translator

import PETRglobals  # global variables
import utilities
import codecs
# import openpyxl as xl
# import pandas as pd
import sys
import json
import re
import urllib
import urllib2

reload(sys)
sys.setdefaultencoding('utf8')

# translator = Translator(service_urls=['translate.google.cn'])


def get_actor_text(meta_strg):
    """ Extracts the source and target strings from the meta string. """
    pass


def write_events(event_dict):
    """
    Formats and writes the coded event data to a file in a standard
    event-data format.

    Parameters
    ----------

    event_dict: Dictionary.
                The main event-holding dictionary within PETRARCH.


    output_file: String.
                    Filepath to which events should be written.


    event_xlsx_path: String.
                    临时添加的，用于得出CAMEO中详细的信息.
    """
    global StorySource
    global NEvents
    global StoryIssues

    # workbook = xl.Workbook()
    # sheet = workbook.active
    # headers = [u'源文章文件名', "句子", "主语", '谓语', '宾语', "命名实体识别", "CAMEO事件编号", "CAMEO事件名称", "备注", "样例1", "样例2"]
    # sheet.append(headers)
    # duplication_row = []
    return_subject = []
    return_relation = []
    return_object = []
    return_code = []

    # df = pd.read_excel(event_xlsx_path, header=0)
    for key in event_dict:
        story_dict = event_dict[key]
        if not story_dict['sents']:
            continue    # skip cases eliminated by story-level discard
        #        print('WE1',story_dict)\

        # story_output = []
        filtered_events = utilities.story_filter(story_dict, key)
        #        print('WE2',filtered_events)
        if 'source' in story_dict['meta']:
            StorySource = story_dict['meta']['source']
        else:
            StorySource = 'NULL'
        if 'url' in story_dict['meta']:
            url = story_dict['meta']['url']
        else:
            url = ''
        for event in filtered_events:
            # line_array = []
            story_date = event[0]
            source = event[1]
            target = event[2]
            code = filter(lambda a: not a == '\n', event[3])

            ids_filtered_list = filtered_events[event]['ids']
            ids = ';'.join(ids_filtered_list)

            if 'issues' in filtered_events[event]:
                iss = filtered_events[event]['issues']
                issues = ['{},{}'.format(k, v) for k, v in iss.items()]
                joined_issues = ';'.join(issues)
            else:
                joined_issues = []

            # line_array.append(ids)
            # line_array.append(story_date)
            # line_array.append(source)
            # line_array.append(target)
            print('Event: {}\t{}\t{}\t{}\t{}\t{}'.format(story_date, source,
                                                         target, code, ids,
                                                         StorySource))
            #            event_str = '{}\t{}\t{}\t{}'.format(story_date,source,target,code)
            # 15.04.30: a very crude hack around an error involving multi-word
            # verbs
            if not isinstance(event[3], basestring):
                event_str = '\t'.join(
                    event[:3]) + '\t010\t' + '\t'.join(event[4:])
            else:
                event_str = '\t'.join(event)
            # print(event_str)
            if joined_issues:
                event_str += '\t{}'.format(joined_issues)
            else:
                event_str += '\t'

            if url:
                event_str += '\t{}\t{}\t{}'.format(ids, url, StorySource)
            else:
                event_str += '\t{}\t{}'.format(ids, StorySource)

            summary = ""
            if PETRglobals.WriteActorText:
                if 'actortext' in filtered_events[event]:
                    event_str += '\t{}\t{}'.format(
                        filtered_events[event]['actortext'][0],
                        filtered_events[event]['actortext'][1])
                    summary += '{}  {}'.format(
                        filtered_events[event]['actortext'][0],
                        filtered_events[event]['actortext'][1])
                else:
                    event_str += '\t---\t---'
                    summary += '    --- ---'

            if PETRglobals.WriteEventText:
                if 'eventtext' in filtered_events[event]:
                    event_str += '\t{}'.format(
                        filtered_events[event]['eventtext'])
                    summary += '    {}'.format(
                        filtered_events[event]['eventtext'])
                else:
                    event_str += '\t---'
                    summary += '    ---'

            # 拿到petrarch2使用的动词
            verb_list = []
            for id_sent in ids_filtered_list:
                sent_id = str(id_sent).split("_")[1]
                key = (source, target, code)
                meta = story_dict['sents'][sent_id]['meta']
                if (key in meta.keys()) is not True:
                    continue
                verbs_meta = meta[key]
                for verb_array in verbs_meta:
                    if isinstance(verb_array, list) is not True:
                        continue
                    verb_list = verb_list.__add__(verb_array)

            subject_result = []
            relation_result = []
            object_result = []
            translated_sentence_result = []
            # content = filtered_events[event]['content']
            # word_array = str(filtered_events[event]['words']).split('||')
            content_spo = str(filtered_events[event]['spo']).replace("\"", "\'")
            content_spo = content_spo.replace(" 's ", "'s ")
            content_spo = content_spo.replace("''s'", "'is'")
            content_spo = content_spo.replace(" 's]", " is]")
            content_spo = content_spo.replace("'", "\"")
            content_spo = content_spo.replace("\"s ", "\'s ")
            content_spo = content_spo.replace(": 's ", ": \"s ")
            content_spo = content_spo.replace("\\is", "is ")
            content_spo = content_spo.replace("\" ", "'").replace("\"\"\"", "\"'\"").replace("\"\",", "'\",")
            try:
                verbs = json.loads(content_spo)['verbs']
            except ValueError:
                print("该对象无法分析：" + content_spo)
                print(str(ValueError))
                continue

            # content_namedentity = filtered_events[event]['namedentity']
            # if content_namedentity is not None:
            #     content_namedentity = str(content_namedentity).split('||')
            #     content_namedentity = transform_ner(content_namedentity, word_array)

            for spo_object in verbs:
                sentence, sorten_sentence, constituency_array, english_word_array = \
                    transform_sentence_form(spo_object['description'])
                # [ South Korea\\\'s foreign residents ] [ broke ] [ the million mark ] [ for the first time in 2009 ] ,
                # with 1.106884 million people. 会翻译为[韩国\\\外国居民][2009年首次][突破][百万大关]，为11068.84万人。
                # 可以看到[2009年首次]被提前了，以下代码就是为了解决该问题。
                translated_sentence = transform_en_2_zh(sentence)
                sorten_sentence = transform_en_2_zh(sorten_sentence)
                words_chinese = __extract_chinese(sorten_sentence)
                # subobject_zh, relation_zh, object_zh = __extract_chinese(translated_sentence)
                # if len(subobject_zh) < 1 or len(relation_zh) < 1 or len(object_zh) < 1:
                #     continue
                # try:
                if len(constituency_array) != len(words_chinese):
                    continue
                if len(words_chinese) == 0:
                    print("发生了意外，通常为翻译软件翻译错误导致")
                    continue
                # except TypeError:
                #     print("该中文无法分析：" + translated_sentence)
                #     print(str(TypeError))
                #     continue
                subject_single_result = []
                relation_single_result = []
                objects_single_result = []     # 宾语（非ARG0）可能有多个，暂时先合并这些宾语
                other_single_result = []
                # 暂时认为，若句子中抽取出的成份个数与翻译成中文后的词语个数不一致，跳过该事件，保证准确性
                # if len(constituency_array) != len(subobject_zh) + len(relation_zh) + len(object_zh):
                #     continue
                index = 0
                for constituency in constituency_array:
                    # 有可能句子中第一个成份不是主体，该情况暂时忽略
                    # if index == 0 and constituency != 'ARG0':
                    #     index = index + 1
                    #     continue

                    if constituency == 'ARG0':
                        subject_single_result.append(words_chinese[index])
                    elif constituency == 'V':
                        # 若petrarch2未使用该动词，则不记录该条信息
                        # print('单词列表：' + str(verb_list))
                        # print('动词：' + english_word_array[index].upper())
                        if verb_list.__contains__(english_word_array[index].upper()) is not True:
                            # 将已经记录的数据删除
                            subject_single_result = []
                            relation_single_result = []
                            objects_single_result = []
                            other_single_result = []
                            break

                        relation_single_result.append(words_chinese[index])
                    elif constituency == 'ARG1':
                        objects_single_result.append(words_chinese[index])
                    elif constituency.startswith('ARG'):
                        other_single_result.append(words_chinese[index])
                    else:
                        print('Unrecognized words and ingredients：{} {}'.format(words_chinese[index], constituency))
                    index = index + 1

                if len(subject_single_result) < 1 or len(relation_single_result) < 1 or len(objects_single_result) < 1:
                    continue

                subject_result = subject_result.__add__(subject_single_result)
                relation_result = relation_result.__add__(relation_single_result)
                objects_single_result = objects_single_result.__add__(other_single_result)
                object_result.append(' '.join(objects_single_result))
                translated_sentence_result.append(translated_sentence)
                # translated_sentence_result.append(spo_object['description'])

            if len(subject_result) < 1 or len(relation_result) < 1 or len(object_result) < 1:
                continue

            if subject_result[0] == '' or relation_result[0] == '' or object_result[0] == '':
                continue

            # # 若以下判断成立，表示一个句子中出现抽取出两个事件。
            # if 1 < len(translated_sentence_result) == len(subject_result):
            #     text_subject_result = "\n".join(subject_result)
            # else:
            #     # 若句子中没有出现抽取出两个事件，此时表示句子中出现了两个主语。
            #     text_subject_result = "，".join(subject_result)
            # text_relation_result = "\n".join(relation_result)
            # text_object_result = "\n".join(object_result)
            if 1 < len(translated_sentence_result) == len(subject_result):
                return_subject = return_subject.__add__(subject_result)
            else:
                # 若句子中没有出现抽取出两个事件，此时表示句子中出现了两个主语。
                return_subject.append("，".join(subject_result))
            return_relation = return_relation.__add__(relation_result)
            return_object = return_object.__add__(object_result)
            return_code.append(code)
            # 删除重复行，保证准确性
            # if duplication_row.__contains__((text_subject_result, text_relation_result, text_object_result, code)):
            #     continue

            # line_array.append(filtered_events[event]['source_file'])
            # if len(translated_sentence_result) < 1:
            #     line_array.append(transform_en_2_zh(content))
            # else:
            #     line_array.append("\n".join(translated_sentence_result))
            # line_array.append(text_subject_result)
            # line_array.append(text_relation_result)
            # line_array.append(text_object_result)
            # line_array.append("\n".join(content_namedentity))
            # line_array.append(code)

            # row = df[df['code_num'] == int(code)]
            # line_array.append(row['name'].values[0])
            # line_array.append(row['remark'].values[0])
            # line_array.append(row['example1'].values[0])
            # line_array.append(row['example2'].values[0])
            # try:
            #     sheet.append(line_array)
            # except ValueError:
            #     print(u"该行无法写入xlsx：" + str(line_array))
            #     print(str(ValueError))
            #     continue
            # duplication_row.append((text_subject_result, text_relation_result, text_object_result, code))
            # if PETRglobals.WriteActorRoot:
            #     if 'actorroot' in filtered_events[event]:
            #         event_str += '\t{}\t{}'.format(
            #             filtered_events[event]['actorroot'][0],
            #             filtered_events[event]['actorroot'][1])
            #     else:
            #         event_str += '\t---\t---'
            #
            # story_output.append(event_str)

        # story_events = '\n'.join(story_output)
        # event_output.append(story_events)
    # workbook.save(output_file)
    return return_subject, return_relation, return_object, return_code


def get_event_code(event_dict):
    global StorySource
    global NEvents
    global StoryIssues

    return_code = []

    for key in event_dict:
        story_dict = event_dict[key]
        if not story_dict['sents']:
            continue  # skip cases eliminated by story-level discard
        #        print('WE1',story_dict)\

        # story_output = []
        filtered_events = utilities.story_filter_by_constituency(story_dict, key)
        #        print('WE2',filtered_events)
        if 'source' in story_dict['meta']:
            StorySource = story_dict['meta']['source']
        else:
            StorySource = 'NULL'
        if 'url' in story_dict['meta']:
            url = story_dict['meta']['url']
        else:
            url = ''
        for event in filtered_events:
            # line_array = []
            story_date = event[0]
            source = event[1]
            target = event[2]
            code = filter(lambda a: not a == '\n', event[3])

            ids_filtered_list = filtered_events[event]['ids']
            ids = ';'.join(ids_filtered_list)

            if 'issues' in filtered_events[event]:
                iss = filtered_events[event]['issues']
                issues = ['{},{}'.format(k, v) for k, v in iss.items()]
                joined_issues = ';'.join(issues)
            else:
                joined_issues = []

            print('Event: {}\t{}\t{}\t{}\t{}\t{}'.format(story_date, source,
                                                         target, code, ids,
                                                         StorySource))
            #            event_str = '{}\t{}\t{}\t{}'.format(story_date,source,target,code)
            # 15.04.30: a very crude hack around an error involving multi-word
            # verbs
            if not isinstance(event[3], basestring):
                event_str = '\t'.join(
                    event[:3]) + '\t010\t' + '\t'.join(event[4:])
            else:
                event_str = '\t'.join(event)
            # print(event_str)
            if joined_issues:
                event_str += '\t{}'.format(joined_issues)
            else:
                event_str += '\t'

            if url:
                event_str += '\t{}\t{}\t{}'.format(ids, url, StorySource)
            else:
                event_str += '\t{}\t{}'.format(ids, StorySource)

            summary = ""
            if PETRglobals.WriteActorText:
                if 'actortext' in filtered_events[event]:
                    event_str += '\t{}\t{}'.format(
                        filtered_events[event]['actortext'][0],
                        filtered_events[event]['actortext'][1])
                    summary += '{}  {}'.format(
                        filtered_events[event]['actortext'][0],
                        filtered_events[event]['actortext'][1])
                else:
                    event_str += '\t---\t---'
                    summary += '    --- ---'

            if PETRglobals.WriteEventText:
                if 'eventtext' in filtered_events[event]:
                    event_str += '\t{}'.format(
                        filtered_events[event]['eventtext'])
                    summary += '    {}'.format(
                        filtered_events[event]['eventtext'])
                else:
                    event_str += '\t---'
                    summary += '    ---'

            return_code.append(code)
    return return_code


def transform_en_2_zh(sentence):
    """
    临时用的，将英文翻译成中文

    Parameters
    ----------

    sentence: String.
                句子

    """
    url = Config().translate_url
    data = {"from": "en", "to": "zh", "apikey": Config().translate_user_key, "src_text": sentence}
    data = urllib.urlencode(data)
    data = data.encode()
    res = urllib2.urlopen(url=url, data=data)
    res = res.read()
    res_dict = json.loads(res)

    if "tgt_text" in res_dict:
        content = res_dict['tgt_text']
    else:
        content = res

    return content


def transform_sentence_form(sentence):
    """
    临时用的，转换每个句子的格式，如：[ARG0: the company] [BV: decided to] [V: develop] [ARG1: anti - aging substances] and control the microorganism
    转换并返回：[ the company ] [ decided to ] [ develop ] [ anti - aging substances ] and control the microorganism

    这样的句子结构翻译成中文后不会丢失语义，并且方便抽取成份，此格式只兼容小牛翻译。
    该方法除了返回新的句子外，还将返回标记数组，如：['ARG0', 'BV', 'V', 'ARG1']。
    另外，翻译会出现[ the results ] [ were ] [ marginal ]翻译为[结果][微不足道]的情况，此时的动词需要额外翻译，这是小牛翻译的问题。

    还会返回单词数组：['the company', 'decided to', 'develop', 'anti - aging substances']
    Parameters
    ----------

    sentence: String.
                句子

    """
    sentence = str(sentence).replace('??', ' ')
    pattern = r"[\[]([\w,\(]+:\s.*?)[\]]"
    result = re.compile(pattern).findall(sentence)
    constituency_array = []
    sorten_array = []
    word_array = []
    for word in result:
        word_split = str(word).split(": ")
        constituency = word_split[0]
        # allennlp的主谓宾抽取会出现特殊情况：[BV(ARG0: South] [BV: Korea] [V: will] [ARG1: face] [ARG2: China] on the 15th .\n
        # 可以看到BV(ARG0: South是有问题，下面代码解决这个问题
        constituency = constituency if constituency.__contains__("(") is not True else constituency.split("(")[1]
        constituency_array.append(constituency)
        if constituency == 'V':
            sorten_array.append('`' + word_split[1] + '`')
        else:
            sorten_array.append('[' + word_split[1] + ']')
        word_array.append(word_split[1])

    def re_sub_pattern(value):
        words = str(value.group()).split(": ")
        if words[0] == '[BV':
            new_value = words[1].replace(']', '')
        elif words[0] != '[V':
            new_value = '[' + words[1]
        else:
            new_value = '`' + words[1].replace(']', '`')

        return new_value

    sentence = re.sub(pattern, re_sub_pattern, sentence)
    sorten_sentence = ' '.join(sorten_array)

    return sentence, sorten_sentence, constituency_array, word_array


def __extract_chinese(sentence):

    if len(sentence) < 1:
        return []

    if isinstance(sentence, list):
        sentence = '，'.join('%s' % word for word in sentence)
    sentence = sentence.replace("-LRB-", "（").replace("-RRB-", "）")
    object_pattern = r"[\[,`](.*?)[\],`|'$]"
    result = re.compile(object_pattern).findall(sentence)

    return result


def transform_ner(en_types, word_array):
    """
    临时用的

    Parameters
    ----------

    en_type: String.
    word_array: str.

    """
    def transform(en_type):
        if en_type == 'PER':
            return u'人'
        elif en_type == 'LOC':
            return u'地点'
        elif en_type == 'ORG':
            return u'组织机构'
        elif en_type == 'MISC':
            return u'杂项'
        else:
            return u'无'

    word = ''
    result_word = []
    for index, value in enumerate(en_types):
        value = str(value)
        if value == 'O':
            continue
        elif value.startswith("B-"):
            word = word_array[index]
        elif value.startswith("L-"):
            word = word + word_array[index]
            value_mark = value.split('-')
            value_mark = value_mark[1] if len(value_mark) == 2 else value_mark[0]
            result_word.append(u"单词：" + word + u"，实体名：" + transform(value_mark))
            word = ''
        else:
            if word == '':
                word = word_array[index]
                value_mark = value.split('-')
                value_mark = value_mark[1] if len(value_mark) == 2 else value_mark[0]
                result_word.append(u"单词：" + word + u"，实体名：" + transform(value_mark))
                word = ''
            else:
                word = word + word_array[index]

    return result_word


def write_nullverbs(event_dict, output_file):
    """
    Formats and writes the null verb data to a file as a set of lines in a JSON format.

    Parameters
    ----------

    event_dict: Dictionary.
                The main event-holding dictionary within PETRARCH.


    output_file: String.
                    Filepath to which events should be written.
    """

    def get_actor_list(item):
        """ Resolves the various ways an actor could be in here """
        if isinstance(item, list):
            return item
        elif isinstance(item, tuple):
            return item[0]
        else:
            return [item]

    event_output = []
    for key, value in event_dict.iteritems():
        if not 'nulls' in value['meta']:
            # print('Error:',value['meta'])  # log this and figure out where it
            # is coming from <later: it occurs for discard sentences >
            continue
        for tup in value['meta']['nulls']:
            if not isinstance(tup[0], int):
                srclst = get_actor_list(tup[1][0])
                tarlst = get_actor_list(tup[1][1])
                jsonout = {'id': key,
                           # <16.06.28 pas> With a little more work we could get the upper/lower
                           'sentence': value['text'],
                           # case version -- see corresponding code in
                           # write_nullactors() -- but
                           'source': ', '.join(srclst),
                           'target': ', '.join(tarlst)}  # hoping to refactor 'meta' and this will do for now.
                if jsonout['target'] == 'passive':
                    continue
                if '(S' in tup[0]:
                    parstr = tup[0][:tup[0].index('(S')]
                else:
                    parstr = tup[0]
                jsonout['parse'] = parstr
                phrstr = ''
                for ist in parstr.split(' '):
                    if ')' in ist:
                        phrstr += ist[:ist.index(')')] + ' '
                jsonout['phrase'] = phrstr

                event_output.append(jsonout)

    if output_file:
        f = codecs.open(output_file, encoding='utf-8', mode='w')
        for dct in event_output:
            f.write('{\n')
            for key in ['id', 'sentence', 'phrase', 'parse']:
                f.write('"' + key + '": "' + dct[key] + '",\n')
            f.write('"source": "' + dct['source'] +
                    '", "target": "' + dct['target'] + '"\n}\n')


def write_nullactors(event_dict, output_file):
    """
    Formats and writes the null actor data to a file as a set of lines in a JSON format.

    Parameters
    ----------

    event_dict: Dictionary.
                The main event-holding dictionary within PETRARCH.


    output_file: String.
                    Filepath to which events should be written.
    """

    global hasnull

    def get_actor_text(evt, txt, index):
        """ Adds code when actor is in dictionary; also checks for presence of null actor """
        global hasnull
        text = txt[index]
        if evt[index].startswith('*') and evt[index].endswith('*'):
            if txt[
                    index]:  # system occasionally generates null strings -- of course... -- so might as well skip these
                hasnull = True
        else:
            text += ' [' + evt[index] + ']'
        return text

    event_output = []
    for key, value in event_dict.iteritems():
        if not value['sents']:
            continue
        for sent in value['sents']:
            if 'meta' in value['sents'][sent]:
                if 'actortext' not in value['sents'][sent]['meta']:
                    continue
                for evt, txt in value['sents'][sent]['meta']['actortext'].iteritems(
                ):  # <16.06.26 pas > stop the madness!!! -- we're 5 levels deep here, which is as bad as TABARI. This needs refactoring!
                    hasnull = False
                    jsonout = {'id': key,
                               'sentence': value['sents'][sent]['content'],
                               'source': get_actor_text(evt, txt, 0),
                               'target': get_actor_text(evt, txt, 1),
                               'evtcode': evt[2],
                               'evttext': ''
                               }
                    if hasnull:
                        if evt in value['sents'][sent]['meta']['eventtext']:
                            jsonout['evttext'] = value['sents'][
                                sent]['meta']['eventtext'][evt]

                        event_output.append(jsonout)

    if output_file:
        f = codecs.open(output_file, encoding='utf-8', mode='w')
        for dct in event_output:
            f.write('{\n')
            for key in ['id', 'sentence', 'source',
                        'target', 'evtcode', 'evttext']:
                f.write('"' + key + '": "' + dct[key] + '",\n')
            f.write('}\n')


def pipe_output(event_dict):
    """
    Format the coded event data for use in the processing pipeline.

    Parameters
    ----------

    event_dict: Dictionary.
                The main event-holding dictionary within PETRARCH.


    Returns
    -------

    final_out: Dictionary.
                StoryIDs as the keys and a list of coded event tuples as the
                values, i.e., {StoryID: [(full_record), (full_record)]}. The
                ``full_record`` portion is structured as
                (story_date, source, target, code, joined_issues, ids,
                StorySource) with the ``joined_issues`` field being optional.
                The issues are joined in the format of ISSUE,COUNT;ISSUE,COUNT.
                The IDs are joined as ID;ID;ID.

    """
    final_out = {}
    for key in event_dict:
        story_dict = event_dict[key]
        if not story_dict['sents']:
            continue    # skip cases eliminated by story-level discard
        filtered_events = utilities.story_filter(story_dict, key)
        if 'source' in story_dict['meta']:
            StorySource = story_dict['meta']['source']
        else:
            StorySource = 'NULL'
        if 'url' in story_dict['meta']:
            url = story_dict['meta']['url']
        else:
            url = ''

        if filtered_events:
            story_output = []
            for event in filtered_events:
                story_date = event[0]
                source = event[1]
                target = event[2]
                code = event[3]

                ids = ';'.join(filtered_events[event]['ids'])

                if 'issues' in filtered_events[event]:
                    iss = filtered_events[event]['issues']
                    issues = ['{},{}'.format(k, v) for k, v in iss.items()]
                    joined_issues = ';'.join(issues)
                    event_str = (story_date, source, target, code,
                                 joined_issues, ids, url, StorySource)
                else:
                    event_str = (story_date, source, target, code, ids,
                                 url, StorySource)

                story_output.append(event_str)

            final_out[key] = story_output
        else:
            pass

    return final_out


if __name__ == '__main__':

    string_ip = u"8月22日晚，通过俄罗斯PlaneRadar网站对空监视雷达的探测，发现一架[临时]`侵入'[克里米亚领空]的[美国海军洛克希德EP-3E白羊座II]远程电子侦察机侵入俄罗斯克里米亚黑海海岸线进行侦察！"
    result = __extract_chinese(string_ip)
    for elm in result:
        print(elm)
