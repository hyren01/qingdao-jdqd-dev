# coding: utf-8
import os
import sys

file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)
dir_path = os.path.join(dir_path, "../")
sys.path.append(dir_path)

import jieba
import re
import jieba.posseg

from pyltp import Segmentor, Postagger, Parser, NamedEntityRecognizer, SementicRoleLabeller
from predict import config

Config = config.Config()

# 传入jieba分词字典路径
vocab_path = os.path.join(dir_path, Config.vocab_path)
jieba.load_userdict(vocab_path)


class LtpParser:
    '''构建ltp解析类'''

    def __init__(self):

        # LTP模型的保存路径，加载各个模型
        self.LTP_DIR = os.path.join(dir_path, Config.ltp_data)
        self.postagger = Postagger()
        self.postagger.load(os.path.join(self.LTP_DIR, Config.pos_model))

        self.parser = Parser()
        self.parser.load(os.path.join(self.LTP_DIR, Config.parser_model))

        self.recognizer = NamedEntityRecognizer()
        self.recognizer.load(os.path.join(self.LTP_DIR, Config.ner_model))

        self.labeller = SementicRoleLabeller()
        self.labeller.load(os.path.join(self.LTP_DIR, Config.role_model))

    def format_labelrole(self, words, postags):
        '''
        对传入的分词语料和词性标注语料进行语义角色标注
        :param words: 句子词列表
        :param postags: 句子词性列表
        :return: 语义角色字典
        '''
        arcs = self.parser.parse(words, postags)
        roles = self.labeller.label(words, postags, arcs)
        roles_dict = {}
        for role in roles:
            roles_dict[role.index] = {arg.name: [arg.name, arg.range.start, arg.range.end] for arg in role.arguments}
        return roles_dict

    def build_parse_child_dict(self, words, postags, arcs):
        '''
        句法分析---为句子中的每个词语维护一个保存句法依存儿子节点的字典
        对依存句法树进行解析，构建子树
        :param words: 词列表
        :param postags: 词性列表
        :param arcs: 依存句法树
        :return: 依存句法子树，
        '''
        child_dict_list = []
        format_parse_list = []
        for index in range(len(words)):
            child_dict = dict()
            for arc_index in range(len(arcs)):
                if arcs[arc_index].head == index + 1:
                    if arcs[arc_index].relation in child_dict:
                        child_dict[arcs[arc_index].relation].append(arc_index)
                    else:
                        child_dict[arcs[arc_index].relation] = []
                        child_dict[arcs[arc_index].relation].append(arc_index)
            child_dict_list.append(child_dict)  # 记录每个word子节点的relation及索引
        # 格式化依存关系列表
        rely_id = [arc.head for arc in arcs]  # 提取依存父节点id
        relation = [arc.relation for arc in arcs]  # 提取依存关系
        heads = ['Root' if id == 0 else words[id - 1] for id in rely_id]  # 匹配依存父节点词语
        for i in range(len(words)):
            # ['ATT', '李克强', 0, 'nh', '总理', 1, 'n']
            a = [relation[i], words[i], i, postags[i], heads[i], rely_id[i] - 1, postags[rely_id[i] - 1]]
            format_parse_list.append(a)

        return child_dict_list, format_parse_list

    def parser_main(self, sentence):
        '''
        传入句子字符串，进行解析
        :param sentence: 句子字符串
        :return: 词列表、词性列表、句子子树、语义角色字典、个数化
        '''
        words = list(jieba.cut(sentence))
        postags = list(self.postagger.postag(words))
        arcs = self.parser.parse(words, postags)
        child_dict_list, format_parse_list = self.build_parse_child_dict(words, postags, arcs)
        roles_dict = self.format_labelrole(words, postags)

        return words, postags, child_dict_list, roles_dict, format_parse_list

    def release_model(self):
        # 释放模型
        self.postagger.release()
        self.parser.release()
        self.recognizer.release()
        self.labeller.release()


class TripleExtractor:
    '''
    三元组抽取类
    '''

    def __init__(self):
        self.parser = LtpParser()

    def split_sents(self, content):
        '''
        对传入的中文字符串进行分句
        :param content: 文章字符串
        :return: 句子列表
        '''
        return [sentence for sentence in re.split(r'[？?！!。；;：:]', content) if sentence]

    def ruler1(self, words, postags, roles_dict, role_index):
        '''
        利用语义角色标注直接获取主谓宾三元组，基于A0,A1,A2
        :param words: 词列表
        :param postags: 词性列表
        :param roles_dict: 角色字典
        :param role_index: 角色id列表
        :return: 主谓宾三元组
        '''
        v = words[role_index]
        role_info = roles_dict[role_index]

        if 'A0' in role_info.keys() and 'A1' in role_info.keys():
            s = ''.join([words[word_index] for word_index in range(role_info['A0'][1], role_info['A0'][2] + 1) if
                         postags[word_index][0] not in ['w', 'u', 'x'] and words[word_index]])
            o = ''.join([words[word_index] for word_index in range(role_info['A1'][1], role_info['A1'][2] + 1) if
                         postags[word_index][0] not in ['w', 'u', 'x'] and words[word_index]])
            if s and o:
                return '0', [s, v, o]

        elif 'A1' in role_info.keys() and 'A2' in role_info.keys():
            s = ''.join([words[word_index] for word_index in range(role_info['A1'][1], role_info['A1'][2] + 1) if
                         postags[word_index][0] not in ['w', 'u', 'x'] and words[word_index]])
            o = ''.join([words[word_index] for word_index in range(role_info['A2'][1], role_info['A2'][2] + 1) if
                         postags[word_index][0] not in ['w', 'u', 'x'] and words[word_index]])
            if s and o:
                return '1', [s, v, o]

        elif 'A3' in role_info.keys() and 'A1' in role_info.keys():
            s = ''.join([words[word_index] for word_index in range(role_info['A3'][1], role_info['A3'][2] + 1) if
                         postags[word_index][0] not in ['w', 'u', 'x'] and words[word_index]])
            o = ''.join([words[word_index] for word_index in range(role_info['A1'][1], role_info['A1'][2] + 1) if
                         postags[word_index][0] not in ['w', 'u', 'x'] and words[word_index]])
            if s and o:
                return '2', [s, v, o]

        return '4', []

    def ruler2(self, words, postags, child_dict_list, arcs, roles_dict):
        '''
        抽取三元组主函数
        :param words: 词列表
        :param postags: 词性列表
        :param child_dict_list: 子树列表
        :param arcs: 依存句法树
        :param roles_dict: 语义角色字典
        :return: 主谓宾三元组列表
        '''
        svos = []
        for index in range(len(postags)):
            tmp = 1
            # 先借助语义角色标注的结果，进行三元组抽取
            if index in roles_dict:
                flag, triple = self.ruler1(words, postags, roles_dict, index)
                if flag in ['0', '1', '2']:
                    # triple[1] = self.complete_str(words, index, child_dict_list, 'ADV')
                    svos.append(triple)
                    tmp = 0
            if tmp == 1:
                # 如果语义角色标记为空，则使用依存句法进行抽取
                # if postags[index] == 'v':
                if postags[index]:
                    # 抽取以谓词为中心的事实三元组
                    child_dict = child_dict_list[index]
                    # 主谓宾
                    if 'SBV' in child_dict and 'VOB' in child_dict:
                        r = words[index]
                        # r = self.complete_str(words, index, child_dict_list, 'ADV')
                        e1 = self.complete_e(words, postags, child_dict_list, child_dict['SBV'][0])
                        e2 = self.complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
                        svos.append([e1, r, e2])

                    # 定语后置，动宾关系
                    relation = arcs[index][0]
                    head = arcs[index][2]
                    if relation == 'ATT':  # 与父节点是定中关系
                        if 'VOB' in child_dict:  # 与子节点是动宾关系
                            e1 = self.complete_e(words, postags, child_dict_list, head - 1)
                            r = words[index]
                            e2 = self.complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
                            temp_string = "{}{}".format(r, e2)
                            if temp_string == e1[:len(temp_string)]:
                                e1 = e1[len(temp_string):]
                            if temp_string not in e1:
                                svos.append([e1, r, e2])

                    # 含有介宾关系的主谓动补关系
                    if 'SBV' in child_dict and 'CMP' in child_dict:
                        e1 = self.complete_e(words, postags, child_dict_list, child_dict['SBV'][0])
                        cmp_index = child_dict['CMP'][0]
                        r = "{}{}".format(words[index], words[cmp_index])
                        if 'POB' in child_dict_list[cmp_index]:
                            e2 = self.complete_e(words, postags, child_dict_list, child_dict_list[cmp_index]['POB'][0])
                            svos.append([e1, r, e2])
        return svos

    def complete_str(self, words, index, child_dict_list, str):
        '''
        对传入的词表根据句法依存关系进行合并补充
        :param words:词表
        :param index:索引列表
        :param child_dict_list:依存子树列表
        :param str:依存关系
        :return:补充后的字符串
        '''
        child_dict = child_dict_list[index]
        pre_word = ''
        if str == 'ADV':
            if 'ADV' in child_dict.keys():
                ADV_index = child_dict['ADV']
                for i in ADV_index:
                    a = self.complete_str(words, i, child_dict_list, 'ATT')
                    b = self.complete_str(words, i, child_dict_list, 'POB')
                    pre_word = "{}{}{}".format(pre_word, a, words[i], b)

            return "{}{}".format(pre_word, words[index])

        elif str == 'ATT':
            if 'ATT' in child_dict.keys():
                for i in range(len(child_dict['ATT'])):
                    pre_word = "{}{}{}".format(pre_word,
                                               self.complete_str(words, child_dict['ATT'][i], child_dict_list, 'ATT'),
                                               words[child_dict['ATT'][i]])
                return pre_word
            else:
                return ''

        elif str == 'POB':
            if 'POB' in child_dict.keys():
                for i in range(len(child_dict['POB'])):
                    sub = ''.join(words[index + 1:child_dict['POB'][i] + 1])
                    pre_word = "{}{}".format(sub, self.complete_str(
                        words, child_dict['POB'][i], child_dict_list, 'POB'))
                return pre_word
            else:
                return ''

    def complete_e(self, words, postags, child_dict_list, word_index):
        '''
        对找出的主语或者宾语进行扩展
        :param words: 词列表
        :param postags: 词性列表
        :param child_dict_list: 依存子树列表
        :param word_index: 词下标
        :return: 完善后的主语字符串
        '''
        child_dict = child_dict_list[word_index]
        prefix = ''
        if 'ATT' in child_dict:
            for i in range(len(child_dict['ATT'])):
                prefix = "{}{}".format(prefix, self.complete_e(words, postags, child_dict_list, child_dict['ATT'][i]))
        postfix = ''
        if postags[word_index] == 'v':
            if 'VOB' in child_dict:
                postfix = "{}{}".format(postfix, self.complete_e(words, postags, child_dict_list, child_dict['VOB'][0]))
            if 'SBV' in child_dict:
                prefix = self.complete_e(words, postags, child_dict_list, child_dict['SBV'][0]) + prefix

        return "{}{}{}".format(prefix, words[word_index], postfix)

    def triples_main(self, content):
        '''
        三元组抽取主控程序
        :param content: 文章字符串
        :return: 三元组列表
        '''

        sentences = self.split_sents(content)
        stopwords = ['', ' ', ',', '，', '"', '“', '”']
        svos = []
        for sentence in sentences:
            words, postags, child_dict_list, roles_dict, arcs = self.parser.parser_main(
                sentence)  # arcs是format_parse_list
            svo = self.ruler2(words, postags, child_dict_list, arcs, roles_dict)
            temp = [item for item in svo if item[0] not in stopwords and item[1]
                    not in stopwords and item[2] not in stopwords and len(item[2]) >= 2]
            svos.extend(temp)

        triples = []
        for item in svos:
            if item not in triples:
                triples.append(item)
        return triples


# 创建模型，将模型加载到内存中
extractor = TripleExtractor()

def get_triples(content):
    '''
    传入文章字符串，抽取文章中的主谓宾三元组
    :param content: 文章字符串
    :return: 三元组列表
    '''
    # 清洗过后的文本内容--str类型，返回各个句子三元组列表[['美国', '攻打', '伊拉克']]
    svos = extractor.triples_main(content)

    return svos


"""测试函数"""
if __name__ == '__main__':
    sentence = "自己进行的又一次极端挑衅行为。"
    svos = get_triples(sentence)
    print(svos)
