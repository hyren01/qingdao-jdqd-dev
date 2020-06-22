# coding=utf-8
from sentence_parser import *
import re
#from event_graph import *
import jieba.posseg as pseg

class TripleExtractor:
    def __init__(self):
        self.parser = LtpParser()

    '''文章分句处理, 切分长句，冒号，分号，感叹号等做切分标识'''

    def split_sents(self, content):
        return [sentence for sentence in re.split(r'[？?！!。；;：:\n\r]', content) if sentence]

    '''利用语义角色标注,直接获取主谓宾三元组,基于A0,A1,A2'''

    def ruler1(self, words, postags, roles_dict, role_index):
        v = words[role_index].strip()
        role_info = roles_dict[role_index]

        if 'A0' in role_info.keys() and 'A1' in role_info.keys():
            s = ''.join([words[word_index] for word_index in range(role_info['A0'][1], role_info['A0'][2]+1) if
                         postags[word_index][0] not in ['w', 'u', 'x'] and words[word_index]]).strip()
            o = ''.join([words[word_index] for word_index in range(role_info['A1'][1], role_info['A1'][2]+1) if
                         postags[word_index][0] not in ['w', 'u', 'x'] and words[word_index]]).strip()
            if s and o:
                return '0', [s, v, o]
        return '4', []

    '''三元组抽取主函数'''
    
    def ruler2(self, words, postags, child_dict_list, arcs, roles_dict):
        svos = []
        for index in range(len(postags)):
            tmp = 1
            # 先借助语义角色标注的结果，进行三元组抽取
            if index in roles_dict:
                flag, triple = self.ruler1(words, postags, roles_dict, index)
                if flag == '0':
                    # triple[1] = self.complete_str(
                    #     words, index, child_dict_list, 'ADV')
                    triple.append(self.complete_str(
                        words, index, child_dict_list, 'ADV').strip())
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
                        r = words[index].strip()
                        # r = self.complete_str(
                        #     words, index, child_dict_list, 'ADV')
                        e1 = self.complete_e(
                            words, postags, child_dict_list, child_dict['SBV'][0]).strip()
                        e2 = self.complete_e(
                            words, postags, child_dict_list, child_dict['VOB'][0]).strip()
                        r_p = self.complete_str(
                            words, index, child_dict_list, 'ADV').strip()
                        svos.append([e1, r, e2, r_p])

                    # 定语后置，动宾关系
                    relation = arcs[index][0]
                    head = arcs[index][2]
                    if relation == 'ATT':  # 与父节点是定中关系
                        if 'VOB' in child_dict:  # 与子节点是动宾关系
                            e1 = self.complete_e(
                                words, postags, child_dict_list, head - 1).strip()
                            r = words[index]
                            e2 = self.complete_e(
                                words, postags, child_dict_list, child_dict['VOB'][0]).strip()
                            temp_string = r + e2
                            if temp_string == e1[:len(temp_string)]:
                                e1 = e1[len(temp_string):]
                            if temp_string not in e1:
                                svos.append([e1, r, e2, self.complete_str(
                                    words, index, child_dict_list, 'ADV').strip()])

                    # 含有介宾关系的主谓动补关系
                    if 'SBV' in child_dict and 'CMP' in child_dict:
                        e1 = self.complete_e(
                            words, postags, child_dict_list, child_dict['SBV'][0])
                        cmp_index = child_dict['CMP'][0]
                        r = words[index] + words[cmp_index]
                        if 'POB' in child_dict_list[cmp_index]:
                            e2 = self.complete_e(
                                words, postags, child_dict_list, child_dict_list[cmp_index]['POB'][0])
                            svos.append([e1, r, e2, self.complete_str(
                                words, index, child_dict_list, 'ADV').strip()])
        return svos

    '''寻找谓语动词前面的程度副词'''

    def complete_str(self, words, index, child_dict_list, str):
        child_dict = child_dict_list[index]
        pre_word = ''
        # post_word = ''
        if str == 'ADV':
            if 'ADV' in child_dict.keys():
                ADV_index = child_dict['ADV']
                for i in ADV_index:
                    pre_word += self.complete_str(words, i, child_dict_list, 'ATT') + \
                        words[i] + \
                        self.complete_str(words, i, child_dict_list, 'POB')
            return pre_word + words[index]

        elif str == 'ATT':
            if 'ATT' in child_dict.keys():
                for i in range(len(child_dict['ATT'])):
                    pre_word += self.complete_str(
                        words, child_dict['ATT'][i], child_dict_list, 'ATT') + words[child_dict['ATT'][i]]

                return pre_word
            else:
                return ''

        elif str == 'POB':
            if 'POB' in child_dict.keys():
                for i in range(len(child_dict['POB'])):
                    sub = ''.join(words[index+1:child_dict['POB'][i]+1])
                    pre_word = sub + self.complete_str(
                        words, child_dict['POB'][i], child_dict_list, 'POB')
                return pre_word
            else:
                return ''

    '''对找出的主语或者宾语进行扩展'''

    def complete_e(self, words, postags, child_dict_list, word_index):
        child_dict = child_dict_list[word_index]
        prefix = ''
        if 'ATT' in child_dict:
            for i in range(len(child_dict['ATT'])):
                prefix += self.complete_e(words, postags,
                                          child_dict_list, child_dict['ATT'][i])
        postfix = ''
        if postags[word_index] == 'v':
            if 'VOB' in child_dict:
                postfix += self.complete_e(words, postags,
                                           child_dict_list, child_dict['VOB'][0])
            if 'SBV' in child_dict:
                prefix = self.complete_e(
                    words, postags, child_dict_list, child_dict['SBV'][0]) + prefix

        return prefix + words[word_index] + postfix

    '''如果三元组第一个元素有代词,则将其替换成前面一个三元组的第三个元素'''

    def modify(self, svos):
        if len(svos) == 1:
            return svos
        else:
            i = 1
            for item in svos[1:]:
                word_pos = ''
                pos = pseg.cut(item[0])
                for item in pos:
                    word_pos += str(item)
                if 'r' in word_pos:
                    svos[i][0] = svos[i - 1][2]
                i += 1
            return svos

    '''程序主控函数'''

    def triples_main(self, content):
        sentences = self.split_sents(content)
        stopwords = ['', ' ', ',', '，', '"', '“', '”',
                     '<', '>', '《', '》', '<', '>', '(', ')', '、']
        svos = []
        r_ps = []
        svos_pre = []
        for sentence in sentences:
            words, postags, child_dict_list, roles_dict, arcs = self.parser.parser_main(
                sentence)  # arcs是format_parse_list
            svo = self.ruler2(
                words, postags, child_dict_list, arcs, roles_dict)
            for item in svo:
                if item[0] not in stopwords and item[1] not in stopwords\
                        and item[2] not in stopwords \
                        and len(item[2]) <= 15 and item[0] != item[2]:
                    svos += [item[:3]]
                    r_ps += [item[-1]]
                    svos_pre += [[item[0], item[-1], item[2]]]
        svos = self.modify(svos)

        return svos, r_ps, svos_pre

'''测试'''
if __name__ == "__main__":
    def hello():
        with open('./content.txt',encoding='utf-8') as f:
            content = f.read()
        extractor = TripleExtractor()
        
        svos, r_ps, svos_pre = extractor.triples_main(content)
        with open('./triple.txt', 'w') as f1:
            for svo in svos:
                f1.write(str(svo))
                f1.write('\n')
        with open('./r_ps.txt', 'w') as f2:
            for r_p in r_ps:
                f2.write(str(r_p))
                f2.write('\n')
        with open('./summary.txt', 'w') as f2:
            for r_p in svos_pre:
                f2.write(str(svos_pre))
                f2.write('\n')
    hello()
    nodes = []
    edges = []
    with open('./triple.txt') as f:
        for line in f:
            triple = line.strip()
            triple = re.sub(r'\'', '', triple)
            triple = ''.join(triple.split(' '))[1:-1]
            triple = triple.split(',')
            nodes.append(triple[0])
            nodes.append(triple[2])
            edges.append(triple)
    handler = CreatePage()
    data_nodes, data_edges = handler.collect_data(nodes, edges)
    handler.create_html(data_nodes, data_edges)
