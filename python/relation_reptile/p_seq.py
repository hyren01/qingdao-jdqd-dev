# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:22:10 2019

@author: 12894
"""

import re

class pattern_seq():
    def __init__(self):
        pass

    def ruler1(self, sentence):
        datas = list()
        word_pairs =[
            [['又', '再', '才', '并'], ['进而']],
            [['首先', '第一'], ['其次', '然后']],
            [['首先', '先是'], ['再', '又', '还', '才']],
            [['一方面'], ['另一方面', '又', '也', '还']]]

        for word in word_pairs:
            pre = word[0]
            pos = word[1]
            pattern = re.compile(r'([^？?！!。；;：:\n\r,，]*)({0})(.*)({1})([^？?！!。；;：:\n\r,，]*)'.format('|'.join(pre), '|'.join(pos)))
            result = pattern.findall(sentence)
            data = dict()
            if result:
                data['tag'] = result[0][1] + '-' + result[0][3]
                data['up'] = result[0][0] + result[0][2]
                data['down'] = result[0][4]
                data['type'] = 'seq'
                datas.append(data)
        if datas:
            return datas[0]
        else:
            return {}
        
    def ruler2(self, sentence):
        pattern = re.compile(r'(.*)(其次|然后|接着|随后|接下来)(.*)')
        result = pattern.findall(sentence)
        data = dict()
        if result:
            data['tag'] = result[0][1]
            data['up'] = result[0][0]
            data['down'] = result[0][2]
            data['type'] = 'seq'
        return data
        
    '''抽取主函数'''
    def extract_triples(self, sentence):
        infos = list()
      #  print(sentence)
        if self.ruler1(sentence):
            infos.append(self.ruler1(sentence))
        elif self.ruler2(sentence):
            infos.append(self.ruler2(sentence))
        return infos
