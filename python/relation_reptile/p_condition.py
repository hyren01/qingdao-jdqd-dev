# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:50:47 2019

@author: 12894
"""

import re

class pattern_condition():
    def __init__(self):
        pass

    def ruler1(self, sentence):
        datas = list()
        word_pairs =[
                [['除非'], ['否则', '才', '不然', '要不']],
                [['除非'], ['否则的话']],
                [['还是', '无论', '不管'], ['还是', '都', '总']],
                [['既然'], ['又', '且', '也', '亦']],
                [['假如'], ['那么', '就', '也', '还']],
                [['假若', '如果'], ['那么', '就', '那', '则', '便']],
                [['假使', '如果'], ['那么', '就', '那', '则', '便']],
                [['尽管', '如果'], ['那么', '就', '那', '则', '便']],
                [['即使', '就是'], ['也', '还是']],
                [['如果', '既然'], ['那么']],
                [['如', '假设'], ['则', '那么', '就', '那']],
                [['如果', '假设'], ['那么', '则', '就', '那']],
                [['万一'], ['那么', '就']],
                [['要是', '如果'], ['就', '那']],
                [['要是', '如果', '假如'], ['那么', '就', '那', '的话']],
                [['一旦'], ['就']],
                [['既然', '假如', '既', '如果'], ['则','就']],
                [['只要'], ['就', '便', '都', '总']],
                [['只有'], ['才', '还']],

        ]

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
                data['type'] = 'condition'
                datas.append(data)
        if datas:
            return datas[0]
        else:
            return {}
        
    '''抽取主函数'''
    def extract_triples(self, sentence):
        infos = list()
      #  print(sentence)
        if self.ruler1(sentence):
            infos.append(self.ruler1(sentence))
        return infos
