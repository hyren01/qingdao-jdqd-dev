# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:38:43 2019

@author: 12894
"""

import re

class pattern_more():
    def __init__(self):
        pass

    def ruler1(self, sentence):
        datas = list()
        word_pairs =[
                [['不但', '不仅'], ['并且']],
                [['不但'], ['而且', '并且', '也', '还']],
                [['不但'], ['而且', '并且', '也', '还']],
                [['不管'], ['都', '也', '总', '始终', '一直']],
                [['不光'], ['而且', '并且', '也', '还']],
                [['虽然', '尽管'], ['不过'], 'more'],
                [['不仅'], ['还', '而且', '并且', '也']],
                [['不论'], ['还是', '也', '总', '都', '始终', '一直']],
                [['不只'], ['而且', '也', '并且', '还']],
                [['不但', '不仅', '不光', '不只'], ['而且']],
                [['尚且', '都', '也', '又', '更'], ['还', '又']],
                [['既然', '既',], ['就', '便', '那', '那么', '也', '还']],
                [['无论', '不管', '不论', '或'], ['或']],
                [['或是'], ['或是']],
                [['或者', '无论', '不管', '不论'], ['或者']],
                [['不是'], ['也']],
                [['要么', '或者'], ['要么', '或者']],
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
                data['type'] = 'more'
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

