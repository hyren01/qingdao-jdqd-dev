# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:09:08 2019

@author: 12894
"""

# -*- coding: utf-8 -*-
import re

class pattern_but():
    def __init__(self):
        pass

    def ruler1(self, sentence):
        datas = list()
        word_pairs =[[['与其'], ['不如']],
                [['虽然','尽管','虽'],['但也','但还','但却','但']],
                [['虽然','尽管','虽'],[ '但','但是也','但是还','但是却',]],
                [['不是'],['而是']],
                [['即使','就算是'],['也','还']],
                [['即便'],['也','还']],
                [['虽然','即使'],['但是','可是','然而','仍然','还是','也', '但']],
                [['虽然','尽管','固然'],['也','还','却']],
                [['与其','宁可'],['决不','也不','也要']],
                [['与其','宁肯'],['决不','也要','也不']],
                [['与其','宁愿'],['也不','决不','也要']],
                [['虽然','尽管','固然'],['也','还','却']],
                [['不管','不论','无论','即使'],['都', '也', '总', '始终', '一直']],
                [['虽'],['可是','倒','但','可','却','还是','但是']],
                [['虽然','纵然','即使'],['倒','还是','但是','但','可是','可','却']],
                [['虽说'],['还是','但','但是','可是','可','却']],
                [['无论'],['都','也','还','仍然','总','始终','一直']],
                [['与其'],['宁可','不如','宁肯','宁愿']]]

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
                data['type'] = 'but'
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
