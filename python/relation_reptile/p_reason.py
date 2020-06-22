# -*- coding: utf-8 -*-
import re

class pattern_reason():
    def __init__(self):
        pass

    '''1由果溯因配套式'''
    def ruler1(self, sentence):
        datas = list()
        word_pairs =[['之所以', '是因为'], ['之所以', '由于'], ['之所以', '缘于']]
        for word in word_pairs:
            pattern = re.compile(r'([^？?！!。；;：:\n\r,，]*)({0})(.*)({1})([^？?！!。；;：:\n\r,，]*)'.format(word[0], word[1]))
            result = pattern.findall(sentence)
            data = dict()
            if result: 
                data['tag'] = result[0][1] + '-' + result[0][3]
                data['up'] = result[0][4]
                data['down'] = result[0][0] + result[0][2]
                data['type'] = 'reason'
                datas.append(data)
        if datas:
            return datas[0]
        else:
            return {}
        
    '''2由因到果配套式'''
    def ruler2(self, sentence):
        datas = list()
        word_pairs =[['因为', '从而'], ['因为', '为此'], ['既然?', '所以'],
                    ['因为', '为此'], ['由于', '为此'], ['除非', '才'],
                    ['只有', '才'], ['由于', '以至于?'], ['既然?', '却'],
                    ['如果', '那么'], ['如果', '则'], ['由于', '从而'],
                    ['既然?', '就'], ['既然?', '因此'], ['如果', '就'],
                    ['只要', '就'], ['因为', '所以'], ['由于', '于是'],
                    ['因为', '因此'], ['由于', '故'], ['因为', '以致于?'],
                    ['因为', '以致'], ['因为', '因而'], ['由于', '因此'],
                    ['因为', '于是'], ['由于', '致使'], ['因为', '致使'],
                    ['由于', '以致于?'], ['因为', '故'], ['因为?', '以至于?'],
                    ['由于', '所以'], ['因为', '故而'], ['由于', '因而']]

        for word in word_pairs:
            pattern = re.compile(r'([^？?！!。；;：:\n\r,，]*)({0})(.*)({1})([^？?！!。；;：:\n\r,，]*)'.format(word[0], word[1]))
            result = pattern.findall(sentence)
            data = dict()
            if result:
                data['tag'] = result[0][1] + '-' + result[0][3]
                data['up'] = result[0][0] + result[0][2]
                data['down'] = result[0][4]
                data['type'] = 'reason'
                datas.append(data)
        if datas:
            return datas[0]
        else:
            return {}
        
    '''3由因到果居中式明确'''
    def ruler3(self, sentence):
        pattern = re.compile(r'(.*)(于是|所以|致使|以致于?|因此|以至于?|从而|因而)(.*)')
        result = pattern.findall(sentence)
        data = dict()
        if result:
            data['tag'] = result[0][1]
            data['up'] = result[0][0]
            data['down'] = result[0][2]
            data['type'] = 'reason'
        return data
    
    '''4由因到果居中式精确'''
    def ruler4(self, sentence):
        pattern = re.compile(r'(.*)(牵动|已致|导致|使|促成|造成|造就|促使|酿成|引发|引起|诱导|引来|促发|引致|诱发|诱致|招致|致使|诱使)(.*)')
        result = pattern.findall(sentence)
        data = dict()
        if result:
            data['tag'] = result[0][1]
            data['up'] = result[0][0]
            data['down'] = result[0][2]
            data['type'] = 'reason'
        return data
    
    '''6由果溯因居中式模糊'''
    def ruler5(self, sentence):
        pattern = re.compile(r'(.*)(根源于|取决|来源于|取决于|缘于|源于|根源于|立足于)(.*)')
        result = pattern.findall(sentence)
        data = dict()
        if result:
            data['tag'] = result[0][1]
            data['up'] = result[0][2]
            data['down'] = result[0][0]
            data['type'] = 'reason'
        return data

    '''抽取主函数'''
    def extract_triples(self, sentence):
        infos = list()
      #  print(sentence)
        if self.ruler1(sentence):
            infos.append(self.ruler1(sentence))
        elif self.ruler2(sentence):
            infos.append(self.ruler2(sentence))
        elif self.ruler3(sentence):
            infos.append(self.ruler3(sentence))
        elif self.ruler4(sentence):
            infos.append(self.ruler4(sentence))
        elif self.ruler5(sentence):
            infos.append(self.ruler5(sentence))
        return infos
