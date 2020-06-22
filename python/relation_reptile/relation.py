# -*- coding: utf-8 -*-
import pandas as pd
import re, jieba
import jieba.posseg as pseg
from pyltp import SentenceSplitter

from p_but import pattern_but
from p_condition import pattern_condition
from p_more import pattern_more
from p_reason import pattern_reason
from p_seq import pattern_seq

pattern_but_e = pattern_but()
pattern_condition_e = pattern_condition()
pattern_more_e = pattern_more()
pattern_reason_e = pattern_reason()
pattern_seq_e = pattern_seq()


'''抽取主控函数'''
def extract_main(content):
    sentences = process_content(content)
    datas = list()
    for sentence in sentences:
        sent = ' '.join([word.word + '/' + word.flag for word in pseg.cut(sentence)])
        if pattern_but_e.extract_triples(sent):
            res = pattern_but_e.extract_triples(sent)
        elif pattern_condition_e.extract_triples(sent):
            res = pattern_condition_e.extract_triples(sent)
        elif pattern_more_e.extract_triples(sent):
            res = pattern_more_e.extract_triples(sent)
        elif pattern_reason_e.extract_triples(sent):
            res = pattern_reason_e.extract_triples(sent)
        elif pattern_seq_e.extract_triples(sent):
            res = pattern_seq_e.extract_triples(sent)
        else:
            res = list()
        if res:
            for data in res:
                if data['tag'] and data['up'] and data['down']:
#                    可以选择抽取三元组提取出事件
#                    up_triple, _, _ = triple_extraction(''.join([word.split('/')[0] for word in data['up'].split(' ') if word.split('/')[0]]))
#                    down_triple, _,_ = triple_extraction(''.join([word.split('/')[0] for word in data['down'].split(' ') if word.split('/')[0]]))
#                    data['up'] = up_triple[0],data['down'] = down_triple[0]
                    data['up'] = ''.join([word.split('/')[0] for word in data['up'].split(' ') if word.split('/')[0]])
                    data['down'] = ''.join([word.split('/')[0] for word in data['down'].split(' ') if word.split('/')[0]])
                    datas.append(data)
    return datas

'''文章分句处理'''
def process_content(content):
    return [sentence for sentence in SentenceSplitter.split(content) if sentence]


'''测试'''
def test():
    with open('D:/myproject/relation/aiticle/you_need/anews3.txt', 'r', encoding = 'utf-8') as f:
        content = f.read()
        content = ''.join(content.split())    
    datas = extract_main(content)
    for data in datas:
        print('******'*4)
        print('up', data['up'])
        print('tag', data['tag'])
        print('down', data['down'])
        print('type', data['type'])

test()
