import jieba
import jieba.posseg as pseg
from jieba import analyse
import numpy as np
import os
import pandas as pd
import re
import json

class TripleRecognition(object):

    def countIDF(self, text, topK):
        tfidf = analyse.extract_tags
        # 基于tfidf算法对新闻长文本抽取前topK个关键词
        keywords = tfidf(text, topK, withWeight=True)
        word = []
        for keyword in keywords:
            word.append(keyword[0])  # 得到前topk频的关键词
        return word

    def splitWords(self, str_a,str_b):
        word = self.countIDF(str_a,topK=10)
        cuta = ""
        seta = set()
        for key in word:
            cuta += key +" "
            seta.add(key) #给集合添加元素，如果添加的元素在集合中已存在，则不执行任何操作
        return [cuta, seta]

    def splitWords_test(self, str_a):
        wordsa = pseg.cut(str_a)
        cuta = ""
        seta = set()
        for key in wordsa:
            cuta += key.word + " "
            seta.add(key.word)
        return [cuta, seta]

    def JaccardSim(self, str_a, str_b):
        seta = self.splitWords(str_a,str_b)[1]
        setb = self.splitWords_test(str_b)[1]
        sa_sb = 1.0 * len(seta&setb) / len(seta | setb)#相似度计算
        return sa_sb

    def splitWordSimilarity(self, str_a, str_b):
        sim=self.JaccardSim(str_a, str_b)
        return sim

def get_event():
    event_all = []
    df = pd.read_excel(r'./input/cleaned_events.xlsx')
    subject = df['主语'].tolist()
    predicate = df['谓语'].tolist()
    object = df['宾语'].tolist()
    for i in range(len(subject)):
        spl = str(object[i])
        if spl == 'nan':
            event = str(subject[i]) + str(predicate[i])
            event_all.append(event)
        else:
            event = str(subject[i]) + str(predicate[i]) + str(object[i])
            event_all.append(event)
    # event_all=new_event()
    return event_all


def split_sents(content):
    a=[sentence for sentence in re.split(r'[？！。]', content) if sentence]
    return(a)

def test():
    handler = TripleRecognition()
    # test_sent = input('enter an sentence to search:').strip()
    event_all=get_event()
    f = open('./input/content.txt', 'r', encoding='utf-8')
    news = f.readlines()
    count=0
    for events in event_all:
        count+=1
        similarity = []
        result = []
        resultfile = './output/' + str(count) + '.json'
        fw = open(resultfile, 'w', encoding='utf-8')
        for i in news:
            sent_similarity=[]

            simi= handler.splitWordSimilarity(i,events)
            similarity.append(simi)
        simi_rank=sorted(enumerate(similarity), key=lambda item: -item[1])#按相似度从大到小进行排序
        for si in simi_rank:
            score = {'id': si[0], 'score': si[1], 'news': news[si[0]]}
            result.append(score)
        for sco in result:
            result_dict = json.dumps(sco, ensure_ascii=False)
            fw.write(result_dict)
            fw.write('\n')
if __name__ == '__main__':
    test()


