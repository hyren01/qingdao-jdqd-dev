import math
import jieba.posseg as pesg
import pandas as pd
import json
import numpy as np
import re
import jieba

# json编码
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

# BM25算法
class BM25(object):
    def __init__(self, docs):
        self.D = len(docs)
        self.avgdl = sum([len(doc)+0.0 for doc in docs]) / self.D
        self.docs = docs
        self.f = []  # 列表的每一个元素是一个dict，dict存储着一个文档中每个词的出现次数
        self.df = {} # 存储每个词及出现了该词的文档数量
        self.idf = {} # 存储每个词的idf值
        self.k1 = 1.5
        self.b = 0.75
        self.init()
    def init(self):
        for doc in self.docs:
            tmp = {}
            for word in doc:
                tmp[word] = tmp.get(word, 0) + 1  # 存储每个文档中每个词的出现次数
            self.f.append(tmp)
            for k in tmp.keys():
                self.df[k] = self.df.get(k, 0) + 1
        for k, v in self.df.items():
            self.idf[k] = math.log(self.D-v+0.5)-math.log(v+0.5)

    def sim(self, doc, index):
        score = 0
        for word in doc:
            if word not in self.f[index]:
                continue
            d = len(self.docs[index])
            score += (self.idf[word]*self.f[index][word]*(self.k1+1)
                      / (self.f[index][word]+self.k1*(1-self.b+self.b*d
                                                      / self.avgdl)))
        return score

    def simall(self, doc):
        scores = []
        for index in range(self.D):
            score = self.sim(doc, index)
            scores.append(score)
        return scores

#读入excel表格中的三元组事件
def get_event():
    event_all = []
    df = pd.read_excel(r'./input/互联网事件识别.xlsx')
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
    word_all = []
    for sent in event_all:
        wor = []
        words = pesg.cut(sent)
        for word, flag in words:
            wor.append(word)
        word_all.append(wor)
    return word_all,event_all

#划分新闻为一个个句子
def split_sents(content):
    return [sentence for sentence in re.split(r'[？?！!。]', content) if sentence]

def main(news_all):
    news = split_sents(news_all)  # 输入一个新闻，并按照句子划分。一段新闻得到多个句子
    event, event_sent = get_event()
    s = BM25(event)  # 用现有的三元组事件建立语料库
    all_sim = []
    for test_sentence in news:  # 对每个句子计算与100多个事件的相似度
        result = []
        test_sentences = [word for word in jieba.cut(test_sentence)]
        similarity = s.simall(test_sentences)  # 得到一个句子与100多个事件的相似度
        all_sim.append(similarity)  # 得到每个新闻所有句子分别与100多个事件的相似度
    s = np.array(all_sim)
    # 把新闻中对应句子相似度值最大的提取作为衡量新闻与事件的相似度
    mazz = []
    for i in range(len(s[0])):
        a = s[:, i]
        maxz = max(a)
        mazz.append(maxz)
    id = s.argmax(axis=0)  # 提取新闻中相似度最大的句子
    si_rank = sorted(enumerate(mazz), key=lambda item: -item[1])  # 从大到小排序，得到新闻与事件的相似度
    result = []
    count = -1
    for i in si_rank:
        count += 1
        out = {"id": i[0] + 1, "score": i[1], "news": news[id[count]]}
        result.append(out)
    resultfile = './anti_bm25_out/' + 'bm.json'
    fw = open(resultfile, 'w', encoding='utf-8')
    for sco in result:
        result_dict = json.dumps(sco, ensure_ascii=False, cls=MyEncoder)
        fw.write(result_dict)
        fw.write('\n')

if __name__ == '__main__':
    f = open('./input/content.txt', 'r', encoding='utf-8')
    news_all = f.read()
    main(news_all)

