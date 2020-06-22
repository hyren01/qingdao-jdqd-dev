import math
import jieba.posseg as pesg
import pandas as pd
from textrank4zh import TextRank4Keyword, TextRank4Sentence
import json

# 测试文本

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

#对输入的长文本新闻内容进行语义分析，提取重要句子
def get_test(news):
    tr4s = TextRank4Sentence()
    test_news = []
    tr4s.analyze(text=news, lower=True, source='all_filters')
    for item in tr4s.get_key_sentences(num=1):
        sentence = item.sentence
    tr4w = TextRank4Keyword()
    tr4w.analyze(text=sentence, lower=True, window=2)
    for item in tr4w.get_keywords(10, word_min_len=1):
        test_news.append((item.word))
    return test_news


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
    word_all = []
    for sent in event_all:
        wor = []
        words = pesg.cut(sent)
        for word, flag in words:
            wor.append(word)
        word_all.append(wor)
    return word_all,event_all


if __name__ == '__main__':
    count=0
    f = open('./input/news.txt', 'r', encoding='utf-8')
    news_all = f.read().split('\n')
    event,event_sent=get_event()
    s = BM25(event)
    for news in news_all:
        count += 1
        result = []
        test_sentence = get_test(news)
        similarity=s.simall(test_sentence)
        simi_rank = sorted(enumerate(similarity), key=lambda item: -item[1])
        for i in simi_rank:
            score = {'id': i[0], 'score': i[1], 'event': event_sent[i[0]]}
            result.append(score)
        resultfile = './output/' + str(count) + '.json'
        fw = open(resultfile, 'w', encoding='utf-8')
        for sco in result:
            result_dict = json.dumps(sco, ensure_ascii=False)
            fw.write(result_dict)
            fw.write('\n')
    print("Prediction End!")
