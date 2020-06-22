import math
import jieba
import jieba.posseg as pesg
import pandas as pd
import json

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

#读入停用词
def split_stopword(stopfile):
    with open(stopfile, 'r', encoding='utf-8') as f:
        stopwords=[]
        word=f.read().split('\n')
        for i in word:
            stopwords.append(i)
    return stopwords

#对所有爬虫得到的新闻文章进行分词、去停用词和相关词性的无意义词处理
def get_all(news_path):
    f = open(news_path, 'r', encoding='utf-8')
    sents = f.read().split('\n')
    doc_all = []
    stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']
    stopwords = split_stopword('./input/stop.txt')
    for sent in sents:
        doc = []
        words = pesg.cut(sent)
        for word, flag in words:
            if flag not in stop_flag and word not in stopwords:
                doc.append(word)
        doc_all.append(doc)
    return doc_all

#读入excel表格中的三元组事件
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
    return(event_all)


if __name__ == '__main__':
    all_sim=[]
    news_all=get_all('./input/content.txt')
    f = open('./input/content.txt', 'r', encoding='utf-8')
    news = f.read().split('\n')
    s = BM25(news_all)
    events=get_event()
    count=0
    for event in events:
        count+=1
        result=[]
        test_sentence = [word for word in jieba.cut(event)]
        similarity=s.simall(test_sentence)
        simi_rank = sorted(enumerate(similarity), key=lambda item: -item[1])
        for i in simi_rank:
            score={'id':i[0],'score':i[1],'news':news[i[0]]}
            result.append(score)
        resultfile = './output/' + str(count) + '.json'
        fw = open(resultfile, 'w', encoding='utf-8')
        for sco in result:
            result_dict=json.dumps(sco,ensure_ascii=False)
            fw.write(result_dict)
            fw.write('\n')
    print("End!")
