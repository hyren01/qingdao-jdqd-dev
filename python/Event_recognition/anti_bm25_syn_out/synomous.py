import jieba.posseg as pesg
import pandas as pd
import synonyms

def split_stopword(stopfile):
    with open(stopfile, 'r', encoding='utf-8') as f:
        stopwords=[]
        word=f.read().split('\n')
        for i in word:
            stopwords.append(i)
    return stopwords

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
    doc_all = []
    for sent in event_all:
        doc = []
        words = pesg.cut(sent)
        for word, flag in words:
            doc.append(word)
        doc_all.append(doc)
    return doc_all

def tongyici():
    news_all=get_event()
    words_all=[]
    for news in news_all:
        words=[]
        for word in news:
            tongyi=synonyms.nearby(word)[0]
            if len(tongyi)>0:
                word=','.join(tongyi)
            else:
                word=word
            words.append(word)
        words_all.append(words)
    return words_all

def new_event():
    words_all=tongyici()
    sentence=[]
    for a in words_all:
        str = ''
        for i in a:
            count = 0
            ii = i.split(',')
            if len(ii) > 1:
                str += ii[0] + "("
                for j in ii[1:-2]:
                    str += j + ','
                    count += 1
                str = str + ii[-1] + ")"
            else:
                str += i
        sentence.append(str)
    return(sentence)


if __name__ == '__main__':
    sent=new_event()

