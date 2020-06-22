# coding: utf-8
# 抽取文章摘要
import os

from sklearn.feature_extraction.text import CountVectorizer
import operator
import jieba
from textrank4zh import TextRank4Keyword, TextRank4Sentence


def mmr_subtract(content):
    '''
    传入文章内容，使用mmr方式抽取文章摘要
    :param content: 文章内容
    :return: 抽取到的摘要语句列表
    '''

    def encode_sen(sen, corpus):
        '''
        对传入的句子进行向量化
        :param sen: 传入的句子
        :param corpus: 传入的语料
        :return: 词袋法向量化后的句子
        '''
        cv = CountVectorizer()
        cv = cv.fit(corpus)
        vec = cv.transform([sen]).toarray()
        return vec[0]

    def cosin_distance(vector1, vector2):
        '''
        计算两个向量之间的夹角余弦值
        :param vector1: 向量1
        :param vector2: 向量2
        :return: 夹角余弦值--相似度
        '''
        dot_product = 0.0
        normA = 0.0
        normB = 0.0
        for a, b in zip(vector1, vector2):
            dot_product += a * b
            normA += a ** 2
            normB += b ** 2
        if normA == 0.0 or normB == 0.0:
            return 0
        else:
            return dot_product / ((normA * normB) ** 0.5)

    def doc_list2str(doc_list):
        '''
        将分词后的句子列表拼接成字符串
        :param doc_list: 分词后的句子列表
        :return: 字符串
        '''
        docu_str = ""
        for wordlist in doc_list:
            docu_str += " ".join(wordlist)
        return docu_str

    def MMR(doc_list, corpus):
        '''
        传入分词后的句子列表和使用" "分割的句子列表
        :param doc_list: 分词后的句子列表
        :param corpus: 间隔后的句子列表
        :return: 摘要句子列表
        '''
        Corpus = corpus
        docu = doc_list2str(doc_list)
        # 将文章向量化
        doc_vec = encode_sen(docu, Corpus)
        # 句子与文章相似度字典
        QDScore = {}
        # 计算句子与文章的相似度
        for sen in doc_list:
            sen = " ".join(sen)

            sen_vec = encode_sen(sen, corpus)
            score = cosin_distance(sen_vec, doc_vec)
            QDScore[sen] = score

        n = 2
        alpha = 0.7
        Summary_set = []
        while n > 0:
            MMRScore = {}
            # select the first sentence of abstract
            if Summary_set == []:
                selected = max(QDScore.items(), key=operator.itemgetter(1))[0]
                Summary_set.append(selected)

            Summary_set_str = " ".join(Summary_set)

            for sentence in QDScore.keys():
                # calculate MMR
                if sentence not in Summary_set:
                    sum_vec = encode_sen(Summary_set_str, corpus)
                    sentence_vec = encode_sen(sentence, corpus)
                    MMRScore[sentence] = alpha * QDScore[sentence] - (1 - alpha) * cosin_distance(sentence_vec, sum_vec)
            if MMRScore.items():
                selected = max(MMRScore.items(), key=operator.itemgetter(1))[0]
                Summary_set.append(selected)
                n -= 1
            else:
                n-=1
        return Summary_set

    # 将文章分割成句子
    sen_list = content.strip().replace('\n','').split("。")
    # 分词后的句子列表
    doc_list = [jieba.lcut(i) for i in sen_list if i]
    # 使用" "将文章各个词分割开
    corpus = [" ".join(i) for i in doc_list if i]
    # 传入分词后的句子列表以及分词后的文章句子列表
    summary_sentences = MMR(doc_list, corpus)
    return summary_sentences


def text_rank_subtract(content):
    '''
    对传入的中文字符串使用text_rank方法抽取摘要
    :param content: 中文字符串
    :return: 摘要语句列表
    '''

    text = content
    tr4s = TextRank4Sentence()
    tr4s.analyze(text=text, lower=True, source='all_filters')
    summary_sentences = [item.sentence for item in tr4s.get_key_sentences(num=3)]
    return summary_sentences


def get_abstract(content):

    '''
    使用text rank 和mmr方法提取摘要，并返回摘要语句列表
    :param content: 清洗后的文本字符串--str
    :return: 摘要语句--list

    '''
    summary_sentences = []
    mmr_summary_sentences = mmr_subtract(content)
    text_rank_summary_sentences =text_rank_subtract(content)

    summary_sentences.extend(mmr_summary_sentences)
    summary_sentences.extend(text_rank_summary_sentences)

    # 将句子中的空格剔除
    summary_sentences = [once.replace(" ", "") for once in summary_sentences if once]
    summary_sentences = list(set(summary_sentences))

    return summary_sentences


if __name__  ==  '__main__':

    file_list = os.listdir('./article')
    for file in file_list:
        file_path = os.path.join('./article', file)
    # file_path = 'D:/work/final_program/article/媒体：1600名美国军人参加在日本的参谋部演习 - 俄罗斯卫星通讯社'
        try:
            with open(file_path, 'r') as file:
                content = file.read()
        except Exception:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
        summary_sentences = get_abstract(content)

        for once in summary_sentences:
            print(once)
