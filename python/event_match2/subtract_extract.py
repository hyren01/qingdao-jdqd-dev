import os
import re


from sklearn.feature_extraction.text import CountVectorizer
import operator
import jieba
from textrank4zh import TextRank4Keyword, TextRank4Sentence


def valid_file(file_path):
    if os.path.exists(file_path) or os.path.isfile(file_path):
        return True
    else:
        print('file {} does not exist'.format(os.path.split(file_path)[-1]))
        return False

def valid_dir(dir_path):
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        return True
    elif not os.path.exists(dir_path):
        os.makedirs(dir_path)
        return True
    elif not os.path.isdir(dir_path):
        return False


def file_read(file_path):
# 读取文件内容
    try:
        with open(file_path, 'r') as file:
            content = file.read().strip().strip('\r\n\t').replace('\t','').replace(u'\u3000','').replace(u'\xa0','').replace(' ','')
            content = re.sub('[？?…！!；;]','。',content)
            content = re.sub('<.*>?', '', content)
        return content
    except Exception:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip().strip('\r\n\t').replace('\t','').replace(u'\u3000','').replace(u'\xa0','').replace(' ','')
            content = re.sub('[？?…！!；;]','。',content)
            content = re.sub('<.*>?','',content)
        return content


# def hanlp_subtract(content):
#
#     summary_sentences = HanLP.extractSummary(content, 3)
#     return summary_sentences


def mmr_subtract(content):

    def encode_sen(sen, corpus):
        """
        input: sentence and corpus
        output :  bag of words vector of sentence
        """
        cv = CountVectorizer()
        cv = cv.fit(corpus)
        vec = cv.transform([sen]).toarray()
        return vec[0]

    def cosin_distance(vector1, vector2):
        """
        input: two bag of words vectors of sentence
        output :  the similarity between the sentence

        """
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
        """
        transform the doc_list to str
        """
        docu_str = ""
        for wordlist in doc_list:
            docu_str += " ".join(wordlist)
        return docu_str

    def MMR(doc_list, corpus):
        """
        input ：corpus and the docment you want to extract
        output :the abstract of the docment
        """
        Corpus = corpus
        docu = doc_list2str(doc_list)
        doc_vec = encode_sen(docu, Corpus)
        QDScore = {}
        #calculate the  similarity of every sentence with the whole corpus
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


    sen_list = content.strip().replace('\n','').split("。")
    doc_list = [jieba.lcut(i) for i in sen_list if i]
    corpus = [" ".join(i) for i in doc_list if i]
    summary_sentences = MMR(doc_list, corpus)
    return summary_sentences


def text_rank_subtract(content):

    text = content
    tr4s = TextRank4Sentence()
    tr4s.analyze(text=text, lower=True, source='all_filters')
    summary_sentences = [item.sentence for item in tr4s.get_key_sentences(num=3)]
    return summary_sentences


def get_subtract(content):
    '''
    使用text rank 和mmr方法提取摘要，并返回摘要语句列表
    :param content:
    :return:
    '''
    content = re.sub('<.*?>', '', content)
    content = re.sub('【.*?】','', content)
    # 剔除邮箱
    content = re.sub('/^([a-z0-9_\.-]+)@([\da-z\.-]+)\.([a-z\.]{2,6})$/','',content)
    content = re.sub('/^[a-z\d]+(\.[a-z\d]+)*@([\da-z](-[\da-z])?)+(\.{1,2}[a-z]+)+$/','',content)
    # 剔除URL
    content = re.sub('	/^<([a-z]+)([^<]+)*(?:>(.*)<\/\1>|\s+\/>)$/','',content)
    # 剔除16进制值
    content = re.sub('	/^#?([a-f0-9]{6}|[a-f0-9]{3})$/','',content)
    # 剔除IP地址
    content = re.sub('/((2[0-4]\d|25[0-5]|[01]?\d\d?)\.){3}(2[0-4]\d|25[0-5]|[01]?\d\d?)/','',content)
    content = re.sub('/^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$/','',content)
    # 剔除用户名密码名
    content = re.sub('/^[a-z0-9_-]{3,16}$/','',content)
    content = re.sub('	/^[a-z0-9_-]{6,18}$/','',content)
    # 剔除HTML标签
    content = re.sub('/^<([a-z]+)([^<]+)*(?:>(.*)<\/\1>|\s+\/>)$/','',content)
    content = content.strip().strip('\r\n\t').replace(u'\u3000', '').replace(u'\xa0', '')
    content = content.replace('\t', '').replace(' ','').replace('\n', '').replace('\r', '')
    # print(content)
    summary_sentences = []
    summary_sentences += mmr_subtract(content)
    summary_sentences +=text_rank_subtract(content)
    summary_sentences = [once.replace(' ','') for once in summary_sentences if once]
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
        summary_sentences = get_subtract(content)

        for once in summary_sentences:
            print(once)

