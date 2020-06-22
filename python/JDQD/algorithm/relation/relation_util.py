import re


def split_sentence(sentence, concat_separate_conj=True):
    """
    将句子拆分成子句. 如果有连词单独作为子句, 则将连词与相关子句合并
    :param concat_separate_conj:
    :param sentence:
    :return:
    """
    sentence = sentence.replace(' ', '')
    conjunctions = ['从而', '所以', '为此', '致使', '导致', '使得', '因此', '因而',
                    '故而', '从而', '以致于', '以致', '于是', '那么', '因为', '由于',
                    '缘于', '那么', '但是', '但', '即']
    min_sentences_and_delimiters = re.split(r'([。.，,？?！!；;：:]|\.{3,10}|…{2,5})', sentence)
    if len(min_sentences_and_delimiters) == 1:
        return min_sentences_and_delimiters, ['']
    min_sentences_and_delimiters.append('')
    min_sentences = []
    delimiters = []
    last_min_sentences = []
    num_split = len(min_sentences_and_delimiters)
    for i in range(num_split // 2):
        min_sentence_index = i * 2
        min_sentence = min_sentences_and_delimiters[min_sentence_index]
        if concat_separate_conj:
            if min_sentence in conjunctions:
                last_min_sentences.append(min_sentence)
                continue
            if last_min_sentences:
                min_sentence = ''.join(last_min_sentences) + min_sentence
                last_min_sentences = []
        delimiter = min_sentences_and_delimiters[min_sentence_index + 1]
        min_sentences.append(min_sentence)
        delimiters.append(delimiter)
    return min_sentences, delimiters


def slice_sentence(min_sentences, delimiters):
    """
    使用滑动窗口截取子句
    :param min_sentences:
    :param delimiters:
    :return:
    """
    num_min_sentences = len(min_sentences)
    sub_sentences = []
    for slice_window in range(1, num_min_sentences + 1):
        for start in range(num_min_sentences):
            end = start + slice_window
            if end > num_min_sentences:
                break
            sub_sentence_list = []
            for i in range(start, end):
                sub_sentence_list.append(min_sentences[i])
                sub_sentence_list.append(delimiters[i])
            sub_sentence = ''.join(sub_sentence_list)
            sub_sentences.append(sub_sentence)
    return sub_sentences


def extract_article(article):
    sentences = split_article_to_sentences(article)
    sents_article = []
    for sentence in sentences:
        min_sentences, delimiters = split_sentence(sentence)
        sub_sentences = slice_sentence(min_sentences, delimiters)
        sents_article.extend(sub_sentences)
    return sents_article


def split_article_to_sentences(article):
    """
    文章分句
    :param article:
    :return:
    """
    article = re.sub(r'([。！？?])([^”’])', r"\1\n\2", article)  # 单字符断句符
    article = re.sub(r'(\.{6})([^”’])', r"\1\n\2", article)  # 英文省略号
    article = re.sub(r'(…{2})([^”’])', r"\1\n\2", article)  # 中文省略号
    article = re.sub(r'([。！？?][”’])([^，。！？?])', r'\1\n\2', article)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    article = article.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    sentences = article.split("\n")
    return sentences


def add_char(words, chars):
    """
    为关键词列表添加逗号
    :param chars:
    :param words:
    :return:
    """
    new_words = []
    for char in chars:
        words_ = [w + char for w in words]
        new_words.extend(words_)
    new_words.extend(words)
    return new_words


def extract(full_sentence, sub_sentences, rules):
    for rule in rules:
        result = rule(full_sentence, sub_sentences)
        if result:
            return rule.__name__, result
    return '', {}


# TODO(zhxin): 切分排比句
# def split_if_branch(sentence):
#     if_branch_num = sentence.count('如果')
#     if if_branch_num < 2:
#         return sentence
#     else:


if __name__ == '__main__':
    # sentence = '麻生太郎表示:“只要朝鲜试射导弹、威胁到日本的国家安全，我们才将立刻要求联合国安理会此进行讨论。”'
    # sentence = '麻生太郎表示:“只要朝鲜试射导弹、威胁到日本的国家安全，我们才将立刻要求联合国安理会此进行讨论'
    sentence = '麻生太郎表示讨论.'
    min_sentences, delimiters = split_sentence(sentence)
    sub_sentences = slice_sentence(min_sentences, delimiters)
