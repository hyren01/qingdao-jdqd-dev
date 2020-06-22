from itertools import product
import relation_util as ru
import pattern

is_words = ['属于', '是', '系']
category_words = ['一种', '一类', '一个']
keywords_is = [[''.join(wp)] for wp in list(product(is_words, category_words))]
contain_words = ['包括', '包含', '有']

# parts = ['tag', 'hyponym', 'hypernym']


def rule101(sentence, sub_sentences):
    # 匹配模式: ..., 是一种...
    return pattern.rule_scks(sentence, sub_sentences, keywords_is)


def rule102(sentence, sub_sentences):
    # 匹配模式: ...是一种...
    return pattern.rule_sks(sentence, sub_sentences, keywords_is)


def rule201(sentence, sub_sentences):
    # 匹配模式: 包括...在内的...
    keywords = list(product(contain_words, ['在内的']))
    return pattern.rule_kscks(sentence, sub_sentences, keywords, comma=False)


def rule3(sentence, sub_sentences):
    # 匹配模式: ..., 包括...等
    keywords = list(product(contain_words, ['等']))
    return pattern.rule_scksk(sentence, sub_sentences, keywords, reverse=True)


def extract(full_sentence, sub_sentences):
    rules = [rule101, rule102, rule201, rule3]
    return ru.extract(full_sentence, sub_sentences, rules)
