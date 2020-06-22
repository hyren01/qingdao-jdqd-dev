from itertools import product
import relation_util as ru
import pattern

base_words = ['不仅仅', '不仅', '不光', '不但', '固然']
also_words = ['而且', '并且', '又', '还', '更是', '但更']
keywords = list(product(base_words, also_words))
keywords_also = [[w] for w in also_words]

parts = ['tag', 'source', 'target']
# parts = ['tag', 'base', 'more']


def rule101(sentence, sub_sentences):
    # 匹配模式: ...不仅..., ...还...
    return pattern.rule_skscsks(sentence, sub_sentences, keywords, parts)


def rule102(sentence, sub_sentences):
    # 匹配模式: ...不仅..., 而且...
    return pattern.rule_skscks(sentence, sub_sentences, keywords, parts)


def rule103(sentence, sub_sentences):
    # 匹配模式: 不仅..., ...而且...
    return pattern.rule_kscsks(sentence, sub_sentences, keywords, parts)


def rule104(sentence, sub_sentences):
    # 匹配模式: 不仅..., 而且...
    return pattern.rule_kscks(sentence, sub_sentences, keywords, parts)


def rule201(sentence, sub_sentences):
    # 匹配模式: ..., ...而且...
    keywords = [[w] for w in also_words]
    return pattern.rule_scsks(sentence, sub_sentences, keywords, parts, period=False)


def rule202(sentence, sub_sentences):
    # 匹配模式: ..., 而且...
    return pattern.rule_scks(sentence, sub_sentences, keywords_also, parts, period=False)


def rule301(sentence, sub_sentences):
    # 匹配模式: ..., 而不仅仅...
    keywords = ['而' + w for w in base_words]
    keywords = keywords + base_words
    keywords = [[w] for w in keywords]
    return pattern.rule_scks(sentence, sub_sentences, keywords, parts, reverse=True, period=False)


def extract(full_sentence, sub_sentences):
    rules = [rule101, rule102, rule103, rule104, rule201, rule202, rule301]
    return ru.extract(full_sentence, sub_sentences, rules)
