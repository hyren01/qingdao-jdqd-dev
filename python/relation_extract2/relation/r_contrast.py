from itertools import product
import relation_util as ru
import pattern

base_words = ['尽管', '虽然', '固然']
contrast_words = ['但是', '然而', '可是', '但', '还是', '却']

# parts = ['tag', 'base', 'contrast']


def rule101(sentence, sub_sentences):
    # 匹配模式: ...虽然..., ...但是...
    keywords = list(product(base_words, contrast_words))
    match_mode = 'long'
    return pattern.rule_skscsks(sentence, sub_sentences, keywords, match_mode)


def rule102(sentence, sub_sentences):
    # 匹配模式: ...虽然..., 但是...
    keywords = list(product(base_words, contrast_words))
    match_mode = 'long'
    return pattern.rule_skscks(sentence, sub_sentences, keywords, match_mode)


def rule103(sentence, sub_sentences):
    # 匹配模式: 虽然..., ...但是...
    keywords = list(product(base_words, contrast_words))
    match_mode = 'long'
    return pattern.rule_kscsks(sentence, sub_sentences, keywords, match_mode)


def rule104(sentence, sub_sentences):
    # 匹配模式: 虽然..., 但是...
    keywords = list(product(base_words, contrast_words))
    match_mode = 'long'
    return pattern.rule_kscks(sentence, sub_sentences, keywords, match_mode)


def rule201(sentence, sub_sentences):
    # 匹配模式: ..., ...但是...
    keywords = [[w] for w in contrast_words]
    return pattern.rule_scsks(sentence, sub_sentences, keywords, period=False)


def rule202(sentence, sub_sentences):
    # 匹配模式: ..., 但是...
    keywords = [[w] for w in contrast_words]
    return pattern.rule_scks(sentence, sub_sentences, keywords, period=False)


def rule301(sentence, sub_sentences):
    # 匹配模式: ...虽然..., ...
    keywords = list(product(base_words, contrast_words))
    return pattern.rule_skscs(sentence, sub_sentences, keywords)


def rule302(sentence, sub_sentences):
    # 匹配模式: 虽然..., ...
    keywords = list(product(base_words, contrast_words))
    return pattern.rule_kscs(sentence, sub_sentences, keywords)


def rule4(sentence, sub_sentences):
    # 匹配模式: ..., 虽然...
    keywords = list(product(base_words, contrast_words))
    match_mode = 'full'
    return pattern.rule_scks(sentence, sub_sentences, keywords, match_mode, reverse=True)


def extract(full_sentence, sub_sentences):
    rules = [rule101, rule102, rule103, rule201, rule202, rule301, rule302, rule4]
    return ru.extract(full_sentence, sub_sentences, rules)

