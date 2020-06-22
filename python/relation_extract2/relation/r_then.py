from itertools import product
import relation_util as ru
import pattern

first_words = ['首先', '先是', '先', '第一步', '第一', '事先']
second_words = ['然后', '之后', '紧接着', '接着', '接下来', '再', '第二步', '第二']
keywords = list(product(first_words, second_words))
# keywords.extend(list(product(['一', '才'], ['就'])))
# TODO(zhxin): 一..., ...就...

# parts = ['tag', 'first', 'sencond']


def rule101(sentence, sub_sentences):
    # 匹配模式: ...首先..., ...然后...
    return pattern.rule_skscsks(sentence, sub_sentences, keywords)


def rule102(sentence, sub_sentences):
    # 匹配模式: ...首先..., 然后...
    return pattern.rule_skscks(sentence, sub_sentences, keywords)


def rule103(sentence, sub_sentences):
    # 匹配模式: ...首先...然后...
    return pattern.rule_skscks(sentence, sub_sentences, keywords, comma=False)


def rule104(sentence, sub_sentences):
    # 匹配模式: 首先..., ...然后...
    return pattern.rule_kscsks(sentence, sub_sentences, keywords)


def rule105(sentence, sub_sentences):
    # 匹配模式: 首先..., 然后...
    return pattern.rule_kscks(sentence, sub_sentences, keywords)


def rule301(sentence, sub_sentences):
    # 匹配模式: ..., ...然后...
    return pattern.rule_scsks(sentence, sub_sentences, keywords)


def rule302(sentence, sub_sentences):
    # 匹配模式: ..., 然后...
    return pattern.rule_scks(sentence, sub_sentences, keywords)


def extract(full_sentence, sub_sentences):
    rules = [rule101, rule102, rule103, rule104, rule105, rule301, rule302]
    return ru.extract(full_sentence, sub_sentences, rules)
