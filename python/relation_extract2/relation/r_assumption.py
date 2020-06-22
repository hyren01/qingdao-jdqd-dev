from itertools import product
import relation_util as ru
import pattern

relation_name = 'assumption'
if_words = ['若', '如果', '假如', '假若', '假使', '假设', '设使', '倘使', '一旦', '要是', '既然']
keywords_if = [[w] for w in if_words]
then_words = ['那么', '则', '就将', '便', '将', '就']
keywords = list(product(if_words, then_words))

# parts = ['tag', 'assumption', 'result']


def rule101(sentence, sub_sentences):
    # 匹配模式: ...如果..., 那么...
    return pattern.rule_skscks(sentence, sub_sentences, keywords)


def rule102(sentence, sub_sentences):
    # 匹配模式: 如果..., 那么...
    match_mode = 'long'
    return pattern.rule_kscks(sentence, sub_sentences, keywords, match_mode)


def rule103(sentence, sub_sentences):
    # 匹配模式: ...如果..., ...就...
    return pattern.rule_skscsks(sentence, sub_sentences, keywords)


def rule104(sentence, sub_sentences):
    # 匹配模式: 如果..., ...就...
    return pattern.rule_kscsks(sentence, sub_sentences, keywords)


def rule201(sentence, sub_sentences):
    # 匹配模式: 如果..., ...
    match_mode = 'full'
    return pattern.rule_kscs(sentence, sub_sentences, keywords_if, match_mode)


def rule301(sentence, sub_sentences):
    # 匹配模式: ..., 如果...的话.
    keywords = list(product(if_words, ['的话']))
    return pattern.rule_scksk(sentence, sub_sentences, keywords)


def rule302(sentence, sub_sentences):
    # 匹配模式: ..., 如果...
    match_mode = 'full'
    return pattern.rule_scks(sentence, sub_sentences, keywords_if, match_mode)


def extract(full_sentence, sub_sentences):
    rules = [rule101, rule102, rule103, rule104, rule201, rule301, rule302]
    return ru.extract(full_sentence, sub_sentences, rules)
