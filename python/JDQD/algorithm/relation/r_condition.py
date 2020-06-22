from itertools import product
import relation_util as ru
import pattern

# todo(zhxin): 考虑合并
cond_words1 = ['只要']
result_words1 = ['就将', '便', '就', '才将', '才']
keywords1 = list(product(cond_words1, result_words1))

cond_words2 = ['只有', '除非', '除了']
result_words2 = ['才能', '才会', '才可以', '才']
keywords2 = list(product(cond_words2, result_words2))

relation_name = 'assumption'
parts = ['tag', 'source', 'target']
# parts = ['tag', 'condition', 'result']


def rule101(sentence, sub_sentences):
    # 匹配模式: ...只要..., ...就...
    return pattern.rule_skscsks(sentence, sub_sentences, keywords1, parts)


def rule102(sentence, sub_sentences):
    # 匹配模式: 只要..., ...就...
    return pattern.rule_kscsks(sentence, sub_sentences, keywords1, parts)


def rule103(sentence, sub_sentences):
    # 匹配模式: 只要..., 就...
    return pattern.rule_kscks(sentence, sub_sentences, keywords1, parts)


def rule201(sentence, sub_sentences):
    # 匹配模式: ...只有..., ...才能...
    return pattern.rule_skscsks(sentence, sub_sentences, keywords2, parts)


def rule202(sentence, sub_sentences):
    # 匹配模式: ...只有..., 才能...
    return pattern.rule_skscks(sentence, sub_sentences, keywords2, parts)


def rule203(sentence, sub_sentences):
    # 匹配模式: 只有..., 才能...
    return pattern.rule_kscks(sentence, sub_sentences, keywords2, parts)


def extract(full_sentence, sub_sentences):
    rules = [rule101, rule102, rule103, rule201, rule202, rule203]
    return ru.extract(full_sentence, sub_sentences, rules)

