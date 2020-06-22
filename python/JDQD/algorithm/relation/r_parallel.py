from itertools import product
import relation_util as ru
import pattern

repeat_words = ['一边是', '一边', '有的是', '有的', '有时候', '有时', '一会儿', '一会', '一面', '也']
repeat_pairs = list(product(repeat_words, repeat_words))
repeat_pairs.append(['一方面', '另一方面'])
parallel_pairs = [['既是', '又是'], ['既是', '也是'], ['既', '又'], ['既', '也']]
parallel_pairs.extend(repeat_pairs)
single_conjs = [['与此相同的是'], ['与之类似的是'], ['与之类似'], ['类似的是'], ['类似地'],
                ['与此相同'], ['与此同时'], ['同时'], ['同样的是'], ['同样'], ['另外'], ['另一方面']]

# parts = ['tag', 'a', 'b']
parts = ['tag', 'source', 'target']


def rule101(sentence, sub_sentences):
    # 匹配模式: ...一边..., 一边...
    match_mode = 'short'
    return pattern.rule_skscks(sentence, sub_sentences, parallel_pairs, parts, match_mode)


def rule102(sentence, sub_sentences):
    # 匹配模式: ...一边...一边...
    match_mode = 'short'
    return pattern.rule_skscks(sentence, sub_sentences, parallel_pairs, parts, match_mode, comma=False)


def rule103(sentence, sub_sentences):
    # 匹配模式: 一边...一边...
    match_mode = 'short'
    return pattern.rule_kscks(sentence, sub_sentences, parallel_pairs, parts, match_mode)


def rule104(sentence, sub_sentences):
    # 匹配模式: 一边...一边...
    match_mode = 'short'
    return pattern.rule_kscks(sentence, sub_sentences, parallel_pairs, parts, match_mode, comma=False)


def rule201(sentence, sub_sentences):
    # 匹配模式: ..., 同时, ...
    return pattern.rule_sckcs(sentence, sub_sentences, parallel_pairs, parts)


def rule202(sentence, sub_sentences):
    # 匹配模式: ..., 同时...
    return pattern.rule_sckcs(sentence, sub_sentences, parallel_pairs, parts, comma2=False)


def rule203(sentence, sub_sentences):
    # 匹配模式: ...同时, ...
    return pattern.rule_sckcs(sentence, sub_sentences, parallel_pairs, parts, comma1=False)


def rule204(sentence, sub_sentences):
    # 匹配模式: ...同时 ...
    return pattern.rule_sckcs(sentence, sub_sentences, parallel_pairs, parts, comma1=False, comma2=False)


def extract(full_sentence, sub_sentences):
    rules = [rule101, rule102, rule103, rule104, rule201, rule202, rule203, rule204]
    return ru.extract(full_sentence, sub_sentences, rules)
