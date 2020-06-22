import relation_util as ru
import pattern

choice_words1 = ['要是', '与其']
choice_words2 = ['还不如', '倒不如', '不如']

word_pairs = [['不是', '就是'], ['宁可', '也不'], ['宁愿', '也不'],
              ['不是', '就是'], ['不是', '即是'], ['或者', '或者']]

# parts = ['tag', 'first', 'sencond']


def rule101(sentence, sub_sentences):
    # 匹配模式: ...与其..., ...不如...
    return pattern.rule_skscsks(sentence, sub_sentences, word_pairs)


def rule102(sentence, sub_sentences):
    # 匹配模式: ...与其..., 不如...
    return pattern.rule_skscks(sentence, sub_sentences, word_pairs)


def rule103(sentence, sub_sentences):
    # 匹配模式: ...与其...不如...
    return pattern.rule_skscks(sentence, sub_sentences, word_pairs, comma=False)


def rule104(sentence, sub_sentences):
    # 匹配模式: 与其..., ...不如...
    return pattern.rule_kscsks(sentence, sub_sentences, word_pairs)


def rule105(sentence, sub_sentences):
    # 匹配模式: 与其..., 不如...
    return pattern.rule_kscks(sentence, sub_sentences, word_pairs)


def rule301(sentence, sub_sentences):
    # 匹配模式: ..., ...不如...
    return pattern.rule_scsks(sentence, sub_sentences, word_pairs)


def rule302(sentence, sub_sentences):
    # 匹配模式: ..., 不如...
    return pattern.rule_scks(sentence, sub_sentences, word_pairs)


def extract(full_sentence, sub_sentences):
    rules = [rule101, rule102, rule103, rule104, rule105, rule301, rule302]
    return ru.extract(full_sentence, sub_sentences, rules)

if __name__ == '__main__':
    import relation

