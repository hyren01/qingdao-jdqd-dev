from itertools import product
import relation_util as ru
import pattern

# TODO(zhxin): 处理关系字组成其他常用词的情况
neg_words = ['不', '不曾', '不会', '并不会', '不可', '不可以', '不可能', '不能', '不能够', '避免', '没有', '并没有']
# TODO(zhxin): 处理关系词前出现以下词语的情况: '可能', '很可能', '成为', '变成', '变为'
# TODO(zhxin): 根据连词(如'但是')找出完整的句子, 以适应精确模式与完整模式的调整

relation_name = 'causality'
# parts = ['tag', 'cause', 'effect']
commas = [',', '，']
commas_str = '[' + ''.join(commas) + ']'
corr_words = ['并且', '而且']

cause_conjs = ['由于', '因为', '缘于', '为了']
keywords_cause = [[w] for w in cause_conjs]

effect_conjs = ['从而', '所以', '为此', '因此', '因而', '故而', '从而', '以致于', '以致', '于是', '那么']
effect_conjs = ru.add_char(effect_conjs, commas)
effect_verbs = ['导致', '引发', '引起', '致使', '使得']
effect_verbs = ru.add_char(effect_verbs, ['了'])

adverbs = ['才', '才会', '曾']


# @todo(zhxin): 自从..., ...就...

# def one_to_rule_them_all(sentence, sub_sentences):
#     pattern1 = ''


def rule101(sentence, sub_sentences):
    # 匹配模式: 因为..., 所以..., 而且...
    keyword_pairs = list(product(cause_conjs, effect_conjs, corr_words))
    return pattern.rule_ksckscas(sentence, sub_sentences, keyword_pairs)


def rule102(sentence, sub_sentences):
    # 匹配模式: 因为..., 所以...
    keyword_pairs = list(product(cause_conjs, effect_conjs))
    return pattern.rule_kscks(sentence, sub_sentences, keyword_pairs, match_mode='long')


def rule2(sentence, sub_sentences):
    # 匹配模式: ...之所以..., 是因为...
    keyword_pairs = product(['之所以'], ['是' + c for c in cause_conjs])
    return pattern.rule_skscks(sentence, sub_sentences, keyword_pairs, reverse=True)


def rule3(sentence, sub_sentences):
    # 匹配模式: xxxx的原因是xxxx
    # TODO(zhxin): xxx表示:"这是为了xxxx."
    keywords = ['的原因是', '原因是']
    keywords.extend(['是' + c for c in cause_conjs])
    keywords = [[w] for w in keywords]
    return pattern.rule_sks(sentence, sub_sentences, keywords, reverse=True)


# todo(zhxin): ...结果是...


def rule401(sentence, sub_sentences):
    # 匹配模式: ...因为..., ...
    return pattern.rule_skscs(sentence, sub_sentences, keywords_cause)


def rule402(sentence, sub_sentences):
    # 匹配模式: 因为..., ...
    return pattern.rule_kscs(sentence, sub_sentences, keywords_cause)


def rule5(sentence, sub_sentences):
    # 匹配模式: ..., 因为...
    return pattern.rule_scks(sentence, sub_sentences, keywords_cause, match_mode='full', reverse=True)


def rule6(sentence, sub_sentences):
    # 模式: ...是由...所引起
    effect_words = [''.join(p) for p in product(['', '所'], effect_verbs)]
    keywords = product(['是由'], effect_words)
    return pattern.rule_sksk(sentence, sub_sentences, keywords, match_mode='long')


def rule7(sentence, sub_sentences):
    # 匹配模式:..., 导致...
    keywords = [[w] for w in effect_verbs]
    return pattern.rule_scks(sentence, sub_sentences, keywords, match_mode='full')


def rule702(sentence, sub_sentences):
    # 匹配模式:...导致...
    keywords = [[w] for w in effect_verbs]
    return pattern.rule_sks(sentence, sub_sentences, keywords)


def rule8(sentence, sub_sentences):
    # 匹配模式: xxx是xxxx的原因
    effect_words = [''.join(p) for p in product(['才是', '是'], effect_verbs)]
    keyword_pairs = product(effect_words, ['的真正原因', '的真实原因', '的原因'])
    return pattern.rule_sksk(sentence, sub_sentences, keyword_pairs)


def extract(full_sentence, sub_sentences):
    rules = [rule101, rule102, rule2, rule3, rule401, rule402, rule5, rule6, rule7, rule702, rule8]
    return ru.extract(full_sentence, sub_sentences, rules)


# TODO '王生分析，朝鲜向来不屈服于外界的制裁压力，而且以往的制裁措施对朝鲜本身的影响也十分有限，“朝鲜还是会根据自己的步子进行卫星发射，现在需要判断的是金正恩何时访华，有可能会在明年中国"两会"之后，这不仅是因为到时中国领导人完成了政府和党两个层面的新老交接，也是因为届时美国奥巴马政府和韩国新总统的对朝政策都会明朗起来，目前金正恩主要还是在考虑如何巩固领导地位。”'
# TODO “朝鲜所有行动都是基于国家生存而采取的自卫措施，并不是为了威胁别人。”
# TODO “美海军之所以采取如此做法，一方面是如果全部是军方人员，将引起侦察对象国的强烈反应，而大量使用民事人员则具有很强的隐蔽性和欺骗性”
# TODO(zhxin): ['对于安倍的这种做法，日本国内一直有舆论认为，名义上日本是为了应对朝鲜不断的导弹威胁，但实际上日本政府有两个目的：一是渲染紧张局势，借机加强军备，构建新的日美导弹防御体系，推进日美在东亚地区的军事一体化；另一个更隐秘的目的是为修宪进行舆论上的准备。', {'tag': '是为了', 'cause': '应对朝鲜不断的导弹威胁，但实际上日本政府有两个目的：一是渲染紧张局势，借机加强军备，构建新的日美导弹防御体系，推进日美在东亚地区的军事一体化；另一个更隐秘的目的是为修宪进行舆论上的准备。', 'effect': '对于安倍的这种做法，日本国内一直有舆论认为，名义上日本'}]


if __name__ == '__main__':
    sentence = '因为我是你爸爸, 所以你是我儿子. '
    min_sentences, delimiters = relation_util.split_sentence(sentence)
    sub_sentences = relation_util.slice_sentence(min_sentences, delimiters)
    print(rule102(sentence, sub_sentences))
