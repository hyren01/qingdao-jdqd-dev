import re

period_str = '[.。!！？?]'
comma_str = '[,，]'


class RelationPattern:
    def __init__(self, pattern_reg, parts_order, reverse=False):
        self.pattern_reg = pattern_reg
        self.parts_order = parts_order
        if reverse:
            parts_order[1], parts_order[2] = parts_order[2], parts_order[1]
        self.pattern = re.compile(self.pattern_reg)

    def match_pattern(self, sentence):
        match_result = self.pattern.findall(sentence)
        parts = ['tag', 'source', 'target']
        if match_result:
            match_result_index = sentence.index(''.join(match_result[0]))
            match_result = match_result[0]
            parts_len = {}
            result = {}
            for p, o in zip(parts, self.parts_order):
                hyphen = '-' if p == 'tag' else ''
                part_content = [match_result[i] for i in o]
                parts_len.update([[i, len(match_result[i])] for i in o])
                part_content = hyphen.join(part_content)
                result[p] = part_content
            accu_len = match_result_index
            tag_indexes = {}
            prev_i = 0
            for k, i in enumerate(self.parts_order[0]):
                for j in range(prev_i, i):
                    accu_len += parts_len[j]
                tag_start = accu_len
                accu_len += parts_len[i]
                tag_end = accu_len
                tag_indexes[result['tag'].split('-')[k]] = [tag_start, tag_end]
                prev_i = i + 1
            result['tag_indexes'] = tag_indexes
            return result
        return {}


class Rule(object):
    def __init__(self, keywords, pattern_reg, parts_order, reverse):
        self.keywords = keywords
        self.pattern_reg = pattern_reg
        self.parts_order = parts_order
        if reverse:
            parts_order[1], parts_order[2] = parts_order[2], parts_order[1]

    def extract(self, sentence, sub_sentences, match_mode):
        if match_mode == 'full':
            if len(sub_sentences) > 3:
                return {}
            sub_sentences = [sentence]
        results = []
        results_size = []
        sub_sentences_r = []
        for sub_sent in sub_sentences:
            result = {}
            for w in self.keywords:
                no_keyword = False
                for w_ in w:
                    if w_ not in sub_sent:
                        no_keyword = True
                        break
                if no_keyword:
                    continue
                pattern_reg = self.pattern_reg.format(*w)
                pattern = RelationPattern(pattern_reg, self.parts_order)
                result = pattern.match_pattern(sub_sent)
                if result:
                    break
            if not result:
                continue
            if result['source'] and result['target']:
                results.append(result)
                results_size.append(len(sub_sent))
                sub_sentences_r.append(sub_sent)
        if not results:
            return {}
        if match_mode == 'short':
            result_index = results_size.index(min(results_size))
        else:
            result_index = results_size.index(max(results_size))
        sub_sent_r = sub_sentences_r[result_index]
        sub_sent_r_index = sentence.index(sub_sent_r)
        rst = results[result_index]
        tag_indexes = rst['tag_indexes']
        for t, idx in tag_indexes.items():
            idx_ = [i + sub_sent_r_index for i in idx]
            tag_indexes.update({t: idx_})
        return rst


def rule_ksckscas(sentence, sub_sentences, keywords, match_mode='short', reverse=False):
    parts_order = [[0, 2], [1], [3]]
    pattern_reg = r'({})(.*[,，])({})(.*[,，]{}.*)'
    rule = Rule(keywords, pattern_reg, parts_order, reverse)
    return rule.extract(sentence, sub_sentences, match_mode)


def rule_kscks(sentence, sub_sentences, keywords, match_mode='short', reverse=False, comma=True):
    parts_order = [[0, 2], [1], [3]]
    if comma:
        pattern_reg = r'({})(.*[,，])({})(.*)'
    else:
        pattern_reg = r'({})(.*)({})(.*)'
    rule = Rule(keywords, pattern_reg, parts_order, reverse)
    return rule.extract(sentence, sub_sentences, match_mode)


def rule_skscks(sentence, sub_sentences, keywords, match_mode='short', reverse=False, comma=True):
    parts_order = [[1, 3], [0, 2], [4]]
    if comma:
        pattern_reg = '(.*)({})(.*[,，])({})(.*)'
    else:
        pattern_reg = '(.*)({})(.*)({})(.*)'
    rule = Rule(keywords, pattern_reg, parts_order, reverse)
    return rule.extract(sentence, sub_sentences, match_mode)


def rule_sks(sentence, sub_sentences, keywords, match_mode='short', reverse=False):
    parts_order = [[1], [0], [2]]
    pattern_reg = r'(.*)({})(.*)'
    rule = Rule(keywords, pattern_reg, parts_order, reverse)
    return rule.extract(sentence, sub_sentences, match_mode)


def rule_kscs(sentence, sub_sentences, keywords, match_mode='short', reverse=False, period=True):
    parts_order = [[0], [1], [2]]
    if period:
        pattern_reg = r'^({})(.*?[,，])(.*?[。.？?！!；;])$'
    else:
        pattern_reg = r'^({})(.*?[,，])(.*)'
    rule = Rule(keywords, pattern_reg, parts_order, reverse)
    return rule.extract(sentence, sub_sentences, match_mode)


def rule_skscs(sentence, sub_sentences, keywords, match_mode='short', reverse=False, period=True):
    parts_order = [[1], [0, 2], [3]]
    if period:
        pattern_reg = r'(.*)({})(.*?[,，])(.*?[。.？?！!；;])$'
    else:
        pattern_reg = r'(.*)({})(.*?[,，])(.*)'
    rule = Rule(keywords, pattern_reg, parts_order, reverse)
    return rule.extract(sentence, sub_sentences, match_mode)


def rule_scsks(sentence, sub_sentences, keywords, match_mode='short', reverse=False, period=True, comma=True):
    parts_order = [[2], [0], [1, 3]]
    pattern_reg = r'(.*?cm)(.*?)({})(.*pr)'
    pr = period_str if period else ''
    cm = comma_str if comma else ''
    pattern_reg = pattern_reg.replace('pr', pr).replace('cm', cm)
    rule = Rule(keywords, pattern_reg, parts_order, reverse)
    return rule.extract(sentence, sub_sentences, match_mode)


def rule_scks(sentence, sub_sentences, keywords, match_mode='short', reverse=False, period=True):
    sub_clause_order = [[0], [2]]
    if period:
        pattern_reg = r'(.*?[,，])({})(.*?[。.？?！!；;])$'
    else:
        pattern_reg = r'(.*?[,，])({})(.*)'
    parts_order = [[1], *sub_clause_order]
    rule = Rule(keywords, pattern_reg, parts_order, reverse)
    return rule.extract(sentence, sub_sentences, match_mode)


def rule_sksk(sentence, sub_sentences, keywords, match_mode='short', reverse=False):
    parts_order = [[1, 3], [2], [0]]
    pattern_reg = r'(.*)({})(.*)({})'
    rule = Rule(keywords, pattern_reg, parts_order, reverse)
    return rule.extract(sentence, sub_sentences, match_mode)


def rule_kscsks(sentence, sub_sentences, keywords, match_mode='short', reverse=False):
    parts_order = [[0, 3], [1], [2, 4]]
    pattern_reg = r'({})(.*?[,，])(.*?)({})(.*)'
    rule = Rule(keywords, pattern_reg, parts_order, reverse)
    return rule.extract(sentence, sub_sentences, match_mode)


def rule_scksk(sentence, sub_sentences, keywords, match_mode='short', reverse=False, period=True):
    parts_order = [[1, 3], [2, 4], [0]]
    if period:
        pattern_reg = r'(.*[,，])({})(.*)({})([.。!！？?])'
    else:
        pattern_reg = r'(.*[,，])({})(.*)({})(.*)'
    rule = Rule(keywords, pattern_reg, parts_order, reverse)
    return rule.extract(sentence, sub_sentences, match_mode)


def rule_skscsks(sentence, sub_sentences, keywords, match_mode='short', reverse=False, period=True, comma=True):
    parts_order = [[1, 4], [0, 2], [3, 5]]
    pattern_reg = r'(.*)({})(.*cm1)(.*)({})(.*pr)'
    pr = period_str if period else ''
    cm1 = comma_str if comma else ''
    pattern_reg = pattern_reg.replace('pr', pr).replace('cm1', cm1)
    rule = Rule(keywords, pattern_reg, parts_order, reverse)
    return rule.extract(sentence, sub_sentences, match_mode)


def rule_skcscskcs(sentence, sub_sentences, keywords, match_mode='short', reverse=False, comma1=True,
                   comma2=True, comma3=True):
    parts_order = [[1, 4], [0, 2], [3, 5]]
    pattern_reg = r'(.*)({}cm1)(.*cm2)(.*)({}cm3)(.*)'
    cm1 = comma_str if comma1 else ''
    cm2 = comma_str if comma2 else ''
    cm3 = comma_str if comma3 else ''
    pattern_reg = pattern_reg.replace('cm1', cm1).replace('cm2', cm2).replace('cm3', cm3)
    rule = Rule(keywords, pattern_reg, parts_order, reverse)
    return rule.extract(sentence, sub_sentences, match_mode)


def rule_sksks(sentence, sub_sentences, keywords, match_mode='short', reverse=False):
    parts_order = [[1, 3], [0, 2], [4]]
    pattern_reg = r'(.*)({})(.*)({})(.*)'
    rule = Rule(keywords, pattern_reg, parts_order, reverse)
    return rule.extract(sentence, sub_sentences, match_mode)


def rule_skcscs(sentence, sub_sentences, keywords, match_mode='short', reverse=False, comma1=True, comma2=True):
    parts_order = [[1], [0, 2], [3]]
    pattern_reg = r'(.*?)({}cm1)(.*?cm2)(.*)'
    cm1 = comma_str if comma1 else ''
    cm2 = comma_str if comma2 else ''
    pattern_reg = pattern_reg.replace('cm1', cm1).replace('cm2', cm2)
    rule = Rule(keywords, pattern_reg, parts_order, reverse)
    return rule.extract(sentence, sub_sentences, match_mode)


def rule_kcscs(sentence, sub_sentences, keywords, match_mode='short', reverse=False, comma1=True, comma2=True):
    parts_order = [[0], [1], [2]]
    pattern_reg = r'({}cm1)(.*?cm2)(.*)'
    cm1 = comma_str if comma1 else ''
    cm2 = comma_str if comma2 else ''
    pattern_reg = pattern_reg.replace('cm1', cm1).replace('cm2', cm2)
    rule = Rule(keywords, pattern_reg, parts_order, reverse)
    return rule.extract(sentence, sub_sentences, match_mode)


def rule_sckcs(sentence, sub_sentences, keywords, match_mode='short', reverse=False, period=True, comma1=True,
               comma2=True):
    pattern_reg = r'(.*cm1)({}cm2)(.*)pr'
    parts_order = [[1], [0], [2]]
    pr = period_str if period else ''
    cm1 = comma_str if comma1 else ''
    cm2 = comma_str if comma2 else ''
    pattern_reg = pattern_reg.replace('pr', pr).replace('cm1', cm1).replace('cm2', cm2)
    rule = Rule(keywords, pattern_reg, parts_order, reverse)
    return rule.extract(sentence, sub_sentences, match_mode)


if __name__ == '__main__':
    sentence = '如果女方表示婉拒亲吻或亲密举动时, 男人将做什么应对呢？'
    sentence2 = '研究负责人雷格南特博士说：“虽然‘伤心综合征’不被人们所认识，但我们发现，只要病人在头48小时内充分得到心理及生理的救助，病人的恢复就会非常好，但如果失去这一最佳治疗时间，就会过早夺取病人的生命。'
    keywords = ['如果', '将']
    keywords2 = ['如果', '就']

    rst = rule_kscsks(sentence, [sentence], [keywords])
    print(rst)
    print([sentence[v[0]: v[1]] for v in rst['tag_indexes'].values()])
