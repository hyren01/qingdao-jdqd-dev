import numpy as np
import re


class RelationPattern:
    def __init__(self, pattern_reg, parts, parts_order, reverse=False):
        self.pattern_reg = pattern_reg
        self.parts = parts
        self.parts_order = parts_order
        if reverse:
            parts_order[1], parts_order[2] = parts_order[2], parts_order[1]
        self.pattern = re.compile(self.pattern_reg)

    def match_pattern(self, sentence):
        match_result = self.pattern.findall(sentence)
        if match_result:
            match_result = match_result[0]
            result = {}
            for p, o in zip(self.parts, self.parts_order):
                hyphen = '-' if p == 'tag' else ''
                part_content = [match_result[i] for i in o]
                part_content = hyphen.join(part_content)
                result[p] = part_content
            return result
        return {}


class Rule(object):
    def __init__(self, keywords, pattern_reg, parts, parts_order, reverse):
        self.keywords = keywords
        self.pattern_reg = pattern_reg
        self.parts = parts
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
                pattern = RelationPattern(pattern_reg, self.parts, self.parts_order)
                result = pattern.match_pattern(sub_sent)
                if result:
                    break
            if not result:
                continue
            result_is_valid = True
            for p in self.parts:
                if not result[p]:
                    result_is_valid = False
                    break
            if result_is_valid:
                results.append(result)
                results_size.append(len(sub_sent))
        if not results:
            return {}
        result_index = np.argmin(results_size) if match_mode == 'short' else np.argmax(results_size)
        return results[result_index]


def rule_ksckscas(sentence, sub_sentences, keywords, parts, match_mode='short', reverse=False):
    parts_order = [[0, 2], [1], [3]]
    pattern_reg = r'({})(.*[,，])({})(.*[,，]{}.*)'
    rule = Rule(keywords, pattern_reg, parts, parts_order, reverse)
    return rule.extract(sentence, sub_sentences, match_mode)


def rule_sckcs(sentence, sub_sentences, keywords, parts, match_mode='short', reverse=False, comma1=True, comma2=True):
    parts_order = [[1], [0], [1]]
    if comma1 and comma2:
        pattern_reg = r'(.*[,，])({})[,，](.*)'
    elif comma1 and not comma2:
        pattern_reg = r'(.*[,，])({})(.*)'
    elif comma2 and not comma1:
        pattern_reg = r'(.*)({})[,，](.*)'
    else:
        pattern_reg = r'(.*)({})(.*)'
    rule = Rule(keywords, pattern_reg, parts, parts_order, reverse)
    return rule.extract(sentence, sub_sentences, match_mode)


def rule_kscks(sentence, sub_sentences, keywords, parts, match_mode='short', reverse=False, comma=True):
    parts_order = [[0, 2], [1], [3]]
    if comma:
        pattern_reg = r'({})(.*[,，])({})(.*)'
    else:
        pattern_reg = r'({})(.*)({})(.*)'
    rule = Rule(keywords, pattern_reg, parts, parts_order, reverse)
    return rule.extract(sentence, sub_sentences, match_mode)


def rule_skscks(sentence, sub_sentences, keywords, parts, match_mode='short', reverse=False, comma=True):
    parts_order = [[1, 3], [0, 2], [4]]
    if comma:
        pattern_reg = '(.*)({})(.*[,，])({})(.*)'
    else:
        pattern_reg = '(.*)({})(.*)({})(.*)'
    rule = Rule(keywords, pattern_reg, parts, parts_order, reverse)
    return rule.extract(sentence, sub_sentences, match_mode)


def rule_sks(sentence, sub_sentences, keywords, parts, match_mode='short', reverse=False):
    parts_order = [[1], [0], [2]]
    pattern_reg = r'(.*)({})(.*)'
    rule = Rule(keywords, pattern_reg, parts, parts_order, reverse)
    return rule.extract(sentence, sub_sentences, match_mode)


def rule_kscs(sentence, sub_sentences, keywords, parts, match_mode='short', reverse=False, period=True):
    parts_order = [[0], [1], [2]]
    if period:
        pattern_reg = r'^({})(.*?[,，])(.*?)[。.？?！!；;]$'
    else:
        pattern_reg = r'^({})(.*?[,，])(.*)'
    rule = Rule(keywords, pattern_reg, parts, parts_order, reverse)
    return rule.extract(sentence, sub_sentences, match_mode)


def rule_skscs(sentence, sub_sentences, keywords, parts, match_mode='short', reverse=False, period=True):
    parts_order = [[1], [0, 2], [3]]
    if period:
        pattern_reg = r'(.*)({})(.*?[,，])(.*?)[。.？?！!；;]$'
    else:
        pattern_reg = r'(.*)({})(.*?[,，])(.*)'
    rule = Rule(keywords, pattern_reg, parts, parts_order, reverse)
    return rule.extract(sentence, sub_sentences, match_mode)


def rule_scsks(sentence, sub_sentences, keywords, parts, match_mode='short', reverse=False, period=True):
    parts_order = [[2], [0], [1, 3]]
    if period:
        pattern_reg = r'(.*?[,，])(.*?)({})(.*?[。.？?！!；;])$'
    else:
        pattern_reg = r'(.*?[,，])(.*?)({})(.*)'
    rule = Rule(keywords, pattern_reg, parts, parts_order, reverse)
    return rule.extract(sentence, sub_sentences, match_mode)


def rule_scks(sentence, sub_sentences, keywords, parts, match_mode='short', reverse=False, period=True):
    sub_clause_order = [[0], [2]]
    if period:
        pattern_reg = r'(.*?[,，])({})(.*?[。.？?！!；;])$'
    else:
        pattern_reg = r'(.*?[,，])({})(.*)'
    parts_order = [[1], *sub_clause_order]
    rule = Rule(keywords, pattern_reg, parts, parts_order, reverse)
    return rule.extract(sentence, sub_sentences, match_mode)


def rule_sksk(sentence, sub_sentences, keywords, parts, match_mode='short', reverse=False):
    parts_order = [[1, 3], [2], [0]]
    pattern_reg = r'(.*)({})(.*)({})'
    rule = Rule(keywords, pattern_reg, parts, parts_order, reverse)
    return rule.extract(sentence, sub_sentences, match_mode)


def rule_skscsks(sentence, sub_sentences, keywords, parts, match_mode='short', reverse=False):
    parts_order = [[1, 4], [0, 2], [3, 5]]
    pattern_reg = r'(.*?)({})(.*?)[,，](.*?)({})(.*)'
    rule = Rule(keywords, pattern_reg, parts, parts_order, reverse)
    return rule.extract(sentence, sub_sentences, match_mode)


def rule_kscsks(sentence, sub_sentences, keywords, parts, match_mode='short', reverse=False):
    parts_order = [[0, 3], [1], [2, 4]]
    pattern_reg = r'({})(.*?)[,，](.*?)({})(.*)'
    rule = Rule(keywords, pattern_reg, parts, parts_order, reverse)
    return rule.extract(sentence, sub_sentences, match_mode)


def rule_scksk(sentence, sub_sentences, keywords, parts, match_mode='short', reverse=False, period=True):
    parts_order = [[1, 3], [2, 4], [0]]
    if period:
        pattern_reg = r'(.*)[,，]({})(.*)({})([.。!！？?])'
    else:
        pattern_reg = r'(.*)[,，]({})(.*)({})(.*)'
    rule = Rule(keywords, pattern_reg, parts, parts_order, reverse)
    return rule.extract(sentence, sub_sentences, match_mode)
