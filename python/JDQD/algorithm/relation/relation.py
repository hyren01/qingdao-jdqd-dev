import r_parallel
import r_choice
import r_causality
import r_further
import r_assumption
import r_then
import relation_util
import r_hypernym
import r_condition
import r_contrast

import relation_util as ru


def extract_all_relations(sentence):
    min_sentences, delimiters = ru.split_sentence(sentence)
    sub_sentences = ru.slice_sentence(min_sentences, delimiters)
    rsts = []
    relations = [r_assumption, r_causality, r_condition, r_choice, r_further, r_parallel, r_contrast, r_then,
                 r_hypernym]
    relation_names = [r.__name__.split('_')[1] for r in relations]
    for r, n in zip(relations, relation_names):
        __, rst = r.extract(sentence, sub_sentences)
        if rst:
            rst['relation'] = n
            rsts.append(rst)
    return rsts


"""
def extract_all_articles_by_rule(rule):
    # iter over articles
    rsts = []
    for n, (sentences_article, sub_sentences_article) in articles_sub_sentences.items():
        print(n)
        # iter over sentences of an article
        for s, sub_sentences in zip(sentences_article, sub_sentences_article):
            rst = rule(s, sub_sentences)
            if rst:
                print(s, rst)
                rsts.append(rst)
    return rsts
"""

def extract_all_relations2(sentence):
    min_sentences, delimiters = relation_util.split_sentence(sentence)
    sub_sentences = relation_util.slice_sentence(min_sentences, delimiters)
    relations = [r_causality, r_condition, r_assumption]
    for relation in relations:
        parts = relation.parts[1:]
        rule, rst = relation.extract(sentence, sub_sentences)
        rule = relation.__name__ + rule
        if rst:
            return rst, parts, rule
    return {}, [], ''


if __name__ == '__main__':
    sent = '如果他来了, 那就告诉他他是我儿子. '
    rst = extract_all_relations(sent)
    print(rst)
