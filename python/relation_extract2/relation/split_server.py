import relation_util
import relation
from flask import Flask, request
import json

from pattern import rule_skcscskcs, rule_kcscs, rule_sckcs, rule_skcscs, rule_scsks, rule_sksks

app = Flask(__name__)

app.config.update(RESTFUL_JSON=dict(ensure_ascii=False))


def split_multi_keywords(sentence, keywords):
    rst = rule_skcscskcs(sentence, [sentence], [keywords])
    if rst:
        return [[rst['source'], rst['target']]]
    rst = rule_skcscskcs(sentence, [sentence], [keywords], comma1=False)
    if rst:
        return [[rst['source'], rst['target']]]
    rst = rule_skcscskcs(sentence, [sentence], [keywords], comma3=False)
    if rst:
        return [[rst['source'], rst['target']]]
    rst = rule_skcscskcs(sentence, [sentence], [keywords], comma1=False, comma3=False)
    if rst:
        return [[rst['source'], rst['target']]]

    rst = rule_sksks(sentence, [sentence], [keywords])
    if rst:
        return [[rst['source'], rst['target']]]
    return None


def split_single_keyword(sentence, keyword):

    min_sentences, delimiters = relation_util.split_sentence(sentence, False)
    # rst, min_sentences, delimiters = relation.extract_all_relations(sentence)
    min_sentences_num = len(min_sentences)

    if min_sentences_num == 1:
        return [sentence.split(keyword[0])]
    if min_sentences_num == 2:
        return [min_sentences]
    keyword_pos = get_keyword_pos(min_sentences, keyword[0])
    if keyword_pos == 0:
        rst = rule_skcscs(sentence, [sentence], [keyword], comma1=False)
        if rst:
            return [[rst['source'], rst['target']]]
        rst = rule_kcscs(sentence, [sentence], [keyword], comma1=False)
        if rst:
            return [[rst['source'], rst['target']]]
        return None
    if keyword_pos == min_sentences_num - 1:
        rst = rule_sckcs(sentence, [sentence], [keyword], comma2=False)
        if rst:
            return [[rst['source'], rst['target']]]
        rst = rule_scsks(sentence, [sentence], [keyword])
        if rst:
            return [[rst['source'], rst['target']]]
        return None

    source1 = ''.join([''.join(z) for z in zip(min_sentences[:keyword_pos], delimiters[:keyword_pos])])
    target1 = ''.join([''.join(z) for z in zip(min_sentences[keyword_pos:], delimiters[keyword_pos:])])

    source2 = ''.join([''.join(z) for z in zip(min_sentences[:keyword_pos + 1], delimiters[:keyword_pos + 1])])
    target2 = ''.join([''.join(z) for z in zip(min_sentences[keyword_pos + 1:], delimiters[keyword_pos + 1:])])

    return [[source1, target1], [source2, target2]]


@app.route("/relation_split", methods=['GET', 'POST'])
def split():
    sentence = request.form.get('sentence')
    print(sentence)
    keyword = request.form.get('keyword')
    keyword = json.loads(keyword)
    if len(keyword) > 1:
        return json.dumps(split_multi_keywords(sentence, keyword))
    else:
        return json.dumps(split_single_keyword(sentence, keyword))


def get_keyword_pos(min_sentences, keyword):
    for i, ms in enumerate(min_sentences):
        if keyword in ms:
            return i
    return len(min_sentences)


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=12320)
