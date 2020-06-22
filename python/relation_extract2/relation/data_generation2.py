import sys
import os
import relation_util
import relation
import glob
import json
import r_causality
import itertools
import server

module_dir = os.path.dirname(__file__)

work1_dir = os.path.join(module_dir, '..')
sys.path.append(work1_dir)
# import extract.event_extract_program_01.predict2 as pred


url = 'http://192.168.1.7:38082/event_extract'


def get_events(sentence):
    print('processing request')
    data = {'sentence': sentence}
    ret = requests.post(url, data=data)
    print('finished request', ret.status_code)
    return json.loads(ret.text)


def extract_trigger(sentence, relation_):
    rst = relation.extract_by_relation(sentence, relation_)
    # print(rst)
    parts = relation_.parts
    if not rst:
        return '', [], [], []
    cause = rst[parts[1]]
    result = rst[parts[2]]

    # print(cause)
    # print(result)

    events_full = get_events(sentence)['data'][0]['events']

    events_cause = get_events(cause)['data'][0]['events']
    #    print(cause_verb_rsts, '---------')
    if not events_cause:
        return sentence, [], [], []

    events_effect = get_events(result)['data'][0]['events']
    if not events_effect:
        return sentence, [], [], []

    cause_triggers = []
    for r in events_cause:
        trigger = r.get('verb')
        if not trigger:
            continue
        cause_triggers.append(trigger)

    if not cause_triggers:
        return sentence, [], [], []

    result_triggers = []
    for r in events_effect:
        trigger = r.get('verb')
        if not trigger:
            continue
        result_triggers.append(trigger)

    if not result_triggers:
        return sentence, [], [], []

    full_triggers = []
    for r in events_full:
        trigger = r.get('verb')
        full_triggers.append(trigger)

    print('-------------------')
    return sentence, cause_triggers, result_triggers, full_triggers


def extract_triggers_from_articles(txt_dir, record_fp, sample_fp, relation_, identifier):
    txts = glob.glob(f'{txt_dir}/*.txt')
    f2 = open(record_fp, 'a+')
    if not os.path.exists(record_fp):
        record = []
    else:
        record = f2.readlines()
        record = [r.replace('\n', '') for r in record]
    with open(sample_fp, 'a', encoding='utf-8') as f1:
        for txt in txts:
            if txt in record:
                continue
            print(relation_.__name__, txt)
            with open(txt, 'r', encoding='utf-8') as f:
                c = f.read().replace('\t', '').replace(u'\u3000', u'').replace('\n', '')
            sentences = relation_util.split_article_to_sentences(c)
            for s in sentences:
                sentence, cause_triggers, result_triggers, full_triggers = extract_trigger(s, relation_)
                if not sentence:
                    continue
                if not cause_triggers or not result_triggers:
                    continue
                print(sentence)
                f1.write(
                    sentence + '\t' + '|'.join(cause_triggers) + '\t' + '|'.join(result_triggers) + '\t' + '|'.join(
                        full_triggers) + '\t' + str(identifier) + '\n')
                f1.flush()
            f2.write(txt + '\n')
            f2.flush()
    return


def readfile(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.readlines()


def combine_pos_samples():
    dir_ = 'samples_causality'
    fns = ['samples_causality.txt', 'sample_shizhe.txt', 'sample_shizhe1.txt', 'sample_shizhe2.txt']
    fns = [dir_ + '/' + fn for fn in fns]
    lines = []
    for fn in fns:
        lines_f = readfile(fn)
        lines.extend(lines_f)
    lines = [l.split('\t')[0] for l in lines]
    return lines


def extract_sentence(sentence, relation_):
    rst = relation.extract_by_relation(sentence, relation_)

    parts = relation_.parts
    if not rst:
        return '', '', '', [], []
    cause = rst[parts[1]]
    result = rst[parts[2]]

    events_cause = get_events(cause)['data'][0]['events']
    #    print(cause_verb_rsts, '---------')
    if not events_cause:
        return sentence, cause, result, [], []

    events_effect = get_events(result)['data'][0]['events']
    if not events_effect:
        return sentence, cause, result, [], []

    cause_triggers = []
    cause_subjects = []
    cause_objects = []
    for r in events_cause:
        trigger = r.get('verb')
        subject = r.get('subject')
        object_ = r.get('object')
        if not trigger:
            continue
        cause_triggers.append(trigger)
        cause_subjects.append(subject)
        cause_objects.append(object_)

    if not cause_triggers:
        return sentence, cause, result, [], []

    result_triggers = []
    result_subjects = []
    result_objects = []
    for r in events_effect:
        trigger = r.get('verb')
        subject = r.get('subject')
        object_ = r.get('object')
        if not trigger:
            continue
        result_triggers.append(trigger)
        result_subjects.append(subject)
        result_objects.append(object_)

    if not result_triggers:
        return sentence, cause, result, [], []

    print('-------------------')
    return sentence, cause, result, list(zip(cause_subjects, cause_triggers, cause_objects)), list(
        zip(result_subjects, result_triggers, result_objects))


def extract_from_pos_samples():
    lines = combine_pos_samples()
    print('finished combining files')
    f = open('samples_causality_pos_svos.txt', 'a', encoding='utf-8')
    import r_causality
    import r_assumption
    import r_condition
    rs = [r_causality, r_assumption, r_condition]
    for l in lines:
        l = l.replace('\n', '')
        for r in rs:
            sentence, cause, result, cause_svos, result_svos = extract_sentence(l, r)
            if len(cause_svos) == 1 and len(result_svos) == 1:
                save_l = f"{sentence}\t{cause}\t{result}\t{'|'.join(cause_svos[0])}\t{'|'.join(result_svos[0])}\n"
                f.write(save_l)
                f.flush()
    f.close()


def extract_tags():
    l = '很多国家可能会保持中立的态度，但是美国干预的姿态则比较明显，因为它认为罗希亚危机可能是缅甸采用了违反或侵犯人权的手段导致，所以要采取比较强烈的干预政策来修正它。'
    print('finished combining files')
    f = open('samples_causality_tags.txt', 'a', encoding='utf-8')
    l = l.replace(' ', '').replace('\t', '').replace('\u3000', '').replace('\n', '')
    rst, __, __ = relation.extract_all_relations(l)
    rst = rst[0]
    tag_idx = rst['tag_indexes']
    if not tag_idx:
        return
    print(tag_idx)
    save_l = f"{l}\t{tag_idx}\n"
    f.write(save_l)
    f.flush()
    f.close()


def save_neg_sentences(txt_dir, record_fp, sample_fp, relation_):
    txts = glob.glob(f'{txt_dir}/*.txt')
    f2 = open(record_fp, 'a+')
    if not os.path.exists(record_fp):
        record = []
    else:
        record = f2.readlines()
        record = [r.replace('\n', '') for r in record]
    cnt = 0
    with open(sample_fp, 'a', encoding='utf-8') as f1:
        for txt in txts:
            if txt in record:
                continue
            # print(txt)
            with open(txt, 'r', encoding='utf-8') as f:
                c = f.read().replace('\t', '').replace(u'\u3000', u'').replace('\n', '')
            sentences = relation_util.split_article_to_sentences(c)
            for s in sentences:
                try:
                    rst = relation.extract_by_relation(s, relation_)
                    if not rst:
                        f1.write(s + '\n')
                        f1.flush()
                        cnt += 1
                        if cnt > 90:
                            break
                except Exception as e:
                    print('Error!', e)
            f2.write(txt + '\n')
            f2.flush()
            if cnt > 90:
                print('finished')
                break
    return


def extract_neg_svo(sentence, relations):
    no_relation = True
    for relation_ in relations:
        rst = relation.extract_by_relation(sentence, relation_)
        if rst:
            no_relation = False
            break

    if not no_relation:
        return None

    events = get_events(sentence)['data'][0]['events']
    #    print(cause_verb_rsts, '---------')
    if len(events) < 2:
        return None

    comb = itertools.combinations(events, 2)

    for c in comb:
        e1 = c[0]
        e2 = c[1]
        trigger1 = e1.get('verb')
        subject1 = e1.get('subject')
        object1 = e1.get('object')
        trigger2 = e2.get('verb')
        subject2 = e2.get('subject')
        object2 = e2.get('object')
        line = f"{sentence}\t{'|'.join([subject1, trigger1, object1])}\t{'|'.join([subject2, trigger2, object2])}\n"
        return line


def save_c():
    import r_causality
    import r_assumption
    import r_condition
    rs = [r_causality, r_assumption, r_condition]
    for r in rs:
        extract_triggers_from_articles('articles_zh', 'record_causality.txt', 'samples_causality.txt', r, 0)


def save_c_svo_neg():
    import r_causality
    import r_assumption
    import r_condition
    rs = [r_causality, r_assumption, r_condition]
    txts = glob.glob(f'articles_zh/*.txt')
    cnt = 0
    with open('samples_causality_svos_neg.txt', 'a', encoding='utf-8') as f:
        for txt in txts:
            if cnt > 1500:
                break
            with open(txt, 'r', encoding='utf-8') as f2:
                c = f2.read().replace('\t', '').replace(u'\u3000', u'').replace('\n', '')
            sentences = relation_util.split_article_to_sentences(c)
            for s in sentences:
                if cnt > 1500:
                    break
                line = extract_neg_svo(s, rs)
                if line:
                    f.write(line)
                    f.flush()
                    cnt += 1


def save_c_neg():
    import r_causality
    import r_assumption
    import r_condition
    rs = [r_causality, r_assumption, r_condition]
    for r in rs:
        save_neg_sentences('articles_zh', 'record_causality_neg.txt', 'samples_causality_neg.txt', r)


if __name__ == '__main__':
    import r_then
    import r_contrast
    import r_hypernym
    import r_parallel

    # save_c_neg()
    # extract_triggers_from_articles('shizheng', 'record_hypernym.txt', 'samples_hypernym.txt', r_hypernym, 3)
    # save_neg_sentences('shizheng', 'record_hypernym_neg.txt', 'samples_hypernym_neg.txt', r_hypernym)
    # extract_from_pos_samples()
    extract_tags()
