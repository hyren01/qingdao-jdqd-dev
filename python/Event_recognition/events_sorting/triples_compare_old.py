from fuzzywuzzy import fuzz
from decimal import Decimal
from algo_pure_triples import *

import jieba
import pandas as pd
import codecs
import json
import synonyms

jieba.load_userdict('/hyren/python/app/qingdao/event_match/events_sorting/ziliao/vocab.txt')

class triples_compare:
    def __init__(self):
        self.extractor = TripleExtractor()

    def cut_sent(self, sentence):
        sentence_cut = jieba.cut(sentence)
        return ' '.join([item for item in sentence_cut])

    def count_sim(self, list_1, list_2):
        assert len(list_2) == 3

        head_1 = ''.join(list_1[0])
        predicate_1 = ''.join(list_1[1])
        synonyms_head_1 = synonyms.nearby(head_1)
        synonyms_predicate_1 = synonyms.nearby(predicate_1)[0]

        head_2 = ''.join(list_2[0])
        predicate_2 = ''.join(list_2[1])

        str_1 = ''.join(list_1)
        str_2 = ''.join(list_2)

        head_sim = 0
        if head_2 in synonyms_head_1:
            head_sim = .3 * fuzz.token_sort_ratio(self.cut_sent(head_1), self.cut_sent(head_2)) + .7 * 100
        else:
            head_sim = fuzz.token_sort_ratio(self.cut_sent(head_1), self.cut_sent(head_2))

        predicate_sim = 0
        if predicate_2 in set(synonyms_predicate_1):
            predicate_sim = 100

        if len(list_1) == 3:
            object_sim = fuzz.token_sort_ratio(list_1[2], list_2[2])
            entire_smi = fuzz.token_sort_ratio(self.cut_sent(str_1), self.cut_sent(str_2))
            final_result = (.2*head_sim + .2*predicate_sim + .4*object_sim + .2*entire_smi)/100
            final_result = final_result/2 if head_sim ==0 or predicate_sim == 0 or object_sim == 0 else None
            return final_result
        else:
            entire_sim = fuzz.token_sort_ratio(self.cut_sent(str_1), self.cut_sent(str_2))
            final_result = (.2*head_sim + .2*predicate_sim + .5*entire_sim)/100
            final_result = final_result/2 if head_sim == 0 or predicate_sim == 0 else None
            return final_result

    def data_read(self, filepath):
        df = pd.read_excel(filepath)
        data = []
        for i in range(len(df)):
            line = list(df.iloc[i].values)
            if isinstance(line[2], float):
                data.append(line[0:2])
            else:
                data.append(line)
        return data

    def main(self, filepath, content):
        data = self.data_read(filepath)
        triples = self.extractor.triples_main(content)
        if triples:
            similarities = {}
            for i,item in enumerate(data):
                score = []
                for j, tri in enumerate(triples):
                    score.append(self.count_sim(item, tri))
                similarities[i] = [max(score), triples[score.index(max(score))]]
            sim_sorted = sorted(similarities.items(), key = lambda x : x[1][0], reverse=True)
            smi = {}
            for item in sim_sorted:
                rate = item[1][0]
                smi[item[0]] = [float(Decimal(rate).quantize(Decimal('0.000'))), item[1][1]]
            return smi


def execute(content):
    print(f"content={content}")
    # compare = triples_compare()
    filepath = '/hyren/python/app/qingdao/event_match/events_sorting/ziliao/cleaned_events.xlsx'
    triples_compare_func = triples_compare()
    sim = triples_compare_func.main(filepath, content)
    sim_json = []
    for key, item in sim.items():
        sim_json.append({'id:': key, 'ratio:': item[0], 'sents:': ','.join(item[1])})
    return json.dumps(sim_json, ensure_ascii=False)


if __name__=="__main__":

    content = codecs.open('./input/content.txt', 'r', 'utf-8').read()

    print(execute(content))
