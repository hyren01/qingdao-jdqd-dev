from fuzzywuzzy import fuzz
from decimal import Decimal
from pure_triples import *

import jieba
import pandas as pd
import codecs
import json
import synonyms

jieba.load_userdict('./input/vocab.txt')

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
            return .005 * head_sim + .003 * predicate_sim + .002 * object_sim
        else:
            entire_sim = fuzz.token_sort_ratio(self.cut_sent(str_1), self.cut_sent(str_2))
            return .005 * head_sim + .003 * predicate_sim + .002 * entire_sim

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

    def content_triples(self, text):
        triples = self.summarizor.summary_main(text)
        return triples

    def main(self, filepath, content):
        data = self.data_read(filepath)
        # triples =self.content_triples(content)
        triples = self.extractor.triples_main(content)
        if triples:
            similarities = {}
            for i,item in enumerate(data):
                score = []
                for j, tri in enumerate(triples):
                    score.append(self.count_sim(item, tri))
                similarities[i] = max(score)
            sim_sorted = sorted(similarities.items(), key = lambda x : x[1], reverse=True)
            smi = {}
            for item in sim_sorted:
                # rate = item[1]/(len(triples))
                rate = item[1]
                smi[item[0]] = float(Decimal(rate).quantize(Decimal('0.000')))
            return smi


if __name__=="__main__":
    compare = triples_compare()
    filepath = './input/cleaned_events.xlsx'
    content = codecs.open('./input/content.txt', 'r', 'utf-8').read()
    sim_sorted = compare.main(filepath, content)
    with open('./output/event_compare.json', 'w') as f:
        json.dump(sim_sorted, f)
        print("comparation completed")
