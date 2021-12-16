import numpy as np
import pandas as pd

import json
import os

from convokit import Corpus, download
corpus = Corpus(filename = download("reddit-corpus-small"))

corpus.print_summary_stats()

convos = corpus.conversations

List = []
for i, (key, value) in enumerate(convos.items()):

    sub_list = []
    for j in range(len(value.get_utterance_ids())):
        utt = corpus.get_utterance(utt_id=value.get_utterance_ids()[j])

        if utt.text == '':
            continue

        if not (utt.text[0] == '[' and utt.text[-1] == ']'):
            sub_list.append(utt.text)

    for k in range(len(sub_list) - 1):
        dic = {}
        dic['listen'], dic['reply'] = sub_list[k], sub_list[k+1]

        List.append(dic)

print(len(List))

def write_json(List, path):
    if os.path.exists(path):
        os.remove(path)
    with open(path, 'w') as f:
        for dic in List:
            jsObj = json.dumps(dic)
            f.writelines(jsObj + '\n')

path = '/home/wyundi/Server/Courses/BIA667/Project/data/convokit/reddit_small.json'
write_json(List[:10000], path)