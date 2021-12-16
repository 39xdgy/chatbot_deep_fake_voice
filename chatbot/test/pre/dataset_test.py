import numpy as np
import pandas as pd

from convokit import Corpus, download
corpus = Corpus(filename = download("reddit-corpus-small"))

corpus.print_summary_stats()

convos = corpus.conversations

for i, (key, value) in enumerate(convos.items()):
    print(i, ': ')

    for j in range(len(value.get_utterance_ids())):
        utt = corpus.get_utterance(utt_id=value.get_utterance_ids()[j])

        if utt.text == '':
            continue

        if not (utt.text[0] == '[' and utt.text[-1] == ']'):
            print("Text:", utt.text, "\n")

    print('\n\n')

    if i == 10:
        break