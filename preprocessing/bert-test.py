import os
import numpy as np
import re
import pandas as pd
from io import StringIO


DEFAULT_PATH='./../data/'
TEST_PATH = os.path.join(DEFAULT_PATH, 'en_data.csv')

for_pd = StringIO()
with open(TEST_PATH, encoding="utf8", errors='ignore') as p:
    for line in p:
        new_line = re.sub(r',', '|', line.rstrip(), count=4)
        print (new_line, file=for_pd)

for_pd.seek(0)

test = pd.read_csv(for_pd, sep='|', header=0)

del test['id']
del test['i_dialog']
del test['i_utterance']
del test['speaker']

with open(os.path.join(DEFAULT_PATH, 'test.txt'), "a") as f:
    for index, row in test.iterrows():
        orig_utterance = row[0]
        cleaned_utterance = re.sub(u'"',u"", orig_utterance)
        cleaned_utterance = re.sub(u'’',u"'", cleaned_utterance)

        line = cleaned_utterance
        print(line)

        f.write(line + '\n')
