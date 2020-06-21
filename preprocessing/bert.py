import os
import numpy as np
import re
import pandas as pd
import json
from pandas.io.json import json_normalize

DEFAULT_PATH='./../data/'
TRAIN_PATH = os.path.join(DEFAULT_PATH, 'friends_train.json')
DEV_PATH = os.path.join(DEFAULT_PATH, 'friends_dev.json')
TEST_PATH = os.path.join(DEFAULT_PATH, 'friends_test.json')

with open(TRAIN_PATH) as train_file:
    train = json.load(train_file)
with open(DEV_PATH) as dev_file:
    dev = json.load(dev_file)
with open(TEST_PATH) as test_file:
    test = json.load(test_file)

train = [lst for sub in train for lst in sub]
dev = [lst for sub in dev for lst in sub]
test = [lst for sub in test for lst in sub]

with open(os.path.join(DEFAULT_PATH, 'bert_train.txt'), "a") as f:
    for line in train:
        orig_utterance = line['utterance']
        orig_label = line['emotion']
        
        if type(orig_utterance) == str:
            cleaned_utterance = re.sub(u'’',u"'", orig_utterance)
            
            line = orig_label + '\t' + cleaned_utterance
            print(line)
            
            f.write(line + '\n')

with open(os.path.join(DEFAULT_PATH, 'bert_train.txt'), "a") as f:
    for line in dev:
        orig_utterance = line['utterance']
        orig_label = line['emotion']
        
        if type(orig_utterance) == str:
            cleaned_utterance = re.sub(u'’',u"'", orig_utterance)
            
            line = orig_label + '\t' + cleaned_utterance
            print(line)
            
            f.write(line + '\n')

with open(os.path.join(DEFAULT_PATH, 'bert_train.txt'), "a") as f:
    for line in test:
        orig_utterance = line['utterance']
        orig_label = line['emotion']
        
        if type(orig_utterance) == str:
            cleaned_utterance = re.sub(u'’',u"'", orig_utterance)
            
            line = orig_label + '\t' + cleaned_utterance
            print(line)
            
            f.write(line + '\n')


