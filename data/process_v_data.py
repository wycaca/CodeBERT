# -*- coding: utf-8 -*-

import os
import json
import numpy as np
from more_itertools import chunked

# DATA_DIR='../data/codesearch'

DATA_DIR='F:\AIForProgram\CodeBERT\data'

def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string


def preprocess_test_data(file_name):
    path = os.path.join(DATA_DIR, '{}.json'.format(file_name))
    print(path)
    with open(path, 'r') as pf:
        data = pf.read()
    json_data = json.loads(data)

    idxs = np.arange(len(json_data))
    data = np.array(json_data, dtype=np.object)

    # np.random.seed(0)   # set random seed so that random things are reproducible
    # np.random.shuffle(idxs)
    # data = data[idxs]
    # batched_data = chunked(data, test_batch_size)

    print("start processing")
    examples = []
    for line in data:
        # if len(batch_data) < test_batch_size:
        #     break # the last batch is smaller than the others, exclude.
        # line_a = json.loads(str(d, encoding='utf-8'))
        doc_token = line['sentences'][0]['text']
        # print(doc_token)
        code_token = line['sql'][0]
        # print(code_token)
        example = (str(1), "nothing", "nothing", doc_token, code_token)
        example = '<CODESPLIT>'.join(example)
        examples.append(example)

    data_path = os.path.join(DATA_DIR, 'train_valid/sql/')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    file_path = os.path.join(data_path, 'valid.txt')
    print(file_path)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(examples))

if __name__ == '__main__':
    # file_names = ['advising', 'atis', 'geography', 'imdb', 'restaurants', 'scholar', 'spider', 'yelp']
    file_names = ['advising']
    for f in file_names:
        preprocess_test_data(f)
