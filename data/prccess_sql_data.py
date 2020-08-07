# -*- coding: utf-8 -*-

import os
import json
import numpy as np
from more_itertools import chunked

DATA_DIR='F:\AIForProgram\CodeBERT\data'

def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string


def preprocess_test_data(file_name, type=0):
    path = os.path.join(DATA_DIR, '{}.json'.format(file_name))
    print(path)
    with open(path, 'r') as pf:
        data = pf.read()
    json_data = json.loads(data)

    # idxs = np.arange(len(json_data))
    data = np.array(json_data, dtype=np.object)

    print("start processing")
    examples = []
    for line in data:
        isNegative = np.random.randint(2)
        if isNegative == 0:
            isDoc = np.random.randint(2)
            random_line_num = np.random.randint(len(data))
            if isDoc:
                doc_token = data[random_line_num]['sentences'][0]['text']
                code_token = line['sql'][0]
            else:
                doc_token = line['sentences'][0]['text']
                code_token = data[random_line_num]['sql'][0]
        else:
            doc_token = line['sentences'][0]['text']
            code_token = line['sql'][0]
        example = (str(isNegative), "nothing", "nothing", doc_token, code_token)
        example = '<CODESPLIT>'.join(example)
        examples.append(example)

    data_path = os.path.join(DATA_DIR, 'train_valid/sql')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    output_file_name = "1.txt"
    if type == 0:
        output_file_name = 'train.txt'
    else:
        output_file_name = 'valid.txt'
    file_path = os.path.join(data_path, output_file_name)
    print(file_path)
    with open(file_path, 'a', encoding='utf-8') as f:
        f.writelines('\n'.join(examples))

if __name__ == '__main__':
    # file_names = ['advising', 'atis', 'geography', 'imdb', 'restaurants', 'scholar', 'spider', 'yelp']
    train_file_names = ['atis', 'geography', 'imdb', 'restaurants', 'scholar', 'spider', 'yelp']
    v_file_names = ['advising']
    for f in train_file_names:
        preprocess_test_data(f)
    preprocess_test_data(v_file_names[0], 1)
