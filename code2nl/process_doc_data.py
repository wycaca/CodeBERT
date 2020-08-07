# -*- coding: utf-8 -*-

import os
import json
import numpy as np
from more_itertools import chunked
from lib.query import Query
from lib.table import Table
# from transformers import BertTokenizer
import re
from typing import List

DATA_DIR='F:\AIForProgram\WikiSQL\data'

def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string


DOCSTRING_REGEX_TOKENIZER = re.compile(r"[^\s,'\"`.():\[\]=*;>{\}+-/\\]+|\\+|\.+|\(\)|{\}|\[\]|\(+|\)+|:+|\[+|\]+|{+|\}+|=+|\*+|;+|>+|\++|-+|/+")


def tokenize_docstring_from_string(docstr: str) -> List[str]:
    return [t for t in DOCSTRING_REGEX_TOKENIZER.findall(docstr) if t is not None and len(t) > 0]

def toQueryStr(file_name, table_dic, type=0, test_batch_size=1000):
    path = os.path.join(DATA_DIR, '{}.jsonl'.format(file_name))
    print(path)
    with open(path, 'r') as pf:
        data = pf.readlines()
    
    idxs = np.arange(len(data))
    data = np.array(data, dtype=np.object)

    np.random.seed(0)   # set random seed so that random things are reproducible
    np.random.shuffle(idxs)
    data = data[idxs]
    batched_data = chunked(data, test_batch_size)

    print("start processing")
    # tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    # tokenizer.add_special_tokens(False)
    examples = []
    for batch_idx, batch_data in enumerate(batched_data):
        if len(batch_data) < test_batch_size:
            break # the last batch is smaller than the others, exclude.
        for d_idx, d in enumerate(batch_data): 
            line = json.loads(str(d), encoding='utf-8')
            doc_str = line['question']
            code_arr = line['sql']
            query = Query(code_arr['sel'], code_arr['agg'], code_arr['conds'])

            id = line['table_id']
            if id in table_dic:
                table = table_dic[id]
                code_str = table.query_str(query)
                example = dict()
                example['docstring_tokens'] = tokenize_docstring_from_string(doc_str)
                example['code_tokens'] = tokenize_docstring_from_string(code_str)
                examples.append(json.dumps(example))

    data_path = os.path.join(DATA_DIR, 'wiki_sql')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    output_file_name = "1.txt"
    if type == 0:
        output_file_name = 'train.jsonl'
    elif type == 2:
        output_file_name = 'test.jsonl'
    else:
        output_file_name = 'valid.jsonl'
    file_path = os.path.join(data_path, output_file_name)
    print(file_path)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(examples))

def read_tableDic(file_name, test_batch_size=1000):
    path = os.path.join(DATA_DIR, '{}.jsonl'.format(file_name))
    print(path)
    with open(path, 'r') as pf:
        data = pf.readlines()
    
    idxs = np.arange(len(data))
    data = np.array(data, dtype=np.object)

    np.random.seed(0)   # set random seed so that random things are reproducible
    np.random.shuffle(idxs)
    data = data[idxs]
    batched_data = chunked(data, test_batch_size)

    print("start processing")
    table_dic = dict()
    for batch_idx, batch_data in enumerate(batched_data):
        if len(batch_data) < test_batch_size:
            break # the last batch is smaller than the others, exclude.
        for d_idx, d in enumerate(batch_data): 
            line = json.loads(str(d), encoding='utf-8')
            id = line['id']
            header = line['header']
            types = line['types']
            rows = line['rows']
            table = Table(id, header, types, rows)
            table_dic[id] = table
    return table_dic


if __name__ == '__main__':
    wiki_table_file_name = "train.tables"
    table_arr = read_tableDic(wiki_table_file_name)

    wiki_sql_train_file_name = "train"
    toQueryStr(wiki_sql_train_file_name, table_arr)

    wiki_table_file_name = "dev.tables"
    table_arr = read_tableDic(wiki_table_file_name)

    wiki_sql_val_file_name = "dev"
    toQueryStr(wiki_sql_val_file_name, table_arr, 1)

    wiki_table_file_name = "test.tables"
    table_arr = read_tableDic(wiki_table_file_name)

    wiki_sql_test_file_name = "test"
    toQueryStr(wiki_sql_test_file_name, table_arr, 2)