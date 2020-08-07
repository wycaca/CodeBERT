# -*- coding: utf-8 -*-

import os
import json
import numpy as np
from more_itertools import chunked
from lib.query import Query
from lib.table import Table

DATA_DIR='F:\AIForProgram\WikiSQL\data'

def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string


def preprocess_test_data(file_name, test_batch_size=1000):
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
    examples = []
    for batch_idx, batch_data in enumerate(batched_data):
        if len(batch_data) < test_batch_size:
            break # the last batch is smaller than the others, exclude.
        for d_idx, d in enumerate(batch_data): 
            line = json.loads(str(d), encoding='utf-8')
            doc_token = line['question']
            example = (str(1), "nothing", "nothing", doc_token)
            example = '<CODESPLIT>'.join(example)
            examples.append(example)
    data_path = os.path.join(DATA_DIR, 'test/wiki_sql')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    output_file_name = "test.txt"
    # if type == 0:
    #     output_file_name = 'train.txt'
    # else:
    #     output_file_name = 'valid.txt'
    file_path = os.path.join(data_path, output_file_name)
    print(file_path)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(examples))

def toQueryStr(file_name, table_arr, type=0, test_batch_size=1000):
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
    examples = []
    for batch_idx, batch_data in enumerate(batched_data):
        if len(batch_data) < test_batch_size:
            break # the last batch is smaller than the others, exclude.
        for d_idx, d in enumerate(batch_data): 
            line = json.loads(str(d), encoding='utf-8')
            doc_token = line['question']
            code_arr = line['sql']
            query = Query(code_arr['sel'], code_arr['agg'], code_arr['conds'])

            id = line['table_id']
            table = Table("table_id", "header", "types", "rows")
            code_str = ''
            for table in table_arr:
                if table.table_id == id:
                    table = table
                    code_str = table.query_str(query)
                    break
                else:
                    continue
            isNegative = np.random.randint(2)
            if isNegative == 0:
                random_line_num = np.random.randint(len(data))
                line = json.loads(str(data[random_line_num]), encoding='utf-8')
                doc_token = line['question']
                code_token = code_str
            else:
                code_token = code_str
            example = (str(isNegative), "nothing", "nothing", doc_token, code_token)
            example = '<CODESPLIT>'.join(example)
            examples.append(example)
    data_path = os.path.join(DATA_DIR, 'train_valid/wiki_sql')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    output_file_name = "1.txt"
    if type == 0:
        output_file_name = 'train.txt'
    else:
        output_file_name = 'valid.txt'
    file_path = os.path.join(data_path, output_file_name)
    print(file_path)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(examples))

def read_tableArr(file_name, test_batch_size=1000):
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
    table_arr = []
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
            table_arr.append(table)
    return table_arr


if __name__ == '__main__':
    # wiki_table_file_name = "train.tables"
    # table_arr = read_tableArr(wiki_table_file_name)

    # wiki_sql_train_file_name = "train"
    # toQueryStr(wiki_sql_train_file_name, table_arr)

    # wiki_table_file_name = "dev.tables"
    # table_arr = read_tableArr(wiki_table_file_name)

    # wiki_sql_val_file_name = "dev"
    # toQueryStr(wiki_sql_val_file_name, table_arr, 1)


    # wiki_table_file_name = "test.tables"
    # table_arr = read_tableArr(wiki_table_file_name)

    # wiki_sql_test_file_name = "test"
    # toQueryStr(wiki_sql_test_file_name, table_arr)
