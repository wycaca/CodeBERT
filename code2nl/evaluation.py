#!/usr/bin/env python
import json
import re
from typing import List
import multiprocessing
from multiprocessing import Pool

from tqdm import tqdm

class MultiReadFile:
    def __init__(self, file=""):
        self.file = file
        self.results = []

    def tokenize_docstring_from_string(self, docstr: List[str]) -> List[str]:
        return [t.lower() for t in docstr if (re.match('table_[0-9]_[0-9]+_[0-9]+', t) is None)]

    def open_file(self, file):
        file = open(file, encoding='utf-8')
        return file

    def chunks(self, l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def parse_chunk(self, data):
        results = []
        for line in data:
            try:
                line_list = self.tokenize_docstring_from_string(line)
                results.append(line_list)
            except Exception as e:
                print(e)
        return results

    def run_process(self, file):
        file = self.open_file(file).readlines()
        listify = [line.split() for line in file]

        data = self.chunks(listify, int(len(listify) / (multiprocessing.cpu_count() - 2)))
        p = Pool(processes=multiprocessing.cpu_count() - 2)
        # p = Pool(1)
        results = [p.apply_async(self.parse_chunk, args=(list(x),)) for x in data]

        # wait for results
        results = [item.get() for item in results]
        self.results = sum(results, [])

if __name__ == '__main__':
    source_file = r'D:\download\\test_1.gold'
    pred_file = r'D:\download\\test_1.output'

    match = 0
    total = 0
    gold_tokens = []
    pred_tokens = []
    
    gold_mult = MultiReadFile()
    gold_mult.run_process(source_file)
    gold_tokens = gold_mult.results

    pred_mult = MultiReadFile()
    pred_mult.run_process(pred_file)
    pred_tokens = pred_mult.results

    for gold_line_list, pred_line_list in zip(gold_tokens, pred_tokens):
        for gold_token, pred_token in zip(gold_line_list, pred_line_list):
            if gold_token == pred_token:
                match = match + 1
            total = total + 1
    print("match token: ", str(match))
    print("total token: ", str(total))
    print(str(match / total))
