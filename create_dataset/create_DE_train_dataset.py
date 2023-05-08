'''
create train dataset for Dual Encoder
sample negatives from top-200 BM25 or DE retrieved passages (train.bm25.tsv or train.mined.tsv)
get corresponding positives from qrels.retrieval.train.tsv
'''

import pandas as pd
import os
from tqdm import tqdm
import random
import json
import time

def sample_negs(qid, pids, num_negs, qrels, min_index, max_index):
    pids = [pid for pid in pids if pid not in qrels[qid]]   # exclude positives
    pids = pids[min_index: max_index]
    if len(pids) < num_negs:
        pad_num = num_negs - len(pids)
        pids += [random.randint(0, 2303643) for _ in range(pad_num)]  # pad with random pid
    sample_pids = random.sample(pids, num_negs)
    return sample_pids

def record_time(start, local_start):
    seconds = time.time() - local_start
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print("Time used: %d:%02d:%02d" % (h, m, s))
    seconds = time.time() - start
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print("Total time used: %d:%02d:%02d" % (h, m, s))
    local_start = time.time()
    return local_start

''' args '''
collection_file = './collection.tsv'
query_file = './queries.train.tsv'
qrels_file = './qrels.retrieval.train.tsv'
bm25_file = './train.bm25.tsv'
de_file = './train.mined.tsv'
bm25 = True # use BM25 hard negatives
min_index = 0   # 采样index起始位置
max_index = 200 # 采样index结束位置 （top200）
num_negs = 30 #采样30个负例
output_file = f"./DE_train_BM25_{min_index}_{max_index}_{num_negs}.jsonl" if bm25 \
    else f"./DE_train_DE_{min_index}_{max_index}_{num_negs}.jsonl"
assert num_negs <= max_index-min_index, 'num_negs should be less than max_index-min_index'
assert os.path.exists(output_file)==False, 'output file already exists!'

top1000_file = bm25_file if bm25 else de_file

''' check file existence '''
file_list = [collection_file, query_file, qrels_file, top1000_file]
for file in file_list:
    assert os.path.exists(file), 'File {} not found!'.format(file)


''' load query, collection, top1000, qrels  '''
# refer to dataset_factory.py in T2Ranking src:
# https://github.com/THUIR/T2Ranking/blob/main/src/dataset_factory.py#L83
start = time.time() # record time
local_start = time.time()

query = pd.read_csv(query_file, sep='\t', header=0, names=['qid', 'text'], quoting=3)
query.index = query.qid
query.pop('qid')
print("query loaded")
local_start = record_time(start, local_start)

collection = pd.read_csv(collection_file, sep='\t', header=0, names=['pid', 'para'], quoting=3)
collection = collection.fillna('NA')
collection.index = collection.pid
collection.pop('pid')
print("collection loaded")
local_start = record_time(start, local_start)

qrels = {}  # {qid: [pid1, pid2, ...]} 正例样本
with open(qrels_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        qid, pid = line.split()
        if qid == 'qid':
            continue
        qid = int(qid)
        pid = int(pid)
        x = qrels.get(qid, [])
        x.append(pid)
        qrels[qid] = x
print("qrels loaded")
local_start = record_time(start, local_start)

top1000 = pd.read_csv(top1000_file, sep='\t', header=0)
if len(top1000.columns) == 3:
    top1000.columns = ['qid', 'pid', 'index']
else:
    top1000.columns = ['qid', 'pid', 'index', 'score']
top1000 = list(top1000.groupby('qid'))
total_num = len(top1000)
print("top1000 loaded")
local_start = record_time(start, local_start)


''' create jsonl file '''
for i in tqdm(range(total_num)):
    cols = top1000[i]
    qid = cols[0]
    pids = list(cols[1]['pid'])
    sample_neg_pids = sample_negs(qid, pids, num_negs, qrels, min_index, max_index)
    pos_pids = qrels.get(qid)

    data = {}
    data['query_id'] = qid
    data['query'] = query.loc[qid]['text']
    positive_passages = []
    negative_passages = []
    for pid in pos_pids:
        psg = {}
        psg['docid'] = pid
        psg['text'] = collection.loc[pid]['para']
        positive_passages.append(psg)
    for pid in sample_neg_pids:
        psg = {}
        psg['docid'] = pid
        psg['text'] = collection.loc[pid]['para']
        negative_passages.append(psg)
    data['positive_passages'] = positive_passages
    data['negative_passages'] = negative_passages

    # write data to jsonl file 
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')    # ensure_ascii=False: write chinese characters
    # TODO: debug
    
local_start = record_time(start, local_start)
