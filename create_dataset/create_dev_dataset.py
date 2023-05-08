'''
create t2ranking_dev.jsonl for dev query decoding
'''
import pandas as pd
import os
import json

''' args '''
query_file = './queries.dev.tsv'
qrels_dev_file = './qrels.retrieval.dev.tsv'
output_file = './t2ranking_dev.jsonl'

assert os.path.exists(output_file)==False, 'output file already exists!'
assert os.path.exists(query_file)==True, 'query file not found!'
assert os.path.exists(qrels_dev_file)==True, 'qrels dev file not found!'

'''load query'''
query = pd.read_csv(query_file, sep='\t', header=0, names=['qid', 'text'], quoting=3)
query.index = query.qid
query.pop('qid')
print("query loaded")

'''load qrels_dev and create jsonl file'''
last_qid = -1
with open(qrels_dev_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        qid, pid = line.split()
        if qid == 'qid' or qid == last_qid:
            continue
        data = {}
        data['query_id'] = int(qid)
        data['query'] = query.loc[int(qid)]['text']
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
        last_qid = qid

