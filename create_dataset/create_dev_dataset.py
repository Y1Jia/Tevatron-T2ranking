'''
create t2ranking_dev.jsonl for dev query decoding
'''
import pandas as pd
import os
import json
from tqdm import tqdm

''' args '''
query_file = './queries.dev.tsv'
output_file = './t2ranking_dev.jsonl'

assert os.path.exists(output_file)==False, 'output file already exists!'
assert os.path.exists(query_file)==True, 'query file not found!'

'''load query'''
query = pd.read_csv(query_file, sep='\t', header=0, names=['qid', 'text'], quoting=3)
print("query loaded")

'''create jsonl file'''
pbar = tqdm(total=len(query))
for i in range(len(query)):
    cols = query.loc[i]
    qid = cols['qid']
    query_text = cols['text']
    data = {}
    data['query_id'] = int(qid)
    data['query'] = query_text
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
    pbar.update(1)

pbar.close()




