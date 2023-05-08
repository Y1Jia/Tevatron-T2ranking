'''
create t2ranking_corpus.jsonl for corpus encoding
'''
import pandas as pd
import os
import json
from tqdm import tqdm

''' args '''
collection_file = './collection.tsv'
output_file = './t2ranking_corpus.jsonl'

assert os.path.exists(output_file)==False, 'output file already exists!'
assert os.path.exists(collection_file)==True, 'collection file not found!'

''' load collection '''
collection = pd.read_csv(collection_file, sep='\t', header=0, names=['pid', 'para'], quoting=3)
collection = collection.fillna('NA')
collection.index = collection.pid
collection.pop('pid')
print("collection loaded")

''' create jsonl file '''
for index,row in tqdm(collection.iterrows()):
    pid = index
    para = row['para']
    data = {'docid': pid, 'text': para}
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')