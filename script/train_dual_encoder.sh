#!/bin/bash
lr=3e-5
epoch=100

# 以下不修改
output_dir="/output"
encode_dir=${output_dir}/"encode"
train_n_passages=2 # positive:negative = 1:1
q_max_len=32
p_max_len=256
batch_size=128
encode_batch_size=1024
search_batch_size=1000
depth=1000 # 检索top1000

# train
python -m tevatron.driver.train \
  --output_dir ${output_dir} \
  --model_name_or_path bert-base-chinese \
  --save_steps 15650 \
  --train_dir ./DE_train_BM25_0_200_30.jsonl \
  --fp16 \
  --per_device_train_batch_size ${batch_size} \
  --train_n_passages ${train_n_passages} \
  --learning_rate ${lr} \
  --q_max_len ${q_max_len} \
  --p_max_len ${p_max_len} \
  --num_train_epochs ${epoch} \
  --logging_steps 100 \
  --overwrite_output_dir \
  --untie_encoder True \
  --lr_scheduler_type constant \
  --use_lamb
  
# encode
mkdir ${encode_dir}
cp ${output_dir}/query_model/config.json ${output_dir}
python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path ${output_dir} \
  --fp16 \
  --per_device_eval_batch_size ${encode_batch_size} \
  --encode_in_path ./t2ranking_dev.jsonl \   
  --encoded_save_path ${encode_dir}/query_emb.pkl \
  --q_max_len ${q_max_len} \
  --encode_is_qry

for s in $(seq -f "%02g" 0 4)
do
python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path ${output_dir} \
  --fp16 \
  --per_device_eval_batch_size ${encode_batch_size} \
  --p_max_len ${p_max_len} \
  --encode_in_path ./t2ranking_corpus.jsonl \   
  --encoded_save_path ${encode_dir}/corpus_emb.${s}.pkl \
  --encode_num_shard 5 \
  --encode_shard_index ${s}
done

# search 
python -m tevatron.faiss_retriever \
--query_reps ${encode_dir}/query_emb.pkl \
--passage_reps ${encode_dir}/'corpus_emb.*.pkl' \
--depth ${depth} \
--batch_size ${search_batch_size} \
--save_text \
--save_ranking_to ${encode_dir}/rank.txt

# eval
python -m tevatron.utils.format.convert_result_to_marco \
              --input ${encode_dir}/rank.txt \
              --output ${encode_dir}/rank.txt.marco
              
python -m tevatron.utils.evaluate.calc_mrr \
 --path_to_reference ./qrels.retrieval.dev.tsv \
 --path_to_candidate ${encode_dir}/rank.txt.marco