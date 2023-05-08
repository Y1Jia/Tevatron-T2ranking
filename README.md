# Tevatron-T2ranking
使用Tevatron，在中文检索数据集T2ranking上训练、评估Dual Encoder。

# 安装环境

```bash
conda create -n tevatron python=3.7.0
cd tevatron
pip install --editable .
cd ..
pip install torch
conda install -c conda-forge faiss-gpu
git clone https://github.com/luyug/GradCache
cd GradCache
pip install .
```

# 数据集

使用[T2ranking](https://github.com/THUIR/T2Ranking)数据集。从[huggingface](https://huggingface.co/datasets/THUIR/T2Ranking)下载数据集，包含以下文件：

| Description                | Filename                  | Num Records | Format                            |
| -------------------------- | ------------------------- | ----------- | --------------------------------- |
| Collection                 | collection.tsv            | 2,303,643   | tsv: pid, passage                 |
| Queries Train              | queries.train.tsv         | 258,042     | tsv: qid, query                   |
| Queries Dev                | queries.dev.tsv           | 24,832      | tsv: qid, query                   |
| Queries Test               | queries.test.tsv          | 24,832      | tsv: qid, query                   |
| Qrels Train for re-ranking | qrels.train.tsv           | 1,613,421   | TREC qrels format (qid - pid rel) |
| Qrels Dev for re-ranking   | qrels.dev.tsv             | 400,536     | TREC qrels format                 |
| Qrels Retrieval Train      | qrels.retrieval.train.tsv | 744,663     | tsv: qid, pid                     |
| Qrels Retrieval Dev        | qrels.retrieval.dev.tsv   | 118,933     | tsv: qid, pid                     |
| BM25 Negatives             | train.bm25.tsv            | 200,359,731 | tsv: qid, pid, index              |
| Hard Negatives             | train.mined.tsv           | 200,376,001 | tsv: qid, pid, index, score       |

在训练Dual Encoder时，需要用到collection.tsv、queries.train.tsv、qrels.retrieval.train.tsv以及train.bm25.tsv（或train.mined.tsv）文件。从qrels.retrieval.train.tsv中读取正例，从train.bm25.tsv采样负例（若数量不够，则采样随机负例）。

Dataset的构造方法可以参考https://github.com/THUIR/T2Ranking/blob/main/src/dataset_factory.py#L199。

Tevatron使用的数据集格式如下图所示：

![tevatron raw dataset format](/photos/Tevatron-raw-data-template-for-IR-datasets.png)



通过create_dataset/create_DE_train_dataset.py构造符合上述格式的数据集。具体的做法是，对于train.bm25.tsv中的每个查询，从BM25排名前200的文档中采样30个作为负例（不足30个，采样随机负例），从qrels.retrieval.train.tsv中读取对应的正例；通过上述文件中读取的qid、pid，在queries.train.tsv、collection.tsv中读取对应的文本。

在encode时同样需要构造符合tevatron格式的数据集，encode query的数据集格式为 `{'query_id':<query_id>, "query":<query text>}`，corpus的数据集格式为`{'docid':<passage id>, "text": <passage text>}`。通过create_dataset/create_dev_dataset.py和create_dataset/create_corpus_dataset.py可构造上述数据集。

# 代码

使用[Tevatron](https://github.com/texttron/tevatron)，进行了少量修改，具体如下：

* 在datasets/dataset.py中，设置DEFAULT_PROCESSORS作为'json'的processor。
* 在arguments.py中的`class TevatronTrainingArguments`中增加`use_lamb`参数用于控制是否使用lamb optimizer。
* 在driver/train.py中，增加optimizers的设置（lamb optimizer），并将其传入trainer用于初始化。
* /utils/evaluate中，参考[t2ranking src/msmarco_eval.py](https://github.com/THUIR/T2Ranking/blob/main/src/msmarco_eval.py)，增加了评估部分代码。

# 训练和评估

训练超参数同t2ranking论文（除t2ranking src中使用了warm up，这里没有使用）。在t2ranking论文中使用的是8块A100（80GB），batch size=128。

通过script/train_dual_encoder.sh，实现训练和评估。

为了在显存受限情况下，复现128*8的batch size，可以使用GradCache，见script/train_dual_encoder_with_GradCache.sh

结果如下：

| batch size | checkpoint (epoch) | MRR@10 | recall@1 | recall@50 | recall@1000 |
| ---------- | ------------------ | ------ | -------- | --------- | ----------- |
| 128        | 10                 | 0.4697 | 0.0643   | 0.6364    | 0.8781      |
| 128        | 20                 | 0.4824 | 0.0667   | 0.6479    | 0.8835      |
| 1024       | 10                 | 0.5005 | 0.0690   | 0.6641    | 0.8829      |
| 1024       | 20                 | 0.5054 | 0.0697   | 0.6743    | 0.8899      |

