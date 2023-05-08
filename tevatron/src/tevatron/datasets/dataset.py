from datasets import load_dataset
from transformers import PreTrainedTokenizer
from .preprocessor import TrainPreProcessor, QueryPreProcessor, CorpusPreProcessor
from ..arguments import DataArguments

DEFAULT_PROCESSORS = [TrainPreProcessor, QueryPreProcessor, CorpusPreProcessor]
PROCESSOR_INFO = {
    'Tevatron/wikipedia-nq': DEFAULT_PROCESSORS,
    'Tevatron/wikipedia-trivia': DEFAULT_PROCESSORS,
    'Tevatron/wikipedia-curated': DEFAULT_PROCESSORS,
    'Tevatron/wikipedia-wq': DEFAULT_PROCESSORS,
    'Tevatron/wikipedia-squad': DEFAULT_PROCESSORS,
    'Tevatron/scifact': DEFAULT_PROCESSORS,
    'Tevatron/msmarco-passage': DEFAULT_PROCESSORS,
    # 'json': [None, None, None]
    'json': DEFAULT_PROCESSORS  # 使用custom dataset时，使用默认的preprocessor
}


class HFTrainDataset:
    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, cache_dir: str):
        data_files = data_args.train_path
        # set train_path the name of custom dataset if use custom dataset. 
        # at the same time, data_args.dataset_name will be 'json', see __post_init__ in arguments.py 
        if data_files: 
            data_files = {data_args.dataset_split: data_files}

        # refer to https://huggingface.co/docs/datasets/v2.11.0/en/package_reference/loading_methods#datasets.load_dataset
        self.dataset = load_dataset(data_args.dataset_name,
                                    data_args.dataset_language,
                                    data_files=data_files, cache_dir=cache_dir, use_auth_token=True)[data_args.dataset_split]
        
        # choose preprocessor from dict above (PROCESSOR_INFO)
        self.preprocessor = PROCESSOR_INFO[data_args.dataset_name][0] if data_args.dataset_name in PROCESSOR_INFO\
            else DEFAULT_PROCESSORS[0]
        self.tokenizer = tokenizer
        self.q_max_len = data_args.q_max_len
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.neg_num = data_args.train_n_passages - 1
        # seperator between 'title' and 'txt' when concate them. see datasets/preprocessor.py
        self.separator = getattr(self.tokenizer, data_args.passage_field_separator, data_args.passage_field_separator) 

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor(self.tokenizer, self.q_max_len, self.p_max_len, self.separator),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running tokenizer on train dataset",
            )
        return self.dataset


class HFQueryDataset:
    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, cache_dir: str):
        data_files = data_args.encode_in_path
        if data_files:
            data_files = {data_args.dataset_split: data_files}
        self.dataset = load_dataset(data_args.dataset_name,
                                    data_args.dataset_language,
                                    data_files=data_files, cache_dir=cache_dir, use_auth_token=True)[data_args.dataset_split]
        self.preprocessor = PROCESSOR_INFO[data_args.dataset_name][1] if data_args.dataset_name in PROCESSOR_INFO \
            else DEFAULT_PROCESSORS[1]
        self.tokenizer = tokenizer
        self.q_max_len = data_args.q_max_len
        self.proc_num = data_args.dataset_proc_num

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor(self.tokenizer, self.q_max_len),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running tokenization",
            )
        return self.dataset


class HFCorpusDataset:
    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, cache_dir: str):
        data_files = data_args.encode_in_path
        if data_files:
            data_files = {data_args.dataset_split: data_files}
        self.dataset = load_dataset(data_args.dataset_name,
                                    data_args.dataset_language,
                                    data_files=data_files, cache_dir=cache_dir, use_auth_token=True)[data_args.dataset_split]
        script_prefix = data_args.dataset_name
        if script_prefix.endswith('-corpus'):
            script_prefix = script_prefix[:-7]
        self.preprocessor = PROCESSOR_INFO[script_prefix][2] \
            if script_prefix in PROCESSOR_INFO else DEFAULT_PROCESSORS[2]
        self.tokenizer = tokenizer
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.separator = getattr(self.tokenizer, data_args.passage_field_separator, data_args.passage_field_separator)

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor(self.tokenizer, self.p_max_len, self.separator),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running tokenization",
            )
        return self.dataset
