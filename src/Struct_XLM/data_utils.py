import ast
import csv
import inspect
import random

import torch
from torch.utils.data import TensorDataset
from transformers import BertTokenizer, XLMRobertaTokenizer

MODEL_CLASSES = {
    'mbert': BertTokenizer,
    'infoxlm': XLMRobertaTokenizer,
    'xlmr': XLMRobertaTokenizer
}


def auto_init_args(init):
    def new_init(self, *args, **kwargs):
        arg_dict = inspect.signature(init).parameters
        arg_names = list(arg_dict.keys())[1:]  # skip self
        proc_names = set()
        for name, arg in zip(arg_names, args):
            setattr(self, name, arg)
            proc_names.add(name)
        for name, arg in kwargs.items():
            setattr(self, name, arg)
            proc_names.add(name)
        remain_names = set(arg_names) - proc_names
        if len(remain_names):
            for name in remain_names:
                setattr(self, name, arg_dict[name].default)
        init(self, *args, **kwargs)

    return new_init


def read_parallel_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        all_data = []
        csv_reader = csv.reader(f)
        for line in csv_reader:
            if len(line) == 5:
                data_line = []
                for l in line:
                    if "[" in l and ']' in l:
                        l = ast.literal_eval(l)
                    data_line.append(l)
                all_data.append(data_line)
    f.close()
    return all_data


class data_holder:
    @auto_init_args
    def __init__(self, train_data, dev_data, test_data, vocab=None):
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        if vocab is not None:
            self.inv_vocab = {i: w for w, i in vocab.items()}
            

class DataProcessor:
    @auto_init_args
    def __init__(self, dp_train, dp_dev, dp_test, experiment):
        self.expe = experiment
        self.dp_train = dp_train
        self.dp_dev = dp_dev
        self.dp_test = dp_test

    def process(self):
        vocab = self._build_pretrain_vocab(self.expe.ml_type)
        train_dataset = self._data_to_idx_dp(self.dp_train, vocab)
        dev_dataset = self._data_to_idx_dp(self.dp_dev, vocab)
        test_dataset = self._data_to_idx_dp(self.dp_test, vocab)

        data = data_holder(train_data=train_dataset, dev_data=dev_dataset, test_data=test_dataset)

        return data

    def _data_to_idx_dp(self, datas, vocab):
        sent1_idx = []
        sent1_mask = []
        sent1_action = []

        sent2_idx = []
        sent2_mask = []
        sent2_action = []

        sent_id = []

        for data in datas:
            id = data[0]
            sent1 = data[1]
            action1 = data[3]
            sent2 = data[2]
            action2 = data[4]
            
            idx1, act1, mask1 = self._sent_to_idx(sent1, action1, vocab)
            idx2, act2, mask2 = self._sent_to_idx(sent2, action2, vocab)

            sent_id.append(int(id))
            sent1_idx.append(idx1)
            sent1_action.append(act1)
            sent1_mask.append(mask1)
            sent2_idx.append(idx2)
            sent2_action.append(act2)
            sent2_mask.append(mask2)

        sent_id_idx = torch.tensor(sent_id, dtype=torch.long)
        all_sent1_idx = torch.tensor([f for f in sent1_idx], dtype=torch.long)
        all_act1_idx = torch.tensor([f for f in sent1_action], dtype=torch.long)
        all_mask1_idx = torch.tensor([f for f in sent1_mask], dtype=torch.long)
        
        all_sent2_idx = torch.tensor([f for f in sent2_idx], dtype=torch.long)
        all_act2_idx = torch.tensor([f for f in sent2_action], dtype=torch.long)
        all_mask2_idx = torch.tensor([f for f in sent2_mask], dtype=torch.long)
        
        dataset = TensorDataset(
            all_sent1_idx, all_sent2_idx, all_mask1_idx, all_mask2_idx, all_act1_idx, all_act2_idx, sent_id_idx
        )
        return dataset

    def _sent_to_idx(self, sent, action, vocab):
        assert len(sent) == len(action)
        x = [vocab.convert_tokens_to_ids("[CLS]")]
        mask = [1]
        act = [1]

        for w, a in zip(sent, action):
            if a == 2:
                a = 0
            tokens = vocab.tokenize(w.lower()) if w not in ("[CLS]", "[SEP]") else [w]
            if tokens:
                xx = vocab.convert_tokens_to_ids(tokens)

                t = [a] + [-1] * (len(tokens) - 1)
                x.extend(xx)
                mask.extend([1 for _ in range(len(xx))])
                act.extend(t)

        x.append(vocab.convert_tokens_to_ids("[SEP]"))
        act.append(0)
        mask.append(0)
        
        if len(x) >= self.expe.max_len:
            print(len(x))
        while len(x) < self.expe.max_len:
            x.append(vocab.convert_tokens_to_ids("[PAD]"))
            act.append(0)
            mask.append(0)
        assert len(x) == len(act) == len(mask), "len(x)={}, len(y)={}, len(mask)={}".format(len(x), len(y), len(mask))
        return x, act, mask

    def _build_pretrain_vocab(self, lm_type):
        Token = MODEL_CLASSES[lm_type]
        vocab = Token.from_pretrained(self.expe.ml_model_path)
        return vocab


class DataProcessor_single:
    @auto_init_args
    def __init__(self, dp_train, dp_dev, dp_test, experiment):
        self.expe = experiment
        self.dp_train = dp_train
        self.dp_dev = dp_dev
        self.dp_test = dp_test

    def process(self):
        vocab = self._build_pretrain_vocab(self.expe.ml_type)
        train_dataset = self._data_to_idx_dp(self.dp_train, vocab)
        dev_dataset = self._data_to_idx_dp(self.dp_dev, vocab)
        test_dataset = self._data_to_idx_dp(self.dp_test, vocab)

        data = data_holder(train_data=train_dataset, dev_data=dev_dataset, test_data=test_dataset)

        return data

    def _data_to_idx_dp(self, datas, vocab):
        sent1_idx = []
        sent1_mask = []
        sent1_action = []
        label = []
        new_data = []
        for i in random.sample([i for i in range(len(datas))], int(len(datas)/3)):
            data = datas[i]
            new_data.append(data)
            sent1 = data[1]
            action1 = data[3]
            sent2 = data[2]
            action2 = data[4]

            idx1, act1, mask1 = self._sent_to_idx(sent1, action1, sent2, action2, vocab)
            label.append(1)
            sent1_idx.append(idx1)
            sent1_action.append(act1)
            sent1_mask.append(mask1)
        random.seed(44)
        for i in random.sample([i for i in range(len(datas))], int(len(datas)/3)):
            data = datas[i]
            id = data[0]
            sent1 = data[1]
            action1 = data[3]
            for i in random.sample([i for i in range(len(datas))], 5):
                rand_id = datas[i][0]
                if rand_id == id:
                    continue
                sent2 = datas[i][2]
                action2 = datas[i][4]

                idx1, act1, mask1 = self._sent_to_idx(sent1, action1, sent2, action2, vocab)
                label.append(0)
                sent1_idx.append(idx1)
                sent1_action.append(act1)
                sent1_mask.append(mask1)

        label = torch.tensor(label, dtype=torch.long)
        all_sent1_idx = torch.tensor([f for f in sent1_idx], dtype=torch.long)
        all_act1_idx = torch.tensor([f for f in sent1_action], dtype=torch.long)
        all_mask1_idx = torch.tensor([f for f in sent1_mask], dtype=torch.long)

        dataset = TensorDataset(
            all_sent1_idx, all_mask1_idx, all_act1_idx, label
        )
        return dataset

    def _sent_to_idx(self, sent1, action1, sent2, action2, vocab):
        assert len(sent1) == len(action1)
        assert len(sent2) == len(action2)
        x = [vocab.convert_tokens_to_ids("[CLS]")]
        mask = [1]
        act = [1]

        for w, a in zip(sent1, action1):
            if a == 2:
                a = 0
            tokens = vocab.tokenize(w.lower()) if w not in ("[CLS]", "[SEP]") else [w]
            if tokens:
                xx = vocab.convert_tokens_to_ids(tokens)

                t = [a] + [-1] * (len(tokens) - 1)
                x.extend(xx)
                mask.extend([1 for _ in range(len(xx))])
                act.extend(t)

        x.append(vocab.convert_tokens_to_ids("[SEP]"))
        act.append(1)
        mask.append(1)

        for w, a in zip(sent2, action2):
            if a == 2:
                a = 1
            tokens = vocab.tokenize(w.lower()) if w not in ("[CLS]", "[SEP]") else [w]
            if tokens:
                xx = vocab.convert_tokens_to_ids(tokens)

                t = [a] + [-1] * (len(tokens) - 1)
                x.extend(xx)
                mask.extend([1 for _ in range(len(xx))])
                act.extend(t)

        x.append(vocab.convert_tokens_to_ids("[SEP]"))
        act.append(1)
        mask.append(1)

        if len(x) >= self.expe.max_len:
            print(len(x))
        while len(x) < self.expe.max_len:
            x.append(vocab.convert_tokens_to_ids("[PAD]"))
            act.append(0)
            mask.append(0)
        assert len(x) == len(act) == len(mask), "len(x)={}, len(y)={}, len(mask)={}".format(len(x), len(y), len(mask))
        return x, act, mask

    def _build_pretrain_vocab(self, lm_type):
        Token = MODEL_CLASSES[lm_type]
        vocab = Token.from_pretrained(self.expe.ml_model_path)
        return vocab