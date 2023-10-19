# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors,
# The HuggingFace Inc. team, and The XTREME Benchmark Authors.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions for NER/POS tagging tasks."""

from __future__ import absolute_import, division, print_function

import logging
import os
import random
import sys
from io import open
from random import sample

from transformers import XLMTokenizer
import torch
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel
from transformers.models.xlm.modeling_xlm import XLMModel

sys.path.append("../src/MRC_Meta_tydi")
from SentenceTransformer import SentenceTransformer

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels, sentence, langs=None):
        """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      words: list. The words of the sequence.
      labels: (Optional) list. The labels for each word of the sequence. This should be
      specified for train and dev.txt examples, but not for test examples.
    """
        self.guid = guid
        self.words = words
        self.labels = labels
        self.langs = langs
        self.sentence = sentence


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, sentence, langs=None,
            representation=None,
            rankings_spt=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.langs = langs
        self.sentence = sentence
        self.representation = representation
        self.rankings_spt = rankings_spt


def read_examples_from_file(file_path, lang, lang2id=None):
    if not os.path.exists(file_path):
        logger.info("[Warming] file {} not exists".format(file_path))
        return []
    guid_index = 1
    examples = []
    subword_len_counter = 0
    if lang2id:
        lang_id = lang2id.get(lang, lang2id['en'])
    else:
        lang_id = 0
    logger.info("lang_id={}, lang={}, lang2id={}".format(lang_id, lang, lang2id))
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        langs = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if word:
                    if lang == 'zh':
                        sentence = ''.join(words)
                    else:
                        sentence = ' '.join(words)
                    examples.append(InputExample(guid="{}-{}".format(lang, guid_index),
                                                 words=words,
                                                 sentence=sentence,
                                                 labels=labels,
                                                 langs=langs))
                    guid_index += 1
                    words = []
                    labels = []
                    langs = []
                    subword_len_counter = 0
                else:
                    print(f'guid_index', guid_index, words, langs, labels, subword_len_counter)
            else:
                splits = line.split("\t")
                word = splits[0].replace(f'{lang}:', '')

                words.append(word)
                langs.append(lang_id)
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            if lang == 'zh':
                sentence = ''.join(words)
            else:
                sentence = ' '.join(words)
            examples.append(InputExample(guid="%s-%d".format(lang, guid_index),
                                         words=words,
                                         sentence=sentence,
                                         labels=labels,
                                         langs=langs))
    return examples


def get_seq_encoder(config):
    if 'bert' in config.model_type:
        return BertModel.from_pretrained(config.model_name_or_path)
    elif 'xlmr' in config.model_type or "infoxlm" in config.model_type:
        return XLMRobertaModel.from_pretrained(config.model_name_or_path)
    elif config.model_type == 'xlm':
        return XLMModel.from_pretrained(config.model_name_or_path)
    else:
        raise ValueError(config.model_type)


def compute_represenation(sents, config, sbert=False, NER_model=None):
    device = f'cuda:{config.gpu_id}'
    if sbert:
        model = SentenceTransformer("xlm-r-distilroberta-base-paraphrase-v1")
    elif NER_model:
        # model = TransformerQa(config)
        if config.model_type == 'bert':
            model = NER_model.bert
        elif 'xlmr' in config.model_type or "infoxlm" in config.model_type:
            model = NER_model.roberta
        elif config.model_type == 'xlm':
            model = NER_model.transformer
    else:
        model = get_seq_encoder(config)
    model.eval()
    model.to(device)
    batch_size = 16
    for i in range(0, len(sents), batch_size):
        items = sents[i: min(len(sents), i + batch_size)]
        if sbert:
            sentences = [item.sentence for item in items]
            layer_output = model.encode(sentences)
            attention=None
        else:
            with torch.no_grad():
                input_ids = torch.tensor([item.input_ids for item in items], dtype=torch.long).to(device)
                segment_ids = torch.tensor([item.segment_ids for item in items], dtype=torch.long).to(device)
                input_mask = torch.tensor([item.input_mask  for item in items], dtype=torch.long).to(device)
                if NER_model is None:
                    all_encoder_layers = model(input_ids, input_mask, segment_ids, output_attentions=True,
                                               output_hidden_states=False, return_dict=True)
                else:
                    all_encoder_layers = model(input_ids, input_mask, segment_ids)
            try:
                # layer_output = all_encoder_layers[1].detach().cpu().numpy()  # batch_size x target_size
                layer_output = all_encoder_layers['pooler_output'].detach().cpu().numpy()
                input_rep = all_encoder_layers['last_hidden_state'].detach().cpu().numpy()
                attention = all_encoder_layers['attentions'][-1]
                attention = torch.index_select(attention, 2, torch.tensor([0]).to('cuda:{}'.format(config.gpu_id))).squeeze(
                    2)
                attention = torch.mean(attention, dim=1).detach().cpu().numpy()
            except:
                attention = None
                input_rep = None
                layer_output = all_encoder_layers[0].detach().cpu().mean(axis=1, keepdim=False).numpy()
        for j, item in enumerate(items):
            item.representation = layer_output[j]
            if attention is not None:
                item.attention = attention[j]
                item.input_rep = input_rep[j]
        # item.representation = layer_output
        if i % (1000 * batch_size) == 0:
            logger.info('  Compute sentence representation. To {}...'.format(i))
    logger.info('  Finish.')
    model.to('cpu')
    del model


def convert_examples_to_features(examples,
                                 label_list,
                                 max_seq_length,
                                 tokenizer,
                                 config,
                                 is_training=False,
                                 cls_token_at_end=False,
                                 cls_token="[CLS]",
                                 cls_token_segment_id=1,
                                 sep_token="[SEP]",
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 pad_token_label_id=-1,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True,
                                 lang='en',
                                 spt_feature=None,
                                 sbert=False,
                                 ner_model=None,
                                 cl_model=None):
    """ Loads a data file into a list of `InputBatch`s
    `cls_token_at_end` define the location of the CLS token:
      - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
      - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
  """
    if is_training and lang != "en":
        random.seed(42)
        sample_id = sample([i for i in range(len(examples))], config.meta_num)
        new_example = [example for i, example in enumerate(examples) if i in sample_id]
        examples = new_example
        del new_example
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            if isinstance(tokenizer, XLMTokenizer):
                word_tokens = tokenizer.tokenize(word, lang=lang)
            else:
                word_tokens = tokenizer.tokenize(word)
            if len(word) != 0 and len(word_tokens) == 0:
                word_tokens = [tokenizer.unk_token]
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            print('truncate token', len(tokens), max_seq_length, special_tokens_count)
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            label_ids = label_ids[:(max_seq_length - special_tokens_count)]

            # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.

        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += ([pad_token] * padding_length)
            input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids += ([pad_token_segment_id] * padding_length)
            label_ids += ([pad_token_label_id] * padding_length)

        if example.langs and len(example.langs) > 0:
            langs = [example.langs[0]] * max_seq_length
        else:
            print('example.langs', example.langs, example.words, len(example.langs))
            print('ex_index', ex_index, len(examples))
            langs = None

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(langs) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
            logger.info("langs: {}".format(langs))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_ids=label_ids,
                          langs=langs,
                          sentence=example.sentence))

    return features


def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels
