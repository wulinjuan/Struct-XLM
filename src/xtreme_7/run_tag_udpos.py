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
"""Fine-tuning models for NER and POS tagging."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import numpy
import torch
# from seqeval.metrics import precision_score, recall_score, f1_score
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import sys
sys.path.append("/data/home10b/wlj/code/Struct-XLM/src/Struct_XLM/")
from BertEncoder import BertModel
from RobertaEncoder import RobertaModel
from actor import Actor
from RobertaForTokenClassification import RobertaForTokenClassification

from seqeval.metrics.sequence_labeling import precision_score, recall_score, f1_score
from utils_tag import convert_examples_to_features
from utils_tag import get_labels
from utils_tag import read_examples_from_file

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    WEIGHTS_NAME,
    BertConfig,
    BertTokenizer,
    BertForTokenClassification,
    XLMConfig,
    XLMTokenizer,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    XLMRobertaForTokenClassification
)
from xlm import XLMForTokenClassification

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer, BertModel),
    "xlm": (XLMConfig, XLMForTokenClassification, XLMTokenizer, RobertaModel),
    "xlmr": (XLMRobertaConfig, RobertaForTokenClassification, XLMRobertaTokenizer, RobertaModel),
}

emb_config = {
    'mbert': 768,
    'xlmr': 1024,
    'infoxlm': 1024
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def Sampling_RL(args, predicted, mask, head_mask, epsilon, Random=True):
    actor_attention = torch.zeros(args.max_len, args.max_len)
    actions = []
    length = torch.sum(mask).item()
    constituent_num = 0  # (1, seq_len, 2)
    length_true = torch.sum(head_mask).item()
    for pos in range(length):
        if Random:
            if random.random() > epsilon:
                action = (0 if random.random() < float(predicted[pos][0].item()) else 1)
            else:
                action = (1 if random.random() < float(predicted[pos][0].item()) else 0)
        else:
            action = np.argmax(predicted[pos].cpu().detach().numpy()).item()
        if action == 1 and head_mask[pos] == 1:
            constituent_num += 1
        actions.append(action)
    pos = 1
    end_one = False
    while pos < length:
        # actor_attention[pos][pos] = 1
        if actions[pos] == 0:
            begin_sub = pos
            for i in range(pos + 1, length):
                if head_mask[i] == -1 and i != pos + 1:
                    continue
                elif head_mask[i] == -1 and i == begin_sub + 1:
                    begin_sub += 1
                actor_attention[pos][i] = 1
                actor_attention[i][pos] = 1
                if actions[i] == 1:
                    end_one = True
                    break
        elif actions[pos] == 1:
            if end_one:
                end_one = False
                if pos + 1 < length and actions[pos + 1] == -1:
                    for i in range(pos + 1, length):
                        actor_attention[pos][i] = 1
                        actor_attention[i][pos] = 1
                        if i + 1 < length and actions[i + 1] != -1:
                            break
                continue
            else:
                actor_attention[pos][pos] = 1
        else:
            if pos + 1 < length and head_mask[pos + 1] != -1:
                pos += 1
                continue
            for i in range(pos + 1, length):
                actor_attention[pos][i] = 1
                actor_attention[i][pos] = 1
                if i + 1 < length and head_mask[i + 1] != -1:
                    break
        pos += 1
    if sum(actions) > length or sum(actions) == 0:
        Rinput = actor_attention.masked_fill(actor_attention == 0, 1)
    else:
        Rinput = actor_attention.masked_fill(actor_attention == 0, -1e9)

    Rinput = Rinput.unsqueeze(0).unsqueeze(0).cuda()  # (1, seq_len, seq_len)
    return actions, Rinput, constituent_num/length_true if constituent_num!=0 else 1


def train(args, train_dataset, model, actorModel, tokenizer, labels, pad_token_label_id, lang2id=None):
    """Train the model."""
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    # model.to('cuda:2')
    actorModel.eval()
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=t_total * args.warmup_steps,
                                                num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    best_score = 0.0
    best_checkpoint = None
    patience = 0
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Add here for reproductibility (even between python 2 and 3)

    meta_train_error = 0.0
    step = 0
    max_f1 = 0

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch if t is not None)
            text = batch[0]
            mask = batch[1]
            segment = batch[2]
            label = batch[3]

            head_mask = mask.masked_fill(label == -1, 0).long()

            context = model(input_ids=text, attention_mask=mask, get_embedding=True)['last_hidden_state']
            predicted = actorModel.get_target_output(context, mask)
            # loss_qry_all = 0.0
            sentence_actor_attention = None
            if args.action:
                constituent_num_total = 0
                for j in range(text.size()[0]):  # bachsize
                    p = predicted[j]
                    m = mask[j]
                    h = head_mask[j]
                    actions, Rinput, constituent_num = Sampling_RL(args, p, m, h, args.epsilon, Random=False)
                    constituent_num_total += constituent_num
                    if j == 0:
                        sentence_actor_attention = Rinput
                    else:
                        sentence_actor_attention = torch.cat([sentence_actor_attention, Rinput])
                constituent_num_total /= args.train_batch_size
            # On the query data
            loss = model(input_ids=text, attention_mask=mask, token_type_ids=segment, labels=label,
                         action=sentence_actor_attention)[0].mean()
            # loss_qry_all += loss_qry
            # loss += (1 * constituent_num_total + 0.1 / constituent_num_total - 0.6) * 0.1 * args.grained
            loss.backward()

            meta_train_error += loss.item()

            # loss_qry_all.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            global_step += 1

            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                # Log metrics
                if args.local_rank == -1 and args.evaluate_during_training:
                    # Only evaluate on single GPU otherwise metrics may not average well
                    results, _ = evaluate(args, model, actorModel, tokenizer, labels, pad_token_label_id, mode="dev",
                                          lang=args.train_langs, lang2id=lang2id)
                    for key, value in results.items():
                        tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                logging_loss = tr_loss

            if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                if args.save_only_best_checkpoint:
                    result, _ = evaluate(args, model, actorModel, tokenizer, labels, pad_token_label_id, mode="dev",
                                         prefix=global_step, lang=args.train_langs, lang2id=lang2id)
                    if result["f1"] > best_score:
                        logger.info("result['f1']={} > best_score={}".format(result["f1"], best_score))
                        best_score = result["f1"]
                        # Save the best model checkpoint
                        output_dir = os.path.join(args.output_dir, "checkpoint-best" + (
                            "" if args.train_langs == 'en' else f"_{args.train_langs}"))
                        best_checkpoint = output_dir
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # Take care of distributed/parallel training
                        model_to_save = model.module if hasattr(model, "module") else model
                        model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving the best model checkpoint to %s", output_dir)
                        logger.info("Reset patience to 0")
                        patience = 0
                    else:
                        patience += 1
                        logger.info("Hit patience={}".format(patience))
                        if args.eval_patience > 0 and patience > args.eval_patience:
                            logger.info("early stop! patience={}".format(patience))
                            epoch_iterator.close()
                            train_iterator.close()
                            if args.local_rank in [-1, 0]:
                                tb_writer.close()
                            return global_step, tr_loss / global_step
                else:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step) + (
                        "" if args.train_langs == 'en' else f"_{args.train_langs}"))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
            # Save the best model checkpoint
        output_dir = os.path.join(args.output_dir,
                                  "checkpoint-final" + ("" if args.train_langs == 'en' else f"_{args.train_langs}"))
        # best_checkpoint = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving the final model checkpoint to %s", output_dir)
    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, actorModel, tokenizer, labels, pad_token_label_id, mode, prefix="", lang="en", lang2id=None,
             print_result=True):
    global preds_list
    eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode, lang=lang,
                                           lang2id=lang2id)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    # if args.n_gpu > 1:
    #  model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation %s in %s *****" % (prefix, lang))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    actorModel.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            text = batch[0]
            mask = batch[1]
            segment = batch[2]
            label = batch[3]
            head_mask = mask.masked_fill(label == -1, 0).long()

            context = model(input_ids=text, attention_mask=mask, get_embedding=True)['last_hidden_state']
            predicted = actorModel.get_target_output(context, mask)
            # loss_qry_all = 0.0
            sentence_actor_attention = None
            for j in range(text.size()[0]):  # bachsize
                p = predicted[j]
                m = mask[j]
                h = head_mask[j]
                actions, Rinput, constituent_num = Sampling_RL(args, p, m, h, args.epsilon, Random=False)
                if j == 0:
                    sentence_actor_attention = Rinput
                else:
                    sentence_actor_attention = torch.cat([sentence_actor_attention, Rinput])
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                segment = batch[2] if args.model_type in ["bert", "xlnet"] else None

            outputs = model(input_ids=text, attention_mask=mask, token_type_ids=segment, labels=label,
                            action=sentence_actor_attention)
            tmp_eval_loss, logits = outputs[:2]

            if args.n_gpu > 1:
                # mean() to average on multi-gpu parallel evaluating
                tmp_eval_loss = tmp_eval_loss.mean()

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = label.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, label.detach().cpu().numpy(), axis=0)

    if nb_eval_steps == 0:
        results = {k: 0 for k in ["loss", "precision", "recall", "f1"]}
        preds_list = [[] for _ in range(out_label_ids.shape[0])]
    else:
        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=2)

        label_map = {i: label for i, label in enumerate(labels)}

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        results = {
            "loss": eval_loss,
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list)
        }

    if print_result:
        logger.info("***** Evaluation result %s in %s *****" % (prefix, lang))
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

    return results, preds_list


def build_support_features(base_features, target_features=None, support_size=2, cl_model=None):
    if target_features == None:
        target_features = base_features
    TOP_K = support_size
    # pdist = nn.PairwiseDistance(p=2)
    target_reprs = np.stack([numpy.array(item.representation) for item in target_features])
    base_reprs = np.stack([numpy.array(item.representation) for item in base_features])  # sample_num x feature_dim

    # compute pairwise cosine distance
    dis = np.matmul(target_reprs, base_reprs.T)  # target_num x base_num

    base_norm = np.linalg.norm(base_reprs, axis=1)  # base_num
    base_norm = np.stack([base_norm] * len(target_features), axis=0)  # target_num x base_num

    dis = dis / base_norm  # target_num x base_num
    relevance = np.argsort(dis, axis=1)
    if cl_model is not None:
        support_size = 64
    for i, item in tqdm(enumerate(target_features), desc='[get meta data]'):
        chosen_ids = relevance[i][-1 * (support_size + 1): -1]
        # logger.info('  Support set info: {}: {}'.format(i, ', '.join([str(id) for id in chosen_ids])))
        # support = [base_features[id] for id in chosen_ids]
        # support_set.append(support)
        if cl_model is not None:
            chosen_dict = {}
            target_features[i].meta_set_num = []
            target_representation = torch.tensor([target_features[i].input_rep], dtype=torch.float).repeat(support_size,
                                                                                                           1, 1).to(
                cl_model.device)
            target_attention = torch.tensor([target_features[i].attention], dtype=torch.float).repeat(support_size,
                                                                                                      1).to(
                cl_model.device)
            target_latent_representation = torch.matmul(target_representation, cl_model.probe.proj)

            chosen_representation = torch.tensor([base_features[id].input_rep for id in chosen_ids],
                                                 dtype=torch.float).to(cl_model.device)
            chosen_attention = torch.tensor([base_features[id].attention for id in chosen_ids], dtype=torch.float).to(
                cl_model.device)
            chosen_latent_representation = torch.matmul(chosen_representation, cl_model.probe.proj)
            score = cl_model.get_wmd(target_attention, chosen_attention, target_latent_representation,
                                     chosen_latent_representation).tolist()
            for i_, id in enumerate(chosen_ids):
                if id not in chosen_dict:
                    chosen_dict[id] = score[i_]
            chosen_dict = sorted(chosen_dict.items(), key=lambda x: x[1])
            for pos in chosen_dict[:TOP_K]:
                pos = int(pos[0])
                try:
                    target_features[i].rankings_spt.append(pos)
                except:
                    target_features[i].rankings_spt = [pos]
            target_features[i].rankings_spt = np.array(target_features[i].rankings_spt)
        else:
            target_features[i].rankings_spt = np.array(chosen_ids)
    return target_features


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode, lang, lang2id=None, few_shot=-1):
    # Make sure only the first process in distributed training process
    # the dataset, and the others will use the cache
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, "cached_{}_{}_{}_{}".format(mode, lang,
                                                                                   list(filter(None,
                                                                                               args.model_name_or_path.split(
                                                                                                   "/"))).pop(),
                                                                                   str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        langs = lang.split(',')
        logger.info("all languages = {}".format(lang))
        features = []
        for lg in langs:
            if 'pos' in args.data_dir:
                data_file = os.path.join(args.data_dir, "{}-{}.tsv".format(mode, lg))
            else:
                data_file = os.path.join(args.data_dir, lg, "{}".format(mode if lg == 'en' else 'test'))
            logger.info("Creating features from dataset file at {} in language {}".format(data_file, lg))
            examples = read_examples_from_file(data_file, lg, lang2id)
            features_lg = convert_examples_to_features(examples, labels, args.max_seq_length, tokenizer, args,
                                                       is_training=(mode == 'train'),
                                                       cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                       cls_token=tokenizer.cls_token,
                                                       cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                       sep_token=tokenizer.sep_token,
                                                       sep_token_extra=bool(args.model_type in ["roberta", "xlmr"]),
                                                       pad_on_left=bool(args.model_type in ["xlnet"]),
                                                       pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[
                                                           0],
                                                       pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                                                       pad_token_label_id=pad_token_label_id,
                                                       lang=lg
                                                       )
            features.extend(features_lg)
        if args.local_rank in [-1, 0]:
            logger.info(
                "Saving features into cached file {}, len(features)={}".format(cached_features_file, len(features)))
            torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process
    # the dataset, and the others will use the cache
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()

    if few_shot > 0 and mode == 'train':
        logger.info("Original no. of examples = {}".format(len(features)))
        features = features[: few_shot]
        logger.info('Using few-shot learning on {} examples'.format(len(features)))

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    if args.model_type == 'xlm' and features[0].langs is not None:
        all_langs = torch.tensor([f.langs for f in features], dtype=torch.long)
        logger.info('all_langs[0] = {}'.format(all_langs[0]))
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_langs)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def load_and_cache_examples_meta(args, tokenizer, labels, pad_token_label_id, mode, lang,
                                 lang2id=None, few_shot=-1, sbert=True, ner_model=None, cl_model=None):
    # Make sure only the first process in distributed training process
    # the dataset, and the others will use the cache
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, "cached_{}_{}_{}_{}".format(mode, 'en_task',
                                                                                   list(filter(None,
                                                                                               args.model_name_or_path.split(
                                                                                                   "/"))).pop(),
                                                                                   str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        spt_features = torch.load(cached_features_file)
    else:
        logger.info("all languages = {}".format(lang))
        spt_features = []
        data_file = os.path.join(args.data_dir, 'en', "train")
        logger.info("Creating features from dataset file at {} in language {}".format(data_file, 'en'))
        examples = read_examples_from_file(data_file, 'en', lang2id)
        spt_features = convert_examples_to_features(examples, labels, args.max_seq_length, tokenizer, args,
                                                    is_training=(mode == 'train'),
                                                    cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                    cls_token=tokenizer.cls_token,
                                                    cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                    sep_token=tokenizer.sep_token,
                                                    sep_token_extra=bool(args.model_type in ["roberta", "xlmr"]),
                                                    pad_on_left=bool(args.model_type in ["xlnet"]),
                                                    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[
                                                        0],
                                                    pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                                                    pad_token_label_id=pad_token_label_id,
                                                    lang='en',
                                                    sbert=True)

    langs = lang.split(',')
    logger.info("all languages = {}".format(lang))
    features = []
    for lg in langs:
        data_file = os.path.join(args.data_dir, lg, "{}".format(mode if lg == 'en' else 'dev'))
        logger.info("Creating features from dataset file at {} in language {}".format(data_file, lg))
        examples = read_examples_from_file(data_file, lg, lang2id)
        features_lg = convert_examples_to_features(examples, labels, args.max_seq_length, tokenizer, args,
                                                   is_training=(mode == 'train'),
                                                   cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                   cls_token=tokenizer.cls_token,
                                                   cls_token_segment_id=2 if args.model_type in [
                                                       "xlnet"] else 0,
                                                   sep_token=tokenizer.sep_token,
                                                   sep_token_extra=bool(args.model_type in ["roberta", "xlmr"]),
                                                   pad_on_left=bool(args.model_type in ["xlnet"]),
                                                   pad_token=
                                                   tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[
                                                       0],
                                                   pad_token_segment_id=4 if args.model_type in [
                                                       "xlnet"] else 0,
                                                   pad_token_label_id=pad_token_label_id,
                                                   lang=lg,
                                                   spt_feature=spt_features,
                                                   sbert=sbert,
                                                   ner_model=ner_model,
                                                   cl_model=cl_model
                                                   )
        features.extend(features_lg)
        if args.local_rank in [-1, 0]:
            logger.info(
                "Saving features into cached file {}, len(features)={}".format(cached_features_file, len(features)))
            torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process
    # the dataset, and the others will use the cache
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()

    if few_shot > 0 and mode == 'train':
        logger.info("Original no. of examples = {}".format(len(features)))
        features = features[: few_shot]
        logger.info('Using few-shot learning on {} examples'.format(len(features)))

    qry_features = build_support_features(spt_features, target_features=features, support_size=2, cl_model=None)
    l_input_ids_s = []
    l_attention_masks_s = []
    l_token_type_ids_s = []
    l_label_ids_s = []

    l_input_ids_q = []
    l_attention_masks_q = []
    l_token_type_ids_q = []
    l_label_ids_q = []

    random.shuffle(qry_features)
    # 出现频率过大的样本，限制在一定的概率出现
    dic_num = {}
    for index in range(len(qry_features)):
        spt = qry_features[index].rankings_spt[0]
        if spt not in dic_num.keys():
            dic_num[spt] = 1
        else:
            dic_num[spt] += 1
    dic_num = sorted(dic_num.items(), key=lambda d: d[1], reverse=True)
    print(dic_num)
    print(len(dic_num))

    s = 0
    print("META Batching")
    for _ in tqdm(range(args.batch_sz)):
        l_input_ids_spt = []
        l_attention_masks_spt = []
        l_token_type_ids_spt = []
        l_all_label_ids_spt = []

        l_input_ids_qry = []
        l_attention_masks_qry = []
        l_token_type_ids_qry = []
        l_all_label_ids_qry = []

        # Pick q_qry from qry examples randomly
        if s + args.q_qry < len(qry_features):
            qry_indices = range(s, s + args.q_qry)
            s = s + args.q_qry
        elif s < len(qry_features):
            t = s + args.q_qry - len(qry_features)
            qry_indices = list(range(s, len(qry_features))) + list(range(0, t))
            s = t
        else:
            s = 0
            qry_indices = range(s, len(qry_features))

        all_feature_index = torch.tensor(qry_indices, dtype=torch.long)
        # l_feature_index.append(all_feature_index)

        # random.sample(len(qry_examples), k=data_args.q_qry)

        for index in qry_indices:
            l_input_ids_qry.append(qry_features[index].input_ids)
            l_attention_masks_qry.append(qry_features[index].input_mask)
            l_token_type_ids_qry.append(qry_features[index].segment_ids)
            l_all_label_ids_qry.append(qry_features[index].label_ids)

            spt_indices = qry_features[index].rankings_spt[:args.meta_example_num]

            for spt_index in spt_indices:
                l_input_ids_spt.append(spt_features[spt_index].input_ids)
                l_attention_masks_spt.append(spt_features[spt_index].input_mask)
                l_token_type_ids_spt.append(spt_features[spt_index].segment_ids)
                l_all_label_ids_spt.append(spt_features[spt_index].label_ids)

        all_input_ids = torch.tensor(l_input_ids_spt, dtype=torch.long)
        l_input_ids_s.append(all_input_ids)
        l_attention_masks_s.append(torch.tensor(l_attention_masks_spt, dtype=torch.long))
        l_token_type_ids_s.append(torch.tensor(l_token_type_ids_spt, dtype=torch.long))
        l_label_ids_s.append(torch.tensor(l_all_label_ids_spt, dtype=torch.long))

        all_input_ids = torch.tensor(l_input_ids_qry, dtype=torch.long)
        l_input_ids_q.append(all_input_ids)
        l_attention_masks_q.append(torch.tensor(l_attention_masks_qry, dtype=torch.long))
        l_token_type_ids_q.append(torch.tensor(l_token_type_ids_qry, dtype=torch.long))
        l_label_ids_q.append(torch.tensor(l_all_label_ids_qry, dtype=torch.long))

    # Convert to Tensors and build dataset
    if args.model_type == 'xlm' and features[0].langs is not None:
        all_langs = torch.tensor([f.langs for f in features], dtype=torch.long)
        logger.info('all_langs[0] = {}'.format(all_langs[0]))
        dataset = TensorDataset(torch.stack(l_input_ids_s), torch.stack(l_attention_masks_s),
                                torch.stack(l_token_type_ids_s), torch.stack(l_label_ids_s),
                                torch.stack(l_input_ids_q), torch.stack(l_attention_masks_q),
                                torch.stack(l_token_type_ids_q), torch.stack(l_label_ids_q),
                                all_langs)
    else:
        dataset = TensorDataset(torch.stack(l_input_ids_s), torch.stack(l_attention_masks_s),
                                torch.stack(l_token_type_ids_s), torch.stack(l_label_ids_s),
                                torch.stack(l_input_ids_q), torch.stack(l_attention_masks_q),
                                torch.stack(l_token_type_ids_q), torch.stack(l_label_ids_q))
    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default='/data/home10b/wlj/code/Struct-XLM/src/xtreme_7/data/udpos/', type=str,
                        help="The input data dir. Should contain the training files for the NER/POS task.")
    parser.add_argument("--model_type", default='xlmr', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default='/data/home10b/wlj/cached_models/xlm-roberta-large',
                        type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: " + "bert-base-multilingual-cased, xlm-roberta-large")
    parser.add_argument("--output_dir", default='output/pos/', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--critic_output_dir", default="/data/home10b/wlj/code/Struct-XLM/src/Struct_XLM/output_new/checkpoint-best", type=str)
    parser.add_argument("--action", default=True)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--act_dropout", default=0.0, type=float)
    parser.add_argument("--margin", default=0.5, type=float)
    parser.add_argument("--s", default=1, type=float)
    parser.add_argument("--sentence_type", default='mean', type=str, help="mean, pool")
    parser.add_argument("--epsilon", type=float, default=0.05)

    ## Other parameters
    parser.add_argument("--labels", default="/data/home10b/wlj/code/Struct-XLM/src/xtreme_7/data/udpos/label.txt", type=str,
                        help="Path to a file containing all labels. If not specified, NER/POS labels are used.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev.txt set.")
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--do_predict_dev", action="store_true",
                        help="Whether to run predictions on the dev.txt set.")
    parser.add_argument("--init_checkpoint",
                        default='', type=str,
                        help="initial checkpoint for train/predict")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--few_shot", default=-1, type=int,
                        help="num of few-shot exampes")

    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0.1, type=float,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--save_only_best_checkpoint", action="store_true",
                        help="Save only the best checkpoint during training")
    parser.add_argument("--eval_all_checkpoints", action="store_true",
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", default=True,
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--predict_langs", type=str,
                        default="en,af,ar,bg,de,el,es,et,eu,fa,fi,fr,he,hi,hu,id,it,ja,kk,ko,mr,nl,pt,ru,ta,te,th,tl,tr,ur,vi,yo,zh",
                        help="prediction languages")
    parser.add_argument("--train_langs", default="en", type=str,
                        help="The languages in the training sets.")
    parser.add_argument("--log_file", type=str, default='output/log.txt', help="log file")
    parser.add_argument("--eval_patience", type=int, default=-1,
                        help="wait N times of decreasing dev.txt score before early stop during training")
    args = parser.parse_args()

    args.edim = emb_config[args.model_type]
    args.ml_model_path = args.model_name_or_path
    args.max_len = args.max_seq_length
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        # Initializes the distributed backend which sychronizes nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()],
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logging.info("Input args: %r" % args)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare NER/POS task
    labels = get_labels(args.labels)
    num_labels = len(labels)
    # Use cross entropy ignore index as padding label id
    # so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    # Make sure only the first process in distributed training loads model/vocab
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class, base_model = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)

    if args.init_checkpoint:
        logger.info("loading from init_checkpoint={}".format(args.init_checkpoint))
        model = model_class.from_pretrained(args.init_checkpoint,
                                            args=args,
                                            config_class=config_class,
                                            model_class=base_model,
                                            config=config,
                                            cache_dir=args.init_checkpoint)
    else:
        logger.info("loading from cached model = {}".format(args.model_name_or_path))
        model = model_class.from_pretrained(args.model_name_or_path,
                                            args=args,
                                            config_class=config_class,
                                            model_class=base_model,
                                            from_tf=bool(".ckpt" in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
    lang2id = config.lang2id if args.model_type == "xlm" else None
    logger.info("Using lang2id = {}".format(lang2id))

    # Make sure only the first process in distributed training loads model/vocab
    if args.local_rank == 0:
        torch.distributed.barrier()
    model.to(args.device)
    output_dir = args.critic_output_dir

    actorModel = Actor(expirement=args, d_model=args.edim, dropout=args.act_dropout, no_cuda=args.no_cuda)
    actorModel.target_policy.load_state_dict(torch.load(os.path.join(output_dir, "target_model_actor.bin")))
    model.encoder.load_state_dict(torch.load(os.path.join(output_dir, "target_model_critic.bin")))
    actorModel.to(args.device)
    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train",
                                                lang=args.train_langs, lang2id=lang2id, few_shot=args.few_shot)
        global_step, tr_loss = train(args, train_dataset, model, actorModel, tokenizer, labels, pad_token_label_id,
                                     lang2id)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use default names for the model,
    # you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        # Save model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Initialization for evaluation
    results = {}
    if os.path.exists(os.path.join(args.output_dir,
                                   'checkpoint-final' + ("" if args.train_langs == 'en' else f"_{args.train_langs}"))):
        best_checkpoint = os.path.join(args.output_dir, 'checkpoint-final' + (
            "" if args.train_langs == 'en' else f"_{args.train_langs}"))
    elif os.path.exists(os.path.join(args.output_dir,
                                     'checkpoint-best' + (
                                     "" if args.train_langs == 'en' else f"_{args.train_langs}"))):
        best_checkpoint = os.path.join(args.output_dir,
                                       'checkpoint-best' + (
                                           "" if args.train_langs == 'en' else f"_{args.train_langs}"))
    else:
        best_checkpoint = args.output_dir
    best_f1 = 0

    # Evaluation
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint,
                                                args=args,
                                                config_class=config_class,
                                                model_class=base_model)
            model.to(args.device)
            result, _ = evaluate(args, model, actorModel, tokenizer, labels, pad_token_label_id, mode="dev",
                                 prefix=global_step,
                                 lang=args.train_langs, lang2id=lang2id)
            if result["f1"] > best_f1:
                best_checkpoint = checkpoint
                best_f1 = result["f1"]
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))
            writer.write("best checkpoint = {}, best f1 = {}\n".format(best_checkpoint, best_f1))

    # Prediction
    all_f1 = ""
    result_file = open("output/udpos_results.txt", "a+", encoding='utf-8')
    if args.do_predict and args.local_rank in [-1, 0]:
        logger.info("Loading the best checkpoint from {}\n".format(best_checkpoint))
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(best_checkpoint,
                                            args=args,
                                            config_class=config_class,
                                            model_class=base_model)
        model.to(args.device)

        output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
        AVG_F1 = 0
        I = 0
        with open(output_test_results_file, "a") as result_writer:
            for lang in args.predict_langs.split(','):
                if not os.path.exists(os.path.join(args.data_dir, 'test-{}.tsv'.format(lang))):
                    logger.info("Language {} does not exist".format(lang))
                    continue
                result, predictions = evaluate(args, model, actorModel, tokenizer, labels, pad_token_label_id,
                                               mode="test",
                                               lang=lang, lang2id=lang2id)

                # Save results
                result_writer.write("=====================\nlanguage={}\n".format(lang))
                for key in sorted(result.keys()):
                    result_writer.write("{} = {}\n".format(key, str(result[key])))
                all_f1 += "%.1f " % (result['precision'] * 100)
                AVG_F1 += result['precision']
                I += 1
            all_f1 += "%.1f " % (AVG_F1 / I * 100)
            # Save predictions
            # output_test_predictions_file = os.path.join(args.output_dir, "test_{}_predictions.txt".format(lang))
            # infile = os.path.join(args.data_dir, lang, "test")
            # idxfile = infile + '.idx'
            # save_predictions(args, predictions, output_test_predictions_file, infile, idxfile)
            print(all_f1, file=result_file)
        if os.path.exists(os.path.join(args.output_dir,
                                       'checkpoint-best' + (
                                               "" if args.train_langs == 'en' else f"_{args.train_langs}"))):
            best_checkpoint = os.path.join(args.output_dir,
                                           'checkpoint-best' + (
                                               "" if args.train_langs == 'en' else f"_{args.train_langs}"))
            all_f1 = ""
            model = model_class.from_pretrained(best_checkpoint,
                                                args=args,
                                                config_class=config_class,
                                                model_class=base_model)
            model.to(args.device)

            output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
            AVG_F1 = 0
            I = 0
            with open(output_test_results_file, "a") as result_writer:
                for lang in args.predict_langs.split(','):
                    if not os.path.exists(os.path.join(args.data_dir, 'test-{}.tsv'.format(lang))):
                        logger.info("Language {} does not exist".format(lang))
                        continue
                    result, predictions = evaluate(args, model, actorModel, tokenizer, labels, pad_token_label_id,
                                                   mode="test",
                                                   lang=lang, lang2id=lang2id)

                    # Save results
                    result_writer.write("=====================\nlanguage={}\n".format(lang))
                    for key in sorted(result.keys()):
                        result_writer.write("{} = {}\n".format(key, str(result[key])))
                    all_f1 += "%.1f " % (result['precision'] * 100)
                    AVG_F1 += result['precision']
                    I += 1
                all_f1 += "%.1f " % (AVG_F1 / I * 100)
                # Save predictions
                # output_test_predictions_file = os.path.join(args.output_dir, "test_{}_predictions.txt".format(lang))
                # infile = os.path.join(args.data_dir, lang, "test")
                # idxfile = infile + '.idx'
                # save_predictions(args, predictions, output_test_predictions_file, infile, idxfile)
                print(all_f1, file=result_file)


def save_predictions(args, predictions, output_file, text_file, idx_file, output_word_prediction=False):
    # Save predictions
    with open(text_file, "r") as text_reader, open(idx_file, "r") as idx_reader:
        text = text_reader.readlines()
        index = idx_reader.readlines()
        assert len(text) == len(index)

    # Sanity check on the predictions
    with open(output_file, "w") as writer:
        example_id = 0
        prev_id = int(index[0])
        for line, idx in zip(text, index):
            if line == "" or line == "\n":
                example_id += 1
            else:
                cur_id = int(idx)
                output_line = '\n' if cur_id != prev_id else ''
                if output_word_prediction:
                    output_line += line.split()[0] + '\t'
                output_line += predictions[example_id].pop(0) + '\n'
                writer.write(output_line)
                prev_id = cur_id


if __name__ == "__main__":
    main()

    # --do_train --do_eval
