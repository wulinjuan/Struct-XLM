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
""" Finetuning multi-lingual models on XNLI/PAWSX (Bert, XLM, XLMRoberta)."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse
import glob
import logging
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import sys
sys.path.append('/data/home10b/wlj/code/Struct-XLM/')
from RobertaForSequenceClassification import RobertaForSequenceClassification
from src.Struct_XLM.BertEncoder import BertModel
from src.Struct_XLM.RobertaEncoder import RobertaModel
from src.Struct_XLM.actor import Actor

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMTokenizer,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from processors.utils import convert_examples_to_features
from processors.xnli import XnliProcessor
from processors.pawsx import PawsxProcessor

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer, BertModel),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer, RobertaModel),
    "xlmr": (XLMRobertaConfig, RobertaForSequenceClassification, XLMRobertaTokenizer, RobertaModel),
}

PROCESSORS = {
    'xnli': XnliProcessor,
    'pawsx': PawsxProcessor,
}

emb_config = {
    'mbert': 768,
    'xlmr': 1024,
    'infoxlm': 1024
}


def compute_metrics(preds, labels):
    scores = {
        "acc": (preds == labels).mean(),
        "num": len(
            preds),
        "correct": (preds == labels).sum()
    }
    return scores


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


def train(args, train_dataset, model, actorModel, tokenizer, lang2id=None):
    """Train the model."""
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    model.to('cuda:0')
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
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

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
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    best_score = 0
    if os.path.exists(args.model_name_or_path) and args.continue_train:
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = 98176*2
        epochs_trained = 0
        steps_trained_in_current_epoch = 98176*2

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

        if os.path.isfile(os.path.join(args.output_dir, "checkpoint-best/optimizer.pt")) and os.path.isfile(
                os.path.join(args.output_dir, "checkpoint-best/scheduler.pt")
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(args.output_dir, "checkpoint-best/optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(args.output_dir, "checkpoint-best/scheduler.pt")))
        best_score = 89.5

    best_checkpoint = None
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            text = batch[0]
            mask = batch[1]
            label = batch[3]
            head_mask = mask.masked_fill(batch[2] == 0, 0).long()
            # inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

            # if args.model_type == "xlm":
            #     inputs["langs"] = batch[4]

            context = model(input_ids=text, attention_mask=mask, get_embedding=True)['last_hidden_state']
            predicted = actorModel.get_target_output(context, mask)
            # loss_qry_all = 0.0
            sentence_actor_attention = None
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

            outputs = model(input_ids=text, attention_mask=mask, labels=label,
                            action=sentence_actor_attention)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                    # Only evaluate on single GPU otherwise metrics may not average well
                    if (args.local_rank == -1 and args.evaluate_during_training):
                        results = evaluate(args, model, actorModel, tokenizer, split=args.train_split,
                                           language=args.train_language,
                                           lang2id=lang2id)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    if args.eval_test_set:
                        output_predict_file = os.path.join(args.output_dir, 'eval_test_results')
                        total = total_correct = 0.0
                        with open(output_predict_file, 'a') as writer:
                            writer.write('\n======= Predict using the model from checkpoint-{}:\n'.format(global_step))
                            for language in args.predict_languages.split(','):
                                result = evaluate(args, model, actorModel, tokenizer, split=args.test_split,
                                                  language=language,
                                                  lang2id=lang2id, prefix='checkpoint-' + str(global_step))
                                writer.write('{}={}\n'.format(language, result['acc']))
                                total += result['num']
                                total_correct += result['correct']
                            writer.write('total={}\n'.format(total_correct / total))

                    if args.save_only_best_checkpoint:
                        result = evaluate(args, model, actorModel, tokenizer, split='dev', language=args.train_language,
                                          lang2id=lang2id, prefix=str(global_step))
                        logger.info(" Dev accuracy {} = {}".format(args.train_language, result['acc']))
                        if result['acc'] > best_score:
                            logger.info(" result['acc']={} > best_score={}".format(result['acc'], best_score))
                            output_dir = os.path.join(args.output_dir, "checkpoint-best")
                            best_checkpoint = output_dir
                            best_score = result['acc']
                            # Save model checkpoint
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = (
                                model.module if hasattr(model, "module") else model
                            )  # Take care of distributed/parallel training
                            model_to_save.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)

                            torch.save(args, os.path.join(output_dir, "training_args.bin"))
                            logger.info("Saving model checkpoint to %s", output_dir)

                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            logger.info("Saving optimizer and scheduler states to %s", output_dir)
                    else:
                        # Save model checkpoint
                        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, best_score, best_checkpoint


def evaluate(args, model, actorModel, tokenizer, split='train', language='en', lang2id=None, prefix="",
             output_file=None,
             label_list=None, output_only_prediction=True):
    """Evalute the model."""
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, split=split, language=language,
                                               lang2id=lang2id, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} {} *****".format(prefix, language))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        sentences = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                text = batch[0]
                mask = batch[1]
                label = batch[3]

                head_mask = mask.masked_fill(batch[2] == 0, 0).long()
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
                outputs = model(input_ids=text, attention_mask=mask, labels=label, action=sentence_actor_attention)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = label.detach().cpu().numpy()
                sentences = text.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, label.detach().cpu().numpy(), axis=0)
                sentences = np.append(sentences, text.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        else:
            raise ValueError("No other `output_mode` for XNLI.")
        result = compute_metrics(preds, out_label_ids)
        results.update(result)

        if output_file:
            logger.info("***** Save prediction ******")
            with open(output_file, 'w') as fout:
                pad_token_id = tokenizer.pad_token_id
                sentences = sentences.astype(int).tolist()
                sentences = [[w for w in s if w != pad_token_id] for s in sentences]
                sentences = [tokenizer.convert_ids_to_tokens(s) for s in sentences]
                # fout.write('Prediction\tLabel\tSentences\n')
                for p, l, s in zip(list(preds), list(out_label_ids), sentences):
                    s = ' '.join(s)
                    if label_list:
                        p = label_list[p]
                        l = label_list[l]
                    if output_only_prediction:
                        fout.write(str(p) + '\n')
                    else:
                        fout.write('{}\t{}\t{}\n'.format(p, l, s))
        logger.info("***** Eval results {} {} *****".format(prefix, language))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

    return results


def load_and_cache_examples(args, task, tokenizer, split='train', language='en', lang2id=None, evaluate=False):
    # Make sure only the first process in distributed training process the
    # dataset, and the others will use the cache
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    processor = PROCESSORS[task]()
    output_mode = "classification"
    # Load data features from cache or dataset file
    lc = '_lc' if args.do_lower_case else ''
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}_{}{}".format(
            split,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
            str(language),
            lc,
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if split == 'train':
            examples = processor.get_train_examples(args.data_dir, language)
        elif split == 'translate-train':
            examples = processor.get_translate_train_examples(args.data_dir, language)
        elif split == 'translate-test':
            examples = processor.get_translate_test_examples(args.data_dir, language)
        elif split == 'dev':
            examples = processor.get_dev_examples(args.data_dir, language)
        elif split == 'pseudo_test':
            examples = processor.get_pseudo_test_examples(args.data_dir, language)
        else:
            examples = processor.get_test_examples(args.data_dir, language)

        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=False,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0,
            lang2id=lang2id,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process the
    # dataset, and the others will use the cache
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()

        # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    else:
        raise ValueError("No other `output_mode` for {}.".format(args.task_name))

    if args.model_type == 'xlm':
        all_langs = torch.tensor([f.langs for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_langs)
    else:
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default='data/XNLI/',
        type=str,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default='xlmr',
        type=str,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default='/data/home10b/wlj/cached_models/xlm-roberta-large',
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list: " + "",
    )
    parser.add_argument(
        "--train_language", default="en", type=str, help="Train language if is different of the evaluation language."
    )
    parser.add_argument(
        "--predict_languages", type=str, default="en,ar,bg,de,el,es,fr,hi,ru,sw,th,tr,ur,vi,zh", help="prediction languages separated by ','."
    )
    parser.add_argument(
        "--output_dir",
        default='output/xnli/',
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--task_name",
        default="xnli",
        type=str,
        help="The task name",
    )

    parser.add_argument("--critic_output_dir", default="/data/home10b/wlj/code/Struct-XLM/src/Struct_XLM/output_no_warm_start", type=str)

    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--act_dropout", default=0.2, type=float)
    parser.add_argument("--margin", default=0.4, type=float)
    parser.add_argument("--s", default=1, type=float)
    parser.add_argument("--sentence_type", default='mean', type=str, help="mean, pool")
    parser.add_argument("--epsilon", type=float, default=0.05)

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run prediction.")
    parser.add_argument("--do_predict_dev", action="store_true", help="Whether to run prediction.")
    parser.add_argument("--init_checkpoint", type=str, default="",
                        help="initial checkpoint for predicting the dev set")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument("--train_split", type=str, default="train", help="split of training set")
    parser.add_argument("--test_split", type=str, default="test", help="split of training set")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=8e-6, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=12500, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=2000, help="Log every X updates steps.")
    parser.add_argument("--log_file", default="train", type=str, help="log file")
    parser.add_argument("--save_steps", type=int, default=2000, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--continue_train", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--eval_test_set",
        action="store_true",
        help="Whether to evaluate test set durinng training",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--save_only_best_checkpoint", action="store_true", help="save only the best checkpoint"
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    args.edim = emb_config[args.model_type]
    args.ml_model_path = args.model_name_or_path
    args.max_len = args.max_seq_length
    logging.basicConfig(
        handlers=[logging.FileHandler(os.path.join(args.output_dir, args.log_file)), logging.StreamHandler()],
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logging.info("Input args: %r" % args)

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
    else:  # Initializes the distributed backend which sychronizes nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", 2)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare dataset
    if args.task_name not in PROCESSORS:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = PROCESSORS[args.task_name]()
    args.output_mode = "classification"
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    # Make sure only the first process in distributed training loads model & vocab
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class, base_model = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    logger.info("config = {}".format(config))

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    lang2id = config.lang2id if args.model_type == "xlm" else None
    logger.info("lang2id = {}".format(lang2id))

    # Make sure only the first process in distributed training loads model & vocab
    if args.local_rank == 0:
        torch.distributed.barrier()
    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.init_checkpoint:
        logger.info("loading from folder {}".format(args.init_checkpoint))
        model = model_class.from_pretrained(
            args.init_checkpoint,
            args=args,
            config_class=config_class,
            model_class=base_model,
            config=config,
            cache_dir=args.init_checkpoint,
        )
    else:
        logger.info("loading from existing model {}".format(args.model_name_or_path))
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            args=args,
            config_class=config_class,
            model_class=base_model,
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
    model.to(args.device)
    output_dir = os.path.join(args.critic_output_dir, "checkpoint-best")

    actorModel = Actor(expirement=args, d_model=args.edim, dropout=args.act_dropout, no_cuda=args.no_cuda)
    actorModel.target_policy.load_state_dict(torch.load(os.path.join(output_dir, "target_model_actor.bin")))
    model.encoder.load_state_dict(torch.load(os.path.join(output_dir, "target_model_critic.bin")))
    actorModel.to(args.device)


    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, split=args.train_split,
                                                language=args.train_language, lang2id=lang2id, evaluate=False)
        global_step, tr_loss, best_score, best_checkpoint = train(args, train_dataset, model, actorModel, tokenizer,
                                                                  lang2id)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        logger.info(" best checkpoint = {}, best score = {}".format(best_checkpoint, best_score))
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir,
                                            args=args,
                                            config_class=config_class,
                                            model_class=base_model)
        # tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    """if os.path.exists(os.path.join(args.output_dir, 'checkpoint-best')):
        best_checkpoint = os.path.join(args.output_dir, 'checkpoint-best')
    else:
        """
    best_checkpoint = args.output_dir
    best_score = 0
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint,
                                                args=args,
                                                config_class=config_class,
                                                model_class=base_model)
            model.to(args.device)
            result = evaluate(args, model, actorModel, tokenizer, split='dev', language=args.train_language,
                              lang2id=lang2id,
                              prefix=prefix)
            if result['acc'] > best_score:
                best_checkpoint = checkpoint
                best_score = result['acc']
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

        output_eval_file = os.path.join(args.output_dir, 'eval_results')
        with open(output_eval_file, 'w') as writer:
            for key, value in results.items():
                writer.write('{} = {}\n'.format(key, value))
            writer.write("Best checkpoint is {}, best accuracy is {}".format(best_checkpoint, best_score))
            logger.info("Best checkpoint is {}, best accuracy is {}".format(best_checkpoint, best_score))

    # Prediction
    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(
            args.model_name_or_path if args.model_name_or_path else best_checkpoint, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(best_checkpoint,
                                            args=args,
                                            config_class=config_class,
                                            model_class=base_model)
        model.to(args.device)
        output_predict_file = os.path.join(args.output_dir, args.test_split + '_results.txt')
        total = total_correct = 0.0
        with open(output_predict_file, 'a') as writer:
            writer.write('======= Predict using the model from {} for {}:\n'.format(best_checkpoint, args.test_split))
            for language in args.predict_languages.split(','):
                output_file = os.path.join(args.output_dir, 'test-{}.tsv'.format(language))
                result = evaluate(args, model, actorModel, tokenizer, split=args.test_split, language=language,
                                  lang2id=lang2id,
                                  prefix='best_checkpoint', output_file=output_file, label_list=label_list)
                writer.write('{}={}\n'.format(language, result['acc']))
                logger.info('{}={}'.format(language, result['acc']))
                total += result['num']
                total_correct += result['correct']
            writer.write('total={}\n'.format(total_correct / total))

    if args.do_predict_dev:
        tokenizer = tokenizer_class.from_pretrained(
            args.model_name_or_path if args.model_name_or_path else best_checkpoint, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(best_checkpoint,
                                            args=args,
                                            config_class=config_class,
                                            model_class=base_model)
        model.to(args.device)
        output_predict_file = os.path.join(args.output_dir, 'dev_results')
        total = total_correct = 0.0
        with open(output_predict_file, 'w') as writer:
            writer.write('======= Predict using the model from {}:\n'.format(args.init_checkpoint))
            for language in args.predict_languages.split(','):
                output_file = os.path.join(args.output_dir, 'dev-{}.tsv'.format(language))
                result = evaluate(args, model, actorModel, tokenizer, split='dev', language=language, lang2id=lang2id,
                                  prefix='best_checkpoint', output_file=output_file, label_list=label_list)
                writer.write('{}={}\n'.format(language, result['acc']))
                total += result['num']
                total_correct += result['correct']
            writer.write('total={}\n'.format(total_correct / total))

    return result


if __name__ == "__main__":
    main()
