import argparse
import os
import pickle
import random
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import logging
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from tqdm import trange, tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from actor import Actor
from data_utils import read_parallel_txt, DataProcessor
from model import Critic

from transformers import BertConfig, XLMRobertaConfig

from BertEncoder import BertModel
from RobertaEncoder import RobertaModel

MODEL_CLASSES = {
    "bert": (BertConfig, BertModel),
    "xlmr": (XLMRobertaConfig, RobertaModel),
    "infoxlm": (XLMRobertaConfig, RobertaModel),
}

logger = logging.getLogger(__name__)

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


def config():
    parser = argparse.ArgumentParser()
    # data config
    parser.add_argument("--data_dir", default='/data/home10b/wlj/code/Struct-XLM/data/UD_data/', type=str)
    parser.add_argument("--output_dir", default="/data/home10b/wlj/code/Struct-XLM/src/Struct_XLM/output_new1/")
    parser.add_argument("--max_len", default=256, type=int, help="the max number of token in a sentence")
    parser.add_argument("--margin", default=0.5, type=float)
    parser.add_argument("--s", default=1, type=float)
    parser.add_argument("--sentence_type", default='pool_mean', type=str, help="pool_mean, mean, pool")
    parser.add_argument("--act_dropout", default=0.2, type=float)
    parser.add_argument("--samplecnt", default=5, type=int)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--grained", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.5)

    parser.add_argument("--batch_size", default=5, type=int)
    parser.add_argument("--save_step", default=50)
    parser.add_argument("--warmup_steps", default=200, type=int,
                        help="Linear warmup over warmup_steps.")
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
    parser.add_argument("--num_train_epochs", default=4.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")

    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev.txt set.")
    parser.add_argument("--warm_start", action="store_true",
                        help="Whether to run eval on the dev.txt set.")
    parser.add_argument("--LMpretrain", action="store_true",
                        help="Whether to run eval on the dev.txt set.")
    parser.add_argument("--RLpretrain", action="store_true",
                        help="Whether to run eval on the dev.txt set.")
    parser.add_argument("--warm_by_action", default=True)
    parser.add_argument("--do_lower_case", default=False,
                        help="Set this flag if you are using an uncased model.")

    # pretrained model config
    parser.add_argument('--ml_type', type=str, default="xlmr",
                        help='pretrained language model type:mbert, xlmr, infoxlm')
    parser.add_argument('--ml_model_path', type=str, default="/data/home10b/wlj/cached_models/xlm-roberta-large",
                        help='pretrained language model file: bert-base-multilingual-cased, xlm-roberta-large, infoxlm-large')

    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--log_file", default="./log.txt")

    args = parser.parse_args()
    return args


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
            action = np.argmax(predicted[pos]).item()
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
        Rinput = actor_attention.masked_fill(actor_attention == 0, 0)

    Rinput = Rinput.unsqueeze(0).unsqueeze(0).cuda()  # (1, seq_len, seq_len)
    return actions, Rinput, constituent_num/length_true if constituent_num!=0 else 1


def sampling_random(args, mask, p_action):
    lenth = torch.sum(mask).item()
    actions = np.copy(p_action.cpu()).tolist()
    # actions += [0] * (args.max_len - lenth)
    actor_attention = torch.zeros(args.max_len, args.max_len).cuda()
    for i in range(lenth):
        actor_attention[0][i] = 1
    pos = 1
    end_one = False
    while pos < lenth:
        #actor_attention[pos][pos] = 1
        if actions[pos] == 0:
            begin_sub=pos
            for i in range(pos + 1, lenth):
                if actions[i] == -1 and i!= pos+1:
                    continue
                elif actions[i] == -1 and i== begin_sub+1:
                    begin_sub +=1
                actor_attention[pos][i] = 1
                actor_attention[i][pos] = 1
                if actions[i] == 1:
                    end_one = True
                    break
        elif actions[pos] == 1:
            if end_one:
                end_one = False
                if pos+1<lenth and actions[pos+1] == -1:
                    for i in range(pos + 1, lenth):
                        actor_attention[pos][i] = 1
                        actor_attention[i][pos] = 1
                        if i+1<lenth and actions[i+1] != -1:
                            break
                continue
            else:
                actor_attention[pos][pos] = 1
        else:
            if pos + 1 < lenth and actions[pos + 1] != -1:
                pos += 1
                continue
            for i in range(pos + 1, lenth):
                actor_attention[pos][i] = 1
                actor_attention[i][pos] = 1
                if i+1 < lenth and actions[i+1] != -1:
                    break
        pos += 1
    # print(actor_attention)
    Rinput = actor_attention.masked_fill(actor_attention == 0, 0)

    Rinput = Rinput.unsqueeze(0).unsqueeze(0).cuda()
    return Rinput


def train(args, data, train_dataloader, dev_dataloader, test_dataloader, criticModel, actorModel, LM_trainable=True,
          RL_trainable=True):
    """Train the model."""
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    def optimizer_schedule(model):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=t_total)
        return optimizer, scheduler

    if RL_trainable:
        args.learning_rate = args.learning_rate*args.tau
        actor_target_optimizer, actor_target_schedule = optimizer_schedule(actorModel.target_policy)
        # actor_target_optimizer = torch.optim.Adam(actorModel.target_policy.parameters(), lr=args.learning_rate)
    critic_target_optimizer, critic_target_schedule = optimizer_schedule(criticModel.target_pred)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        criticModel.target_pred = torch.nn.DataParallel(criticModel.target_pred)
        actorModel.target_policy = torch.nn.DataParallel(actorModel.target_policy)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(data.train_data))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    best_loss = 999
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            global_step += 1
            criticModel.zero_grad()  # 复制target模型的数data参数
            actorModel.zero_grad()
            totloss = 0.
            batch = tuple(t.to(args.device) for t in batch if t is not None)

            text1 = batch[0]
            action1 = batch[4]
            mask1 = batch[2]
            head_mask1 = mask1.masked_fill(action1 == -1, 0)
            text2 = batch[1]
            mask2 = batch[3]
            action2 = batch[5]
            head_mask2 = mask2.masked_fill(action2 == -1, 0)
            sentence_id = batch[6]

            true_batch = min(args.batch_size * args.n_gpu, text1.size()[0])

            sentence_actor_attention1 = None
            sentence_actor_attention2 = None
            losslist1 = []
            losslist2 = []
            aveloss_list1 = [0.0 for _ in range(true_batch)]
            aveloss_list2 = [0.0 for _ in range(true_batch)]
            actionlist1 = []
            actionlist2 = []
            total_loss = 0.
            if RL_trainable:
                criticModel.train(False)
                actorModel.train()
                context1, context2 = \
                    criticModel(text1, text2, mask1, mask2, None, None, sentence_id,
                                print_sentence=True)
                context1 = context1['last_hidden_state'].masked_fill(head_mask1.unsqueeze(-1) == 0, -1e9)
                context2 = context2['last_hidden_state'].masked_fill(head_mask2.unsqueeze(-1) == 0, -1e9)

                predicted1 = actorModel.get_target_output(context1, mask1)
                predicted2 = actorModel.get_target_output(context2, mask2)
                for j in range(args.samplecnt):
                    action_list1 = []
                    action_list2 = []
                    loss_list1 = []
                    loss_list2 = []
                    for i in range(true_batch):
                        p1 = predicted1[i]
                        p2 = predicted2[i]
                        m1 = mask1[i]
                        m2 = mask2[i]
                        h1 = head_mask1[i]
                        h2 = head_mask2[i]
                        actions1, Rinput1, constituent_num1 = Sampling_RL(args, p1, m1, h1, args.epsilon, Random=True)
                        actions2, Rinput2, constituent_num2 = Sampling_RL(args, p2, m2, h2, args.epsilon, Random=True)
                        action_list1.append(actions1)
                        action_list2.append(actions2)
                        #loss_list1.append(-(1 * constituent_num1 + 0.1 / constituent_num1-0.6) * 0.1 * args.grained)
                        #loss_list2.append(-(1 * constituent_num2 + 0.1 / constituent_num2-0.6) * 0.1 * args.grained)

                        if i == 0:
                            sentence_actor_attention1 = Rinput1
                            sentence_actor_attention2 = Rinput2
                        else:
                            sentence_actor_attention1 = torch.cat([sentence_actor_attention1, Rinput1])
                            sentence_actor_attention2 = torch.cat([sentence_actor_attention2, Rinput2])
                    loss = criticModel(text1, text2, mask1, mask2, sentence_actor_attention1,
                                       sentence_actor_attention2, sentence_id)[0]
                    for k in range(true_batch):
                        # print(loss[1])
                        if args.n_gpu == 1:
                            if len(loss_list1) != true_batch:
                                loss_list1.append(loss[1][int(k // args.n_gpu)].item())
                                loss_list2.append(loss[2][int(k // args.n_gpu)].item())
                            else:
                                loss_list1[k] += loss[1][int(k // args.n_gpu)].item()
                                loss_list2[k] += loss[2][int(k // args.n_gpu)].item()
                        else:
                            loss_list1[k] += loss[1][int(k // args.n_gpu)][int(k % args.n_gpu)].item()
                            loss_list2[k] += loss[2][int(k // args.n_gpu)][int(k % args.n_gpu)].item()
                        aveloss_list1[k] += loss_list1[k]
                        aveloss_list2[k] += loss_list2[k]
                    loss[0].sum().backward()
                    losslist1.append(loss_list1)
                    losslist2.append(loss_list2)
                    actionlist1.append(action_list1)
                    actionlist2.append(action_list2)

                if LM_trainable:
                    criticModel.train()
                    actorModel.train()
                    critic_target_optimizer.zero_grad()
                    loss = criticModel(text1, text2, mask1, mask2, sentence_actor_attention1,
                                       sentence_actor_attention2, sentence_id)[0][0]
                    if not RL_trainable:
                        total_loss += loss
                    loss.sum().backward()
                    critic_target_optimizer.step()
                    critic_target_schedule.step()

                grad_all = []
                flag = 0
                for j in range(args.samplecnt):
                    rr_all1 = []
                    rr_all2 = []
                    for m in range(true_batch):
                        avg1 = aveloss_list1[m] / args.samplecnt
                        total_loss -= avg1
                        avg2 = aveloss_list2[m] / args.samplecnt
                        total_loss -= avg2
                        rr_temp=[]
                        for pos in range(len(actionlist1[j][m])):
                            rr = [0, 0]
                            rr[actionlist1[j][m][pos]] = ((losslist1[j][m] - avg1) * args.alpha)
                            rr_temp.append(rr)
                        rr_all1.append(rr_temp)
                        rr_temp=[]
                        for pos in range(len(actionlist2[j][m])):
                            rr = [0, 0]
                            rr[actionlist2[j][m][pos]] = ((losslist2[j][m] - avg2) * args.alpha)
                            rr_temp.append(rr)
                        rr_all2.append(rr_temp)
                    g, out1 = actorModel.get_batch_gradient(context1, mask=mask1, reward_batch=rr_all1)
                    if flag == 0:
                        for l in range(len(g)):
                            grad_all.append(g[l])
                        flag = 1
                    else:
                        for l in range(len(g)):
                            grad_all[l] += g[l]
                    g, out2 = actorModel.get_batch_gradient(context2, mask=mask2, reward_batch=rr_all2)
                    for l in range(len(g)):
                        grad_all[l] += g[l]
                    actor_target_optimizer.zero_grad()
                    actorModel.assign_network_gradients(grad_all)
                    # print("previous grad: ", actorModel.active_policy.b.grad)
                    actor_target_optimizer.step()
                    actor_target_schedule.step()
            else:
                for i in range(true_batch):
                    act1 = action1[i]
                    act2 = action2[i]
                    m1 = mask1[i]
                    m2 = mask2[i]
                    if args.warm_by_action:
                        if i == 0:
                            sentence_actor_attention1 = sampling_random(args, m1, act1)
                            sentence_actor_attention2 = sampling_random(args, m2, act2)
                        else:
                            sentence_actor_attention1 = torch.cat(
                                [sentence_actor_attention1, sampling_random(args, m1, act1)])
                            sentence_actor_attention2 = torch.cat(
                                [sentence_actor_attention2, sampling_random(args, m2, act2)])

                criticModel.train()
                critic_target_optimizer.zero_grad()
                loss = criticModel(text1, text2, mask1, mask2, sentence_actor_attention1,
                                   sentence_actor_attention2, sentence_id)[0][0]
                if not RL_trainable:
                    total_loss += loss
                loss.sum().backward()
                critic_target_optimizer.step()  # 更新active
                critic_target_schedule.step()

            if RL_trainable:
                criticModel.train(False)
                actorModel.train()
                if LM_trainable:
                    criticModel.train()
                    actorModel.train()
            else:
                # print("Again RL False and LSTM True")
                criticModel.train()
                actorModel.train(False)
            if global_step % args.save_step == 0:
                loss_dev_no_actor = eval_model(args, criticModel, dev_dataloader)
                if args.warm_by_action and RL_trainable:
                    loss_dev = eval_model_RL(args, criticModel, actorModel, dev_dataloader, RL_trained=RL_trainable)
                    print("batch ", global_step, "total loss ", total_loss, "| dev: ", loss_dev, "| dev_no_actor: ", loss_dev_no_actor)
                    if loss_dev < best_loss:
                        best_loss = loss_dev
                        # Save the best model checkpoint
                        output_dir = os.path.join(args.output_dir,
                                                  u"checkpoint-best{}{}".format("_no_policy" if not RL_trainable else "", "_actor_only" if RL_trainable and not LM_trainable else ""))

                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # Take care of distributed/parallel training
                        model_to_save = criticModel.target_pred.module if hasattr(criticModel.target_pred,
                                                                                  "module") else criticModel.target_pred
                        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "target_model_critic.bin"))

                        model_to_save = actorModel.target_policy.module if hasattr(actorModel.target_policy,
                                                                                   "module") else actorModel.target_policy
                        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "target_model_actor.bin"))
                        logger.info('Saved model to %s' % output_dir)
                else:
                    print("batch ", global_step, "total loss ", total_loss, "| dev_no_actor: ",
                          loss_dev_no_actor)
                    if loss_dev_no_actor < best_loss:
                        best_loss = loss_dev_no_actor
                        # Save the best model checkpoint
                        output_dir = os.path.join(args.output_dir,
                                                  u"checkpoint-best{}{}".format("_no_policy" if not RL_trainable else "", "_actor_only" if RL_trainable and not LM_trainable else ""))

                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # Take care of distributed/parallel training
                        model_to_save = criticModel.target_pred.module if hasattr(criticModel.target_pred,
                                                                                  "module") else criticModel.target_pred
                        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "target_model_critic.bin"))

                        model_to_save = actorModel.target_policy.module if hasattr(actorModel.target_policy,
                                                                                   "module") else actorModel.target_policy
                        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "target_model_actor.bin"))
                        logger.info('Saved model to %s' % output_dir)
                    tr_loss += total_loss

    loss_test_no_actor = eval_model(args, criticModel, test_dataloader)
    loss_dev_no_actor = eval_model(args, criticModel, dev_dataloader)
    if args.warm_by_action and RL_trainable:
        loss_test = eval_model_RL(args, criticModel, actorModel, test_dataloader, RL_trained=RL_trainable)
        loss_dev = eval_model_RL(args, criticModel, actorModel, dev_dataloader, RL_trained=RL_trainable)
        print("batch ", global_step, "total loss ", total_loss, "----test: ", loss_test, "| dev: ", loss_dev,
              "----test_no_actor: ", loss_test_no_actor, "| dev_no_actor: ", loss_dev_no_actor)

        if loss_dev < best_loss:
            # Save the best model checkpoint
            output_dir = os.path.join(args.output_dir, u"checkpoint-best{}{}".format("_no_policy" if not RL_trainable else "", "_actor_only" if RL_trainable and not LM_trainable else ""))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # Take care of distributed/parallel training
            model_to_save = criticModel.target_pred.module if hasattr(criticModel.target_pred,
                                                                      "module") else criticModel.target_pred
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, "target_model_critic.bin"))

            model_to_save = actorModel.target_policy.module if hasattr(actorModel.target_policy,
                                                                       "module") else actorModel.target_policy
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, "target_model_actor.bin"))
            # torch.save(actorModel.state_dict(), os.path.join(output_dir, "training_model_actor.bin"))
            # torch.save(args, os.path.join(output_dir, "training_model_actor.bin"))
            logger.info('Saved model to %s' % output_dir)
    elif loss_dev_no_actor < best_loss:
        print("batch ", global_step, "total loss ", total_loss,
              "----test_no_actor: ", loss_test_no_actor, "| dev_no_actor: ", loss_dev_no_actor)
        # Save the best model checkpoint
        output_dir = os.path.join(args.output_dir, u"checkpoint-best{}{}".format("_no_policy" if not RL_trainable else "", "_actor_only" if RL_trainable and not LM_trainable else ""))

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Take care of distributed/parallel training
        model_to_save = criticModel.target_pred.module if hasattr(criticModel.target_pred,
                                                                  "module") else criticModel.target_pred
        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "target_model_critic.bin"))

        model_to_save = actorModel.target_policy.module if hasattr(actorModel.target_policy,
                                                                   "module") else actorModel.target_policy
        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "target_model_actor.bin"))
        # torch.save(actorModel.state_dict(), os.path.join(output_dir, "training_model_actor.bin"))
        # torch.save(args, os.path.join(output_dir, "training_model_actor.bin"))
        logger.info('Saved model to %s' % output_dir)
    elif RL_trainable:
        # Save the best model checkpoint
        output_dir = os.path.join(args.output_dir, u"checkpoint-final")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Take care of distributed/parallel training
        model_to_save = criticModel.target_pred.module if hasattr(criticModel.target_pred,
                                                                  "module") else criticModel.target_pred
        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "target_model_critic.bin"))

        model_to_save = actorModel.target_policy.module if hasattr(actorModel.target_policy,
                                                                   "module") else actorModel.target_policy
        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "target_model_actor.bin"))
        # torch.save(actorModel.state_dict(), os.path.join(output_dir, "training_model_actor.bin"))
        # torch.save(args, os.path.join(output_dir, "training_model_actor.bin"))
        logger.info('Saved model to %s' % output_dir)

    return global_step, tr_loss / global_step


def eval_model(args, model, val_iter):
    total_epoch_loss = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            batch = tuple(t.to(args.device) for t in batch if t is not None)
            text1 = batch[0]
            mask1 = batch[2]
            text2 = batch[1]
            mask2 = batch[3]
            sentence_id = batch[6]
            loss = model(text1, text2, mask1, mask2, None, None, sentence_id)[0][0]
            total_epoch_loss += loss.mean().item()

    return total_epoch_loss / len(val_iter)


def eval_model_RL(args, criticModel, actorModel, val_iter, RL_trained=False):
    total_epoch_loss = 0
    criticModel.eval()
    actorModel.eval()
    sentence_actor_attention1 = None
    sentence_actor_attention2 = None

    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            batch = tuple(t.to(args.device) for t in batch if t is not None)
            text1 = batch[0]
            action1 = batch[4]
            mask1 = batch[2]
            head_mask1 = mask1.masked_fill(action1 == -1, 0)
            text2 = batch[1]
            mask2 = batch[3]
            action2 = batch[5]
            head_mask2 = mask2.masked_fill(action2 == -1, 0)
            sentence_id = batch[6]
            true_batch = min(args.batch_size * args.n_gpu, text1.size()[0])

            context1, context2 = \
                criticModel(text1, text2, mask1, mask2, None, None, sentence_id,
                            print_sentence=True)
            context1 = context1['last_hidden_state'].masked_fill(head_mask1.unsqueeze(-1) == 0, -1e9)
            context2 = context2['last_hidden_state'].masked_fill(head_mask2.unsqueeze(-1) == 0, -1e9)

            predicted1 = actorModel.get_target_output(context1, mask1)
            predicted2 = actorModel.get_target_output(context2, mask2)

            if RL_trained:
                for i in range(true_batch):
                    p1 = predicted1[i].cpu()
                    p2 = predicted2[i].cpu()
                    m1 = mask1[i]
                    m2 = mask2[i]
                    h1 = head_mask1[i]
                    h2 = head_mask2[i]
                    actions1, Rinput1, constituent_num1 = Sampling_RL(args, p1, m1, h1, args.epsilon, Random=False)

                    actions2, Rinput2, constituent_num2 = Sampling_RL(args, p2, m2, h2, args.epsilon, Random=False)
                    if i == 0:
                        sentence_actor_attention1 = Rinput1
                        sentence_actor_attention2 = Rinput2
                    else:
                        sentence_actor_attention1 = torch.cat([sentence_actor_attention1, Rinput1])
                        sentence_actor_attention2 = torch.cat([sentence_actor_attention2, Rinput2])
            else:
                for i in range(true_batch):
                    act1 = action1[i]
                    act2 = action2[i]
                    m1 = mask1[i]
                    m2 = mask2[i]
                    if i == 0:
                        sentence_actor_attention1 = sampling_random(args, m1, act1)
                        sentence_actor_attention2 = sampling_random(args, m2, act2)
                    else:
                        sentence_actor_attention1 = torch.cat([sentence_actor_attention1, sampling_random(args, m1, act1)])
                        sentence_actor_attention2 = torch.cat([sentence_actor_attention2, sampling_random(args, m2, act2)])
            loss = criticModel(text1, text2, mask1, mask2, sentence_actor_attention1, sentence_actor_attention2,
                               sentence_id)[0][0]
            total_epoch_loss += loss.mean().item()

    return total_epoch_loss / len(val_iter)


def eval_model_syn(args, criticModel, actorModel, val_iter):
    total_epoch_loss = 0
    criticModel.eval()
    actorModel.eval()
    sam_num_en = 0
    pred_num_en = 0
    true_num_en = 0
    sam_num_tgt = 0
    pred_num_tgt = 0
    true_num_tgt = 0

    const_num_en = []
    const_num_tgt = []

    const_num_en_g = []
    const_num_tgt_g = []

    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            batch = tuple(t.to(args.device) for t in batch if t is not None)
            text1 = batch[0]
            action1 = batch[4]
            mask1 = batch[2]
            head_mask1 = mask1.masked_fill(action1 == -1, 0)
            text2 = batch[1]
            mask2 = batch[3]
            action2 = batch[5]
            head_mask2 = mask2.masked_fill(action2 == -1, 0)
            sentence_id = batch[6]
            true_batch = min(args.batch_size * args.n_gpu, text1.size()[0])

            context1, context2 = \
                criticModel(text1, text2, mask1, mask2, None, None, sentence_id,
                            print_sentence=True)
            context1 = context1['last_hidden_state'].masked_fill(head_mask1.unsqueeze(-1) == 0, -1e9)
            context2 = context2['last_hidden_state'].masked_fill(head_mask2.unsqueeze(-1) == 0, -1e9)

            predicted1 = actorModel.get_target_output(context1, mask1)
            predicted2 = actorModel.get_target_output(context2, mask2)

            for i in range(true_batch):
                p1 = predicted1[i].cpu()
                p2 = predicted2[i].cpu()
                m1 = mask1[i]
                m2 = mask2[i]
                h1 = head_mask1[i]
                h2 = head_mask2[i]
                actions1, Rinput1, constituent_num1 = Sampling_RL(args, p1, m1, h1, args.epsilon, Random=False)

                actions2, Rinput2, constituent_num2 = Sampling_RL(args, p2, m2, h2, args.epsilon, Random=False)

                act1 = action1[i]
                act2 = action2[i]
                count_en = 0
                count_tgt = 0
                count_en_g = 0
                count_tgt_g =0
                for h_1, a1, a2 in zip(h1, actions1, act1):
                    if h_1 == 0:
                        continue
                    if a1 == 1:
                        count_en+=1
                        pred_num_en += 1
                    if a2 == 1:
                        count_en_g += 1
                        true_num_en += 1
                    if a1 == a2 and a2 == 1:
                        sam_num_en += 1
                for h_1, a1, a2 in zip(h2, actions2, act2):
                    if h_1 == 0:
                        continue
                    if a1 == 1:
                        count_tgt+=1
                        pred_num_tgt += 1
                    if a2 == 1:
                        count_tgt_g += 1
                        true_num_tgt += 1
                    if a1 == a2 and a2 == 1:
                        sam_num_tgt += 1
                const_num_en.append(count_en)
                const_num_en_g.append(count_en_g)
                const_num_tgt.append(count_tgt)
                const_num_tgt_g.append(count_tgt_g)

    P_en = sam_num_en/true_num_en
    R_en = sam_num_en/pred_num_en
    P_tgt = sam_num_tgt/true_num_tgt
    R_tgt = sam_num_tgt/pred_num_tgt
    F_en = (2*P_en*R_en)/(P_en+R_en)
    F_tgt = (2*P_tgt*R_tgt)/(P_tgt+R_tgt)

    print(sum(const_num_en)/len(const_num_en))
    print(sum(const_num_en_g)/len(const_num_en_g))
    print(sum(const_num_tgt)/len(const_num_tgt))
    print(sum(const_num_tgt_g)/len(const_num_tgt_g))
    print(P_en, R_en, P_tgt, R_tgt)

    return F_en*100, F_tgt*100


def main():
    args = config()
    args.edim = emb_config[args.ml_type]

    print("*" * 25 + " DATA PREPARATION " + "*" * 25)
    data_cache_file = os.path.join(args.data_dir, u"train_{}.pkl".format(args.ml_type))
    if not os.path.exists(data_cache_file):
        train_dp = read_parallel_txt(args.data_dir + "train.csv")
        dev_dp = read_parallel_txt(args.data_dir + "dev.csv")
        test_dp = read_parallel_txt(args.data_dir + "test.csv")
        dp = DataProcessor(dp_train=train_dp,
                           dp_dev=dev_dp,
                           dp_test=test_dp,
                           experiment=args)
        data = dp.process()
        output_hal = open(data_cache_file, 'wb')
        str = pickle.dumps(data, protocol=4)
        output_hal.write(str)
        output_hal.close()
    else:
        with open(data_cache_file, 'rb') as file:
            data = pickle.loads(file.read())

    print("*" * 25 + " DATA PREPARATION " + "*" * 25)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        # Initializes the distributed backend which sychronizes nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 0
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

    print("*" * 25 + " MODEL INITIALIZATION " + "*" * 25)

    config_class, model_class = MODEL_CLASSES[args.ml_type]
    criticModel = Critic(expirement=args, config_class=config_class, model_class=model_class)
    actorModel = Actor(expirement=args, d_model=args.edim, dropout=args.act_dropout, no_cuda=args.no_cuda)
    # Make sure only the first process in distributed training loads model/vocab
    if args.local_rank == 0:
        torch.distributed.barrier()
    logger.info("Training/evaluation parameters %s", args)
    train_dataset = data.train_data
    dev_dataset = data.dev_data
    test_dataset = data.test_data

    args.train_batch_size = args.batch_size * max(1, args.n_gpu)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=RandomSampler(train_dataset),
                                  batch_size=args.train_batch_size,
                                  drop_last=True)

    args.dev_batch_size = args.batch_size * max(1, args.n_gpu)
    dev_dataloader = DataLoader(dev_dataset,
                                sampler=RandomSampler(dev_dataset),
                                batch_size=args.dev_batch_size,
                                drop_last=True)

    args.test_batch_size = args.batch_size * max(1, args.n_gpu)
    test_dataloader = DataLoader(test_dataset,
                                 sampler=RandomSampler(test_dataset),
                                 batch_size=args.test_batch_size,
                                 drop_last=True,
                                 shuffle=False)

    criticModel.to(args.device)
    actorModel.to(args.device)
    if args.do_train:
        if not args.warm_start:
            args.num_train_epochs = 5
        else:
            if not args.LMpretrain:
                args.num_train_epochs = 2
                global_step, tr_loss = train(args, data, train_dataloader, dev_dataloader, test_dataloader, criticModel,
                                             actorModel, LM_trainable=True, RL_trainable=False)
                logger.info("LM pretrained global_step = %s, average loss = %s", global_step, tr_loss)
            criticModel.target_pred.to(args.device)
            output_dir = os.path.join(args.output_dir, "checkpoint-best_no_policy")
            print("Load pretrained LM from ", output_dir)
            criticModel.target_pred.load_state_dict(
                    torch.load(os.path.join(output_dir, "target_model_critic.bin"), map_location=torch.device('cpu')))
            if not args.RLpretrain:
                args.num_train_epochs = 2
                global_step, tr_loss = train(args, data, train_dataloader, dev_dataloader, test_dataloader, criticModel,
                                             actorModel, LM_trainable=False, RL_trainable=True)
                logger.info("RL pretrained global_step = %s, average loss = %s", global_step, tr_loss)
    
            output_dir = os.path.join(args.output_dir, "checkpoint-best_actor_only")
            print("Load action from ", output_dir)
            actorModel.target_policy.load_state_dict(torch.load(os.path.join(output_dir, "target_model_actor.bin")))
            actorModel.target_policy.to(args.device)
            args.num_train_epochs = 3
        global_step, tr_loss = train(args, data, train_dataloader, dev_dataloader, test_dataloader, criticModel,
                                     actorModel, LM_trainable=True, RL_trainable=True)
        logger.info("total global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_eval:
        if not args.do_train:
            output_dir = os.path.join(args.output_dir, u"checkpoint-best")
            criticModel.target_pred.load_state_dict(torch.load(os.path.join(output_dir, "target_model_critic.bin")))
            actorModel.target_policy.load_state_dict(torch.load(os.path.join(output_dir, "target_model_actor.bin")))
            loss_test_no_actor = eval_model(args, criticModel, test_dataloader)
            loss_test = eval_model_RL(args, criticModel, actorModel, test_dataloader, RL_trained=True)
            print("----test: ", loss_test,
                  "----test_no_actor: ", loss_test_no_actor)

def main_lm():
    args = config()
    args.edim = emb_config[args.ml_type]

    print("*" * 25 + " DATA PREPARATION " + "*" * 25)
    data_cache_file = os.path.join(args.data_dir, u"train_{}.pkl".format(args.ml_type))
    if not os.path.exists(data_cache_file):
        train_dp = read_parallel_txt(args.data_dir + "train.csv")
        dev_dp = read_parallel_txt(args.data_dir + "dev.csv")
        test_dp = read_parallel_txt(args.data_dir + "test.csv")
        dp = DataProcessor(dp_train=train_dp,
                           dp_dev=dev_dp,
                           dp_test=test_dp,
                           experiment=args)
        data = dp.process()
        output_hal = open(data_cache_file, 'wb')
        str = pickle.dumps(data, protocol=4)
        output_hal.write(str)
        output_hal.close()
    else:
        with open(data_cache_file, 'rb') as file:
            data = pickle.loads(file.read())

    print("*" * 25 + " DATA PREPARATION " + "*" * 25)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        # Initializes the distributed backend which sychronizes nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 0
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

    print("*" * 25 + " MODEL INITIALIZATION " + "*" * 25)

    config_class, model_class = MODEL_CLASSES[args.ml_type]
    criticModel = Critic(expirement=args, config_class=config_class, model_class=model_class)
    actorModel = Actor(expirement=args, d_model=args.edim, dropout=args.act_dropout, no_cuda=args.no_cuda)
    # Make sure only the first process in distributed training loads model/vocab
    if args.local_rank == 0:
        torch.distributed.barrier()
    logger.info("Training/evaluation parameters %s", args)
    train_dataset = data.train_data
    dev_dataset = data.dev_data
    test_dataset = data.test_data

    args.train_batch_size = args.batch_size * max(1, args.n_gpu)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=RandomSampler(train_dataset),
                                  batch_size=args.train_batch_size,
                                  drop_last=True)

    args.dev_batch_size = args.batch_size * max(1, args.n_gpu)
    dev_dataloader = DataLoader(dev_dataset,
                                sampler=RandomSampler(dev_dataset),
                                batch_size=args.dev_batch_size,
                                drop_last=True)

    args.test_batch_size = args.batch_size * max(1, args.n_gpu)
    test_dataloader = DataLoader(test_dataset,
                                 sampler=RandomSampler(test_dataset),
                                 batch_size=args.test_batch_size,
                                 drop_last=True)

    criticModel.to(args.device)
    actorModel.to(args.device)
    if args.do_train:
        if not args.LMpretrain:
            args.num_train_epochs = 5
            global_step, tr_loss = train(args, data, train_dataloader, dev_dataloader, test_dataloader, criticModel,
                                         actorModel, LM_trainable=True, RL_trainable=False)
            logger.info("LM pretrained global_step = %s, average loss = %s", global_step, tr_loss)
        criticModel.target_pred.to(args.device)
        output_dir = os.path.join(args.output_dir, "checkpoint-best_no_policy")
        print("Load pretrained LM from ", output_dir)
        criticModel.target_pred.load_state_dict(
                torch.load(os.path.join(output_dir, "target_model_critic.bin"), map_location=torch.device('cpu')))

    if args.do_eval:
        output_dir = os.path.join(args.output_dir, "checkpoint-best_no_policy")
        criticModel.target_pred.load_state_dict(torch.load(os.path.join(output_dir, "target_model_critic.bin")))
        actorModel.target_policy.load_state_dict(torch.load(os.path.join(output_dir, "target_model_actor.bin")))
        print("Load pretrained LM from ", output_dir)
        loss_test_no_actor = eval_model(args, criticModel, test_dataloader)
        print("----test_no_actor: ", loss_test_no_actor)


def main_syn_test():
    args = config()
    args.edim = emb_config[args.ml_type]

    print("*" * 25 + " DATA PREPARATION " + "*" * 25)
    data_cache_file = os.path.join(args.data_dir, u"test_{}.pkl".format(args.ml_type))
    if not os.path.exists(data_cache_file):
        train_dp = read_parallel_txt(args.data_dir + "train.csv")
        dev_dp = read_parallel_txt(args.data_dir + "dev.csv")
        test_dp = read_parallel_txt(args.data_dir + "test.csv")
        dp = DataProcessor(dp_train=train_dp,
                           dp_dev=dev_dp,
                           dp_test=test_dp,
                           experiment=args)
        data = dp.process()
        output_hal = open(data_cache_file, 'wb')
        str = pickle.dumps(data, protocol=4)
        output_hal.write(str)
        output_hal.close()
    else:
        with open(data_cache_file, 'rb') as file:
            data = pickle.loads(file.read())

    print("*" * 25 + " DATA PREPARATION " + "*" * 25)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        # Initializes the distributed backend which sychronizes nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 0
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

    print("*" * 25 + " MODEL INITIALIZATION " + "*" * 25)

    config_class, model_class = MODEL_CLASSES[args.ml_type]
    criticModel = Critic(expirement=args, config_class=config_class, model_class=model_class)
    actorModel = Actor(expirement=args, d_model=args.edim, dropout=args.act_dropout, no_cuda=args.no_cuda)
    # Make sure only the first process in distributed training loads model/vocab
    if args.local_rank == 0:
        torch.distributed.barrier()
    logger.info("Training/evaluation parameters %s", args)
    train_dataset = data.train_data
    dev_dataset = data.dev_data
    test_dataset = data.test_data

    args.test_batch_size = args.batch_size * max(1, args.n_gpu)
    test_dataloader = DataLoader(test_dataset,
                                 sampler=RandomSampler(test_dataset),
                                 batch_size=args.test_batch_size,
                                 drop_last=True)

    criticModel.to(args.device)
    actorModel.to(args.device)

    output_dir = os.path.join(args.output_dir, "checkpoint-best")
    criticModel.target_pred.load_state_dict(torch.load(os.path.join(output_dir, "target_model_critic.bin")))
    actorModel.target_policy.load_state_dict(torch.load(os.path.join(output_dir, "target_model_actor.bin")))

    print("Load pretrained LM from ", output_dir)
    F1_en, F1_tgt = eval_model_syn(args, criticModel, actorModel, test_dataloader)
    print("----F1_en: ", F1_en, "----F1_tgt: ",F1_tgt)

if __name__ == '__main__':
    main()
    # main_lm()
    #main_syn_test()
