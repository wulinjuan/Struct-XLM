from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, layer_norm_eps=1e-05, hidden_dropout_prob=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    

class PolicyNetwork(nn.Module):
    def __init__(self, d_model, dropout=0.8):
        super(PolicyNetwork, self).__init__()
        self.d_model = d_model
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        # self.linear_output = nn.Linear(d_model, d_model)
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = SelfOutput(d_model)

        self.actor_linear = nn.Linear(d_model, 2)  # 0 represent "in the constituent", 1 represent the tail of
        # constituent, 2 represent apart of puncuation

    def forward(self, context, mask):
        # (batch, seq_len, d_model)
        context = self.norm(context)

        key = self.linear_key(context)  # (batch, seq_len, d_model)
        query = self.linear_query(context)  # (batch, seq_len, d_model)
        value = self.linear_value(context)  # (batch, seq_len, d_model)
        mask_new = []
        for i in range(mask.size()[0]):
            length = torch.sum(mask[i]).item()
            mask_ = mask[i].unsqueeze(0).repeat(length,1).cpu()
            zero_tensor = torch.zeros((mask_.size()[1]-length, mask_.size()[1]))
            mask_ = torch.cat((mask_, zero_tensor), dim=0)
            mask_ = torch.tril(mask_, diagonal=0)
            mask_new.append(mask_)
        mask = torch.stack(mask_new, dim=0).cuda()

        scores = torch.matmul(query, key.transpose(-2, -1)) / self.d_model
        # print(scores.size(), ",", mask.size())
        scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        # attention_probs = self.dropout(scores)
        context_layer = torch.matmul(scores, value)
        context_emb = self.out(context_layer, context)
        actor = self.actor_linear(context_emb)
        actor = F.softmax(actor, dim=-1)
        # (batch, seq_len, d_model)

        return actor, context_emb


class PolicyNetwork1(nn.Module):
    def __init__(self, d_model, dropout=0.8, no_cuda=False):
        super(PolicyNetwork1, self).__init__()
        self.d_model = d_model
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_query = nn.Linear(d_model, d_model)
        # self.linear_output = nn.Linear(d_model, d_model)
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.no_cuda = no_cuda

    def forward(self, context, eos_mask):
        batch_size, seq_len = context.size()[:2]

        context = self.norm(context)

        if self.no_cuda:
            a = torch.from_numpy(np.diag(np.ones(seq_len - 1, dtype=np.int32), 1))
            b = torch.from_numpy(np.diag(np.ones(seq_len, dtype=np.int32), 0))
            c = torch.from_numpy(np.diag(np.ones(seq_len - 1, dtype=np.int32), -1))
            tri_matrix = torch.from_numpy(np.triu(np.ones([seq_len, seq_len], dtype=np.float32), 0))
        else:
            a = torch.from_numpy(np.diag(np.ones(seq_len - 1, dtype=np.int32), 1)).cuda()
            b = torch.from_numpy(np.diag(np.ones(seq_len, dtype=np.int32), 0)).cuda()
            c = torch.from_numpy(np.diag(np.ones(seq_len - 1, dtype=np.int32), -1)).cuda()
            tri_matrix = torch.from_numpy(np.triu(np.ones([seq_len, seq_len], dtype=np.float32), 0)).cuda()

        # mask = eos_mask & (a+c) | b
        mask = eos_mask & (a + c)

        key = self.linear_key(context)
        query = self.linear_query(context)

        scores = torch.matmul(query, key.transpose(-2, -1)) / self.d_model

        scores = scores.masked_fill(mask == 0, -1e9)
        neibor_attn = F.softmax(scores, dim=-1)
        neibor_attn = torch.sqrt(neibor_attn * neibor_attn.transpose(-2, -1) + 1e-9)

        t = torch.log(torch.tensor(neibor_attn + 1e-9)).masked_fill(a == 0, 0).matmul(tri_matrix)
        g_attn = tri_matrix.matmul(t).exp().masked_fill((tri_matrix.int() - b) == 0, 0)
        g_attn = g_attn + g_attn.transpose(-2, -1) + neibor_attn.masked_fill(b == 0, 1e-9)

        return g_attn, neibor_attn


class Actor(nn.Module):
    def __init__(self, expirement, d_model, dropout=0.8, no_cuda=False):
        super(Actor, self).__init__()
        self.expe = expirement
        self.target_policy = PolicyNetwork(d_model=d_model, dropout=dropout)

    def get_target_logOutput(self, h, x):
        out, _ = self.target_policy(h, x)
        logOut = torch.log(out)
        return logOut

    def get_target_output(self, context, mask):
        out, _ = self.target_policy(context, mask)
        return out

    def get_gradient(self, context, mask, reward, idx):
        out, _ = self.target_policy(context, mask)
        logout = torch.log(out)[0][idx].view(-1)
        index = reward.index(0)
        index = (index + 1) % 2
        # print(out, reward, index, logout[index].view(-1), logout)
        # print(logout[index].view(-1))
        grad = torch.autograd.grad(logout[index].view(-1), self.target_policy.parameters())  # torch.cuda.FloatTensor(reward[index])
        # print(grad[0].size(), grad[1].size(), grad[2].size())
        # print(grad[0], grad[1], grad[2])
        for i in range(len(grad)):
            grad[i].data = grad[i].data * reward[index]
        # print(grad[0], grad[1], grad[2])
        return grad, out

    def get_batch_gradient(self, context, mask, reward_batch):
        out, _ = self.target_policy(context, mask)
        logout = torch.log(out)
        for i, reward_sent in enumerate(reward_batch):
            for j, reward in enumerate(reward_sent):
                index = reward.index(0)
                logout[i][j][index] = -1
                if j == len(reward_sent)-1:
                    for k in range(j+1, logout.size()[1]):
                        logout[i][j][0] = -1
                        logout[i][j][1] = -1
        # print(out, reward, index, logout[index].view(-1), logout)
        # print(logout[index].view(-1))
        grad_output = torch.zeros_like(logout)
        for i, reward_sent in enumerate(reward_batch):
            for j, reward in enumerate(reward_sent):
                index = reward.index(0)
                index = (index + 1) % 2
                grad_output[i][j][index] = reward[index]
        grad = torch.autograd.grad(logout, self.target_policy.parameters(), grad_outputs=grad_output)  # torch.cuda.FloatTensor(reward[index])
        # print(grad[0].size(), grad[1].size(), grad[2].size())
        # print(grad[0], grad[1], grad[2])
        # print(grad[0], grad[1], grad[2])
        return grad, out

    def assign_network_gradients(self, grad):
        i = 0
        tau = 0.5
        for name, x in self.target_policy.named_parameters():
            x.grad = deepcopy(grad[i].data * (tau) + x.data * (1 - tau))
            i += 1