from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdMSoftmaxLoss(nn.Module):

    def __init__(self, s=1, m=0.4):
        '''
        AM Softmax Loss
        '''
        super(AdMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m

    def forward(self, x1, x2, sentence_id):
        """
        x1,x2 shape (N, in_features)
        """
        N = x1.size()[0]
        x12 = F.normalize(torch.matmul(x1, x2.transpose(0, 1)), dim=1)   # (N,N)
        x21 = F.normalize(torch.matmul(x2, x1.transpose(0, 1)), dim=1)  # (N,N)
        labels = sentence_id.cpu().numpy().tolist()

        loss1 = []
        loss2 = []
        loss = 0.0
        more_label = False
        label = []
        for i in range(N):
            label_1 = []
            for j,l in enumerate(labels):
                if labels[i] == l:
                    label_1.append(j)
            label.append(label_1)
            if len(label_1) > 1:
                more_label = True
        if not more_label:
            numerator = self.s * (torch.diagonal(x12) - self.m)   # (1,N)
            excl = torch.cat([torch.cat((x12[i, :y[0]], x12[i, y[0] + 1:])).unsqueeze(0) for i, y in enumerate(label)], dim=0)  #(N,N-1)
            denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)  #
            L12 = numerator - torch.log(denominator)

            numerator = self.s * (torch.diagonal(x21) - self.m)
            excl = torch.cat([torch.cat((x21[i, :y[0]], x21[i, y[0] + 1:])).unsqueeze(0) for i, y in enumerate(label)], dim=0)
            denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
            L21 = numerator - torch.log(denominator)
        else:
            numerator = self.s * (torch.cat([torch.sum(torch.exp(x12[i,y] - self.m), dim=0).view(-1,) for i,y in enumerate(label)], dim=0))
            excl = self.s * (torch.cat([torch.sum(torch.exp(x12[i,list(set([k for k in range(N)]).difference(set(y)))]), dim=0).view(-1,) for i,y in enumerate(label)], dim=0))
            denominator = numerator + excl
            L12 = torch.log(numerator) - torch.log(denominator)

            numerator = self.s * (
                torch.cat([torch.sum(torch.exp(x21[i, y] - self.m), dim=0).view(-1,) for i, y in enumerate(label)], dim=0))
            excl = self.s * (torch.cat(
                [torch.sum(torch.exp(x21[i, list(set([k for k in range(N)]).difference(set(y)))]), dim=0).view(-1,) for i, y in
                 enumerate(label)], dim=0))
            denominator = numerator + excl
            L21 = torch.log(numerator) - torch.log(denominator)

        loss1 = L12
        loss2 = L21
        loss = -torch.mean(L12)-torch.mean(L21)
        return loss, loss1, loss2


class StructTransformer(nn.Module):

    def __init__(self, experiment, config_class, model_class):
        super(StructTransformer, self).__init__()
        self.expe = experiment

        self.config = config_class.from_pretrained(self.expe.ml_model_path)
        self.model_dim = self.config.hidden_size
        self.encoder = model_class.from_pretrained(self.expe.ml_model_path)
        self.admSoftmaxLoss = AdMSoftmaxLoss(s=self.expe.s, m=self.expe.margin)
        self.sentence_type = self.expe.sentence_type

        #classifier_dropout = (
        #    self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob
        #)
        #self.dropout = nn.Dropout(classifier_dropout)

    def mean_pool_embedding(self, embeds, masks):
        """
        Args:
          embeds: list of torch.FloatTensor, (B, L, D)
          masks: torch.FloatTensor, (B, L)
        Return:
          sent_emb: list of torch.FloatTensor, (B, D)
      """
        embeds = (embeds * masks.unsqueeze(2).float()).sum(dim=1) / masks.sum(dim=1).view(-1, 1).float()
        return embeds

    def forward(self, input1, input2, mask1, mask2, action1, action2, sentence_id, print_sentence=False):
        output1 = self.encoder(input1.long(), attention_mask=mask1, struct_action_probs=action1)
        output2 = self.encoder(input2.long(), attention_mask=mask2, struct_action_probs=action2)
        if print_sentence:
            return output1, output2
        if self.sentence_type == 'mean':
            input1 = output1['last_hidden_state'].mean(dim=1)
            input2 = output2['last_hidden_state'].mean(dim=1)
        elif self.sentence_type == 'pool_mean':
            input1 = self.mean_pool_embedding(output1['last_hidden_state'], mask1)
            input2 = self.mean_pool_embedding(output2['last_hidden_state'], mask2)
        elif self.sentence_type == 'pool':
            input1 = output1['pooler_output']
            input2 = output2['pooler_output']

        #sequence1_output = self.dropout(input1)
        #sequence2_output = self.dropout(input2)
        loss = self.admSoftmaxLoss(input1, input2, sentence_id)

        return loss, (input1, input2)


class Critic(nn.Module):
    def __init__(self, expirement, config_class, model_class):
        super(Critic, self).__init__()
        self.expe = expirement
        self.target_pred = StructTransformer(expirement, config_class, model_class)

    def forward(self, input1, input2, mask1, mask2, action1, action2, sentence_id, print_sentence=False):
        out = self.target_pred(input1, input2, mask1, mask2, action1, action2, sentence_id, print_sentence=print_sentence)
        return out