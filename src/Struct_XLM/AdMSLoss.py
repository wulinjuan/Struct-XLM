import torch
import torch.nn as nn


class AdMSoftmaxLoss(nn.Module):

    def __init__(self, s=1, m=0.4):
        '''
        AM Softmax Loss
        '''
        super(AdMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m

    def forward(self, x1, x2):
        """
        x1,x2 shape (N, in_features)
        """
        N = x1.size()[0]
        x12 = torch.matmul(x1, x2.transpose(0, 1))   # (N,N)
        x21 = torch.matmul(x2, x1.transpose(0, 1))  # (N,N)
        labels = [i for i in range(N)]

        numerator = self.s * (torch.diagonal(x12) - self.m)   # (1,N)
        excl = torch.cat([torch.cat((x12[i, :y], x12[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)  #(N,N-1)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)  #
        L12 = numerator - torch.log(denominator)

        numerator = self.s * (torch.diagonal(x21) - self.m)
        excl = torch.cat([torch.cat((x21[i, :y], x21[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L21 = numerator - torch.log(denominator)

        return -torch.mean(L12)-torch.mean(L21)
