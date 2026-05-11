import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Attention2(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention2, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

    def forward(self, x, isNorm=True):
        ## x: N x L
        A = self.attention(x)  ## N x K
        A = torch.transpose(A, 1, 0)  # KxN
        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N
        return A  ### K x N


class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N

        return A  ### K x N


class Attention_Gated_Sharp(nn.Module):
    """
    改进版 Gated Attention：
    1. V-branch 将 Tanh 替换为 LayerNorm + GELU，避免大值饱和导致 patch 不可区分
    2. 可配置温度 tau（<1 时锐化 softmax，使注意力更集中）
    """

    def __init__(self, L=512, D=128, K=1, tau=0.5):
        super().__init__()
        self.L = L
        self.D = D
        self.K = K
        self.tau = tau

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.LayerNorm(self.D),
            nn.GELU(),
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid(),
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        A_V = self.attention_V(x)
        A_U = self.attention_U(x)
        A = self.attention_weights(A_V * A_U)
        A = torch.transpose(A, 1, 0)
        if isNorm:
            A = F.softmax(A / self.tau, dim=1)
        return A


class Attention_with_Classifier(nn.Module):
    def __init__(self, L=512, D=128, K=1, num_cls=2, droprate=0):
        super(Attention_with_Classifier, self).__init__()
        self.attention = Attention_Gated(L, D, K)
        self.classifier = Classifier_1fc(L, num_cls, droprate)
    def forward(self, x): ## x: N x L
        AA = self.attention(x)  ## K x N
        afeat = torch.mm(AA, x) ## K x L
        pred = self.classifier(afeat) ## K x num_cls
        return pred

class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x


class residual_block(nn.Module):
    def __init__(self, nChn=512):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
            )
    def forward(self, x):
        tt = self.block(x)
        x = x + tt
        return x


class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x

class DimReduction_MoE(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x

class MHAtt_to_scalar(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.proj = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x: (N, D) or (B, N, D)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, N, D)
        attn_out, _ = self.attn(x, x, x)    # (B, N, D)
        out = self.proj(attn_out).squeeze(-1)  # (B, N)
        if out.size(0) == 1:
            return out.squeeze(0)  # (N,)
        return out




class ABMIL(nn.Module):

    def __init__(self, input_Dim=1024, mDim=512, num_cls=2, numLayer_Res=0, droprate=0,
                 use_sharp_attn=False, attn_tau=0.5):
        super().__init__()
        if use_sharp_attn:
            self.attention = Attention_Gated_Sharp(mDim, D=max(256, mDim // 4), tau=attn_tau)
        else:
            self.attention = Attention_Gated(mDim)
        self.dimReduction = DimReduction(input_Dim, mDim, numLayer_Res=numLayer_Res)
        self.classifier = Classifier_1fc(mDim, num_cls, droprate)

    def forward(self, data):
        inputs_tensor=data
        tmidFeat = self.dimReduction(inputs_tensor).squeeze(0)
        tAA = self.attention(tmidFeat).squeeze(0)
        # with open('slide_attention.txt', 'a') as f:
        #         f.write(f'ABMIL , slide_attention:{tAA}\n')
        tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
        tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
        tattFeat_tensor = tattFeat_tensor.unsqueeze(0)
        return tattFeat_tensor




if __name__ == "__main__":
    data = torch.randn((1, 6000, 1024)).cuda()
    model = ABMIL(mDim=64).cuda()
    results_dict = model(data)
    print(results_dict.shape)