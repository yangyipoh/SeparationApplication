""" Attention Pooling Module """
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F


class MultiheadAttentionPool(nn.Module):
    """Implementation of https://arxiv.org/pdf/2008.01077v1.pdf with multihead"""
    def __init__(self, n_head:int, n_dim:int):
        super().__init__()
        self.n_head = n_head
        self.q_proj = nn.Linear(n_dim, 1)
        self.k_proj = nn.Linear(n_dim, n_dim*n_head, bias=False)
        self.v_proj = nn.Linear(n_dim, n_dim*n_head)
        self.out = nn.Linear(n_dim*n_head, n_dim)

    def forward(self, x:Tensor) -> Tensor:
        """
        Args:
            x (Tensor): (batch, seq, feat)

        Returns:
            Tensor: (batch, feat)
        """
        # x = [batch, seq, feat]
        batch, seq, feat = x.shape
        scale = 1/(feat**-0.5)
        k = self.k_proj(x)  # [batch, seq, feat*head]
        k = k.view(batch, seq, self.n_head, feat).permute(0, 2, 1, 3)   # [batch, n_head, seq, feat]
        v = self.v_proj(x)  # [batch, seq, feat*head]
        v = v.view(batch, seq, self.n_head, feat).permute(0, 2, 1, 3)   # [batch, n_head, seq, feat]

        qk = self.q_proj(k).permute(0, 1, 3, 2)*scale   # QK^T/sqrt(d) [batch, n_head, 1, seq]
        w = F.softmax(qk, dim=-1).to(v.dtype)   # softmax(QK^T/sqrt(d)) [batch, n_head, 1, seq]

        # [batch, n_head, 1, feat] -> [batch, n_head, feat] -> [batch, n_head*feat]
        wv = torch.squeeze(w @ v, dim=2).view(batch, self.n_head*feat)
        wv = self.out(wv)   # [batch, feat]
        return wv, w
