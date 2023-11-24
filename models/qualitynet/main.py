""" Module implementing model """
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch import Tensor
import torchaudio
from coral_pytorch.dataset import corn_label_from_logits

from models.qualitynet.attention_pool import MultiheadAttentionPool

class PositionalEncoding(nn.Module):
    """
    Positional Encoding proposed in "Attention Is All You Need".
    Since transformer contains no recurrence and no convolution, in order for the model to make
    use of the order of the sequence, we must add some positional information.

    "Attention Is All You Need" use sine and cosine functions of different frequencies:
        PE_(pos, 2i)    =  sin(pos / power(10000, 2i / d_model))
        PE_(pos, 2i+1)  =  cos(pos / power(10000, 2i / d_model))
    """
    def __init__(self, d_model: int = 512, max_len: int = 40000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)    # pe = (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, length: int) -> Tensor:
        """get positional encoding

        Args:
            length (int): sequence length

        Returns:
            Tensor: positional encoding of shape (1, length, d_model)
        """
        return self.pe[:, :length]


@dataclass
class QualityNetConfig:
    """ SignalQuality parameters

    Args:
        freeze_pretrain (bool): freeze WavLM layer
        n_head (int): number of heads in attention pooling
        dropout (float): dropout layer
        model_type (str): if model uses 'regression' or 'CORN'
    """
    freeze_pretrain:bool
    n_head:int
    dropout:float
    model_type:str


class QualityNet(nn.Module):
    """Signal Quality model"""
    def __init__(self, params:QualityNetConfig):
        super().__init__()
        self.params = params

        # get base model
        bundle = torchaudio.pipelines.WAVLM_BASE_PLUS
        self.wavlm = bundle.get_model()
        if params.freeze_pretrain:
            self.freeze_model()
        final_dim = 768

        # attention
        self.pos_enc = PositionalEncoding(d_model=final_dim, max_len=1200)
        self.attn = MultiheadAttentionPool(n_head=params.n_head, n_dim=final_dim)

        # final layer
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=params.dropout)
        match params.model_type:
            case 'regression': self.fc = nn.Linear(final_dim, 1)
            case 'CORN': self.fc = nn.Linear(final_dim, 4)

    def forward(self, x:Tensor) -> Tensor:
        """
        Args:
            x (Tensor): (batch, seq)

        Returns:
            Tensor: (batch) if regression, (batch, class) if classification
        """
        x, _ = self.wavlm(x)    # [batch, seq, feat]

        # attention
        _, seq, _ = x.shape
        x += self.pos_enc(seq)
        x, w = self.attn(x)       # [batch, feat]

        x = self.dropout(self.activation(x))
        x = self.fc(x)          # [batch, out]
        if self.params.model_type == 'regression':
            x = torch.clamp(x, min=0, max=4)    # clamp if its regression model
            return torch.squeeze(x, dim=1), w
        return x, w

    def freeze_model(self):
        """freeze pretrain model"""
        for p in self.wavlm.parameters():
            p.requires_grad = False

    def get_label(self, x:Tensor) -> Tensor:
        """waveform to label

        Args:
            x (Tensor): input waveform (batch, seq)

        Returns:
            Tensor: class prediction (batch,)
        """
        x, _ = self(x)
        match self.params.model_type:
            case 'regression':
                return torch.round(x)
            case 'CORN':
                return corn_label_from_logits(x).float()
