""" MaskNet Model """
from typing import Tuple, Optional, Union
from dataclasses import dataclass

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.distributions.normal import Normal

from models.masknet.mask_generator import MaskGenerator, MaskGenConfig


@dataclass
class MaskNetConfig:
    """ configuration for MaskNet

    Args:
        num_sources (int): number of output sources. Defaults to 2
        kernel_size (int): convolution kernel size.
        num_feats (int): feature vectors size.
        stochastic (bool): set if model is stochastic. Defaults to False
    """
    kernel_size:int
    num_feats:int
    maskgen:Union[MaskGenConfig, dict]
    stochastic:bool=False
    num_sources:int=2

    def __post_init__(self):
        assert self.kernel_size%2 == 1, f'kernel size should be odd, got {self.kernel_size}'
        self.stride = self.kernel_size//2
        if isinstance(self.maskgen, dict):
            self.maskgen = MaskGenConfig(input_dim=self.num_feats, num_sources=self.num_sources,
                                         **self.maskgen)


class MaskNet(nn.Module):
    """ MaskNet Model """
    def __init__(self, config:MaskNetConfig):
        super().__init__()
        self.config = config

        # ----------------- sub-module ---------------------
        self.encoder = self.get_encoder()
        self.mask_generator = MaskGenerator(config.maskgen)
        self.decoder = self.get_decoder()
        self.decoder_var = self.get_decoder(bias=True) if config.stochastic else None

    def get_encoder(self) -> nn.Module:
        """ get encoder """
        num_feats = self.config.num_feats
        kernel_size = self.config.kernel_size
        stride = self.config.stride
        # padding = self.config.kernel_size//2
        padding = 0

        enc = nn.Conv1d(in_channels=1, out_channels=num_feats, kernel_size=kernel_size,\
                        stride=stride, padding=padding, bias=False)
        return enc

    def get_decoder(self, bias:bool=False) -> nn.Module:
        """ get decoder """
        num_feats = self.config.num_feats
        kernel_size = self.config.kernel_size
        stride = self.config.stride
        # padding = self.config.kernel_size//2
        padding = 0

        dec = nn.ConvTranspose1d(in_channels=num_feats, out_channels=1, kernel_size=kernel_size,\
                                 stride=stride, padding=padding,bias=bias)
        return dec


    def _pad_seq(self, x:Tensor) -> Tuple[Tensor, int]:
        """ pad input sequence such that trans_conv(conv(x)) yields the same shape

        Args:
            x (Tensor): input sequence

        Returns:
            Tuple[Tensor, int]: padded input sequence and the amount of padding
        """
        batch, channel, seq_len = x.shape
        stride = self.config.stride
        num_strides = (seq_len-1)//stride
        rem_in_seq = seq_len - (num_strides*stride + 1) # remainder in seq that is not convolved
        if rem_in_seq == 0:
            return x, 0

        padding_req = stride - rem_in_seq   # required amount of padding
        pad = torch.zeros(
            batch,
            channel,
            padding_req,
            dtype=x.dtype,
            device=x.device,
        )
        return torch.cat([x, pad], dim=2), padding_req

    def forward_module(self, x:Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """ forward call before sampling

        Args:
            x (Tensor): input sequence of shape (B, 1, T)

        Returns:
            Tuple[Tensor, Optional[Tensor]]: mean and var if stochastic, else var is None
        """
        x, num_pads = self._pad_seq(x)             # B, 1, T'

        x = self.encoder(x)                # B, F, M
        x = self.mask_generator(x)*x.unsqueeze(1)  # B, S, F, M

        if self.decoder_var is not None:
            var = self.decoder_forward(x, num_pads, self.decoder_var)
            var = F.softplus(var, beta=1, threshold=20) + 1.e-3   # pylint: disable=not-callable
        else:
            var = None
        x = self.decoder_forward(x, num_pads, self.decoder) # B, S, T
        return x, var

    def decoder_forward(self, x:torch.Tensor, num_pads:int, decoder:nn.Module) -> Tensor:
        """decoder feedforward """
        num_sources = self.config.num_sources
        num_feats = self.config.num_feats
        batch = x.shape[0]

        x = x.view(batch*num_sources, num_feats, -1)  # B*S, F, M
        x = decoder(x)                     # B*S, 1, L
        x = x.view(batch, num_sources, -1)          # B, S, L
        if num_pads > 0:
            x = x[..., :-num_pads]
        return x

    def forward(self, x:Tensor) -> Tensor:
        """ forward call for the model

        Args:
            x (Tensor): input sequence of shape (B, 1, T)

        Returns:
            Tensor: separated sequence of shape (B, S, T)
        """
        x, var = self.forward_module(x)
        if var is not None:
            m = Normal(loc=x, scale=var)
            x = m.rsample()    
        return x
    