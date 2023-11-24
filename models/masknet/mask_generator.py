""" Mask Generator Model """
from dataclasses import dataclass

import torch
from torch import nn
from torch import Tensor

from models.masknet.transformer import TransformerEncoder
from models.masknet.deep_encoder import DeepEncoder


@dataclass
class MaskGenConfig:
    """ configuration for MaskGenerator

    Args:
        input_dim (int): input dimension
        num_sources (int): output number of sources
        feat_dim (int): feature dimension
        conv_kernel_size (int): kernel size
        conv_layers (int): convolution layers
        individual_mask (bool): use of 1 transformer for each source
        mask_type (str): use of transformer, transformer_relative, conformer
        num_heads (int): number of heads in MHA in transformer
        ffn_dim_expand (int): feedforward model expansion
        num_layers (int): transformer layers
        dropout (int): use of dropout
    """
    input_dim:int
    feat_dim:int
    conv_kernel_size:int
    conv_layers:int
    individual_mask:bool
    trans_type:str
    trans_num_heads:int
    trans_ffn_expand:int
    trans_num_layers:int
    dropout:int
    num_sources:int=2


class MaskGenerator(nn.Module):
    """ Mask Generator Model """
    def __init__(self, config:MaskGenConfig):
        super().__init__()
        self.config = config

        input_dim = config.input_dim
        feat_dim = config.feat_dim
        self.input_norm = nn.GroupNorm(num_groups=input_dim, num_channels=input_dim, eps=1e-8)
        self.input_conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=feat_dim, kernel_size=1),
            DeepEncoder(
                in_channels=feat_dim,
                out_channels=feat_dim,
                kernel_size=config.conv_kernel_size,
                num_layers=config.conv_layers,
            ),
        )

        if config.individual_mask:
            self.mask_generator = nn.ModuleList([
                self.get_mask_type() for _ in range(config.num_sources)])
            self.output_conv = nn.ModuleList([
                self.get_output_conv(config.input_dim) for _ in range(config.num_sources)])
        else:
            self.mask_generator = self.get_mask_type()
            self.output_conv = self.get_output_conv(config.input_dim*config.num_sources)

        self.output_activation = nn.GELU()
        self.mask_activation = nn.ReLU()

    def get_mask_type(self) -> nn.Module:
        """ get transformer mask """
        num_feats = self.config.feat_dim
        num_layers = self.config.trans_num_layers
        num_heads = self.config.trans_num_heads
        ffn_expand = self.config.trans_ffn_expand
        dropout = self.config.dropout
        match self.config.trans_type:
            case 'transformer':
                return TransformerEncoder(encoder_dim=num_feats, num_layers=num_layers,
                    num_attention_heads=num_heads, ffn_expansion=ffn_expand, dropout=dropout)
            case _:
                raise ValueError(f'Unknown Mask Type, got {self.config.trans_type}')

    def get_output_conv(self, out_channel:int) -> nn.Module:
        """ get output convolution """
        num_feats = self.config.feat_dim
        return nn.Conv1d(in_channels=num_feats, out_channels=out_channel, kernel_size=1)

    def forward(self, x:Tensor) -> Tensor:
        """ model feedforward

        Args:
            x (Tensor): 3D Tensor with shape (B, F, T)

        Returns:
            Tensor: 4D Tensor with shape (B, S, F, T)
        """
        x = self.input_conv(self.input_norm(x))  # (B, num_feat, T)
        x = torch.permute(x, (0, 2, 1))          # (B, T, num_feat)

        # if not using individual mask
        if not self.config.individual_mask:
            x = self.mask_generator(x)
            x = torch.permute(x, (0, 2, 1))   # [B, num_feat, T]

            x = self.output_activation(x)
            x = self.output_conv(x)   # [B, F*S, T]
            x = self.mask_activation(x)

            batch = x.shape[0]
            return x.view(batch, self.config.num_sources, self.config.input_dim, -1) # [B, S, F, T]

        # if using individual mask
        out = []
        for i in range(self.config.num_sources):
            xi = self.mask_generator[i](x)
            xi = torch.permute(xi, (0, 2, 1))

            xi = self.output_activation(xi)
            xi = self.output_conv[i](xi)
            out.append(xi)
        return torch.stack(out, dim=1)
