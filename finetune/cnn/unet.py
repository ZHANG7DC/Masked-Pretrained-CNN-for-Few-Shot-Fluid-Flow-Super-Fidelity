# Portions of this code adapted from:
#   Keyu Tian et al., “Designing BERT for Convolutional Networks: Sparse and Hierarchical Masked Modeling” (SparK, 2023)
#   Repository: https://github.com/keyu-tian/SparK
#   License: MIT
#
# Modified 2025
import math
from typing import List

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

#from utils.misc import is_pow2n
from timm import create_model
def is_pow2n(x):
    return x > 0 and (x & (x - 1) == 0)
class UNetBlock(nn.Module):
    def __init__(self, cin, cout, bn2d):
        """
        a UNet block with 2x up sampling
        """
        super().__init__()
        self.up_sample = nn.ConvTranspose2d(cin, cin, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cin, kernel_size=3, stride=1, padding=1, bias=False), bn2d(cin), nn.ReLU6(inplace=True),
            nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False), bn2d(cout),
        )
    
    def forward(self, x):
        x = self.up_sample(x)
        return self.conv(x)


class LightDecoder(nn.Module):
    def __init__(self, up_sample_ratio, width=768, sbn=True, out_chans=8):   # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
        super().__init__()
        self.width = width
        assert is_pow2n(up_sample_ratio)
        n = round(math.log2(up_sample_ratio))
        channels = [self.width // 2 ** i for i in range(n + 1)] # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
        bn2d = nn.SyncBatchNorm if sbn else nn.BatchNorm2d
        self.dec = nn.ModuleList([UNetBlock(cin, cout, bn2d) for (cin, cout) in zip(channels[:-1], channels[1:])])
        self.proj = nn.Conv2d(channels[-1], out_chans, kernel_size=1, stride=1, bias=True)
        
        self.initialize()
    
    def forward(self, to_dec: List[torch.Tensor]):
        x = 0
        for i, d in enumerate(self.dec):
            if i < len(to_dec) and to_dec[i] is not None:
                x = x + to_dec[i]
            x = self.dec[i](x)
        return self.proj(x)
    
    def extra_repr(self) -> str:
        return f'width={self.width}'
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

class UNet(nn.Module):
    def __init__(self, in_chans, out_chans, encoder_name='resnet50', decoder_width=768, sbn=False):
        super().__init__()

        # --- Create Encoder ---
        kwargs = dict(
            drop_path_rate=0.05,
            pretrained=False,
            num_classes=0,
            global_pool='',
            in_chans=in_chans,
            features_only=True
        )
        self.encoder = create_model(encoder_name, **kwargs)

        # --- Feature Info ---
        feature_info = self.encoder.feature_info
        self.enc_channels = [info['num_chs'] for info in feature_info]  # [256, 512, 1024, 2048]

        # --- Create projections to match decoder width hierarchy ---
        self.projs = nn.ModuleList([
            #nn.Conv2d(64, 48, kernel_size=3, padding=1),
            nn.Conv2d(256, 96, kernel_size=3, padding=1),
            nn.Conv2d(512, 192, kernel_size=3, padding=1),
            nn.Conv2d(1024, 384, kernel_size=3, padding=1),
            nn.Conv2d(2048, 768, kernel_size=1)            
        ])

        # --- Decoder ---
        self.decoder = LightDecoder(
            up_sample_ratio=32,  # from encoder's deepest feature to output size
            width=decoder_width,
            sbn=True,
            out_chans=out_chans
        )

    def forward(self, x):
        # --- Get multi-scale features from encoder ---
        feats = self.encoder(x)[1:]  # list of feature maps from shallow to deep (4 stages)
        # --- Project encoder features to decoder widths ---
        to_dec = []
        for feat, proj in zip(feats[::-1], self.projs[::-1]):  # reverse: deepest to shallowest
            to_dec.append(proj(feat))
        to_dec
        # --- Decode ---
        out = self.decoder(to_dec)
        return out

