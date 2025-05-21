from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import copy
import math
import numpy as np
import torch.fft

from .tcn import SingleStageTCN
from .SP import MultiScale_GraphConv

import torch.nn.functional as F


def exponential_descrease(idx_decoder, p=3):
    return math.exp(-p*idx_decoder)

class Linear_Attention(nn.Module):
    def __init__(self,
                 in_channel,
                 n_features,
                 out_channel,
                 n_heads=4,
                 drop_out=0.05
                 ):
        super().__init__()
        self.n_heads = n_heads

        self.query_projection = nn.Linear(in_channel, n_features)
        self.key_projection = nn.Linear(in_channel, n_features)
        self.value_projection = nn.Linear(in_channel, n_features)
        self.out_projection = nn.Linear(n_features, out_channel) #64-64
        self.dropout = nn.Dropout(drop_out) #0.05 dropout

    def elu(self, x):
        return torch.sigmoid(x)
        # return torch.nn.functional.elu(x) + 1
        
    def forward(self, queries, keys, values, mask):

        B, L, _ = queries.shape
        _, S, _ = keys.shape
        queries = self.query_projection(queries).view(B, L, self.n_heads, -1) 
        keys = self.key_projection(keys).view(B, S, self.n_heads, -1)         
        values = self.value_projection(values).view(B, S, self.n_heads, -1)   
        
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2) #（n,head,t,c）

        queries = self.elu(queries)
        keys = self.elu(keys)
        KV = torch.einsum('...sd,...se->...de', keys, values) # （n,head,t,c）,（n,head,t,c）->(n,head,c,c)
        Z = 1.0 / torch.einsum('...sd,...d->...s',queries, keys.sum(dim=-2)+1e-6) #（n,head,t,c）,（n,head,c） ->(n,head,t)

        x = torch.einsum('...de,...sd,...s->...se', KV, queries, Z).transpose(1, 2) #(n,head,c,c),(n,head,t,c),(n,head,t)->(n,head,t,c)

        x = x.reshape(B, L, -1) #（n,t,c）
        x = self.out_projection(x)
        x = self.dropout(x) #0.05 dropout

        return x * mask[:, 0, :, None]

class FFT_conv_filter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FFT_conv_filter, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1x1_mag = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.conv1x1_mag2 = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        self.batchnorm = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x, mask):
        N, C, T = x.shape

        x_fft = torch.fft.rfft(x, dim=2)

        x_fft_real = x_fft.real
        x_fft_imag = x_fft.imag

        x_fft_real_imag = torch.cat([x_fft_real, x_fft_imag], dim=0)
        x_fft_real_imag = self.conv1x1_mag(x_fft_real_imag)
        x_fft_real_imag = self.conv1x1_mag2(x_fft_real_imag)

        x_fft_real, x_fft_imag = torch.chunk(x_fft_real_imag, 2, dim=0)

        out_fft = torch.complex(x_fft_real,x_fft_imag)

        out_ifft = torch.fft.irfft(out_fft, dim=2, n=T)
        out_ifft = out_ifft + x

        return out_ifft * mask


class CustomProcessing(nn.Module):
    def __init__(self, V):
        super(CustomProcessing, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((64, V))
        self.linear1 = nn.Linear(V, 4)  # (N, C, T, V) -> (N, C, T, 4)
        self.linear2 = nn.Linear(64, 4)  # (N, C, T, 4) -> (N, C, 4, 4)
        self.linear3 = nn.Linear(16, 4)  # (N, C, 16) -> (N, C, 4)

    def forward(self, x):
        x = self.pool(x)
        x = self.linear1(x)  # (N, C, T, V) -> (N, C, T, 4)
        x = self.linear2(x.transpose(2, 3))
        x = x.transpose(2, 3)  #  (N, C, 4, 4)
        x = x.flatten(2)  #  (N, C, 16)
        x = self.linear3(x)  # (N, C, 16) -> (N, C, 4)
        x = x.permute(2, 0, 1)
        return x

class FFT_filter(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size=64, joint_num=17):
        super(FFT_filter, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.choose_filter = CustomProcessing(joint_num)

        self.Gfilter1 = nn.Parameter(torch.ones(1, 1, 40, joint_num))
        self.Gfilter2 = nn.Parameter(torch.ones(1, 1, 48, joint_num))
        self.Gfilter3 = nn.Parameter(torch.ones(1, 1, 56, joint_num))
        self.Gfilter4 = nn.Parameter(torch.ones(1, 1, 64, joint_num))

        self.scale = nn.Parameter(torch.zeros(4, 1, 64, 1, 1))


    def forward(self, x, mask):
        N, C, T, V = x.shape

        x_fft = torch.fft.rfft(x, dim=2)

        N, C, T1, V  = x_fft.shape

        routeing = torch.abs(x)
        routeing = self.choose_filter(routeing)
        scale_dy = routeing.unsqueeze(-1).unsqueeze(-1)

        scale_dy = scale_dy.softmax(dim=0)

        expanded_filter1 = F.interpolate(self.Gfilter1, size=(T1,V), mode="nearest")
        out_fft1 = x_fft * expanded_filter1
        out_ifft1 = torch.fft.irfft(out_fft1, dim=2, n=T)

        expanded_filter2 = F.interpolate(self.Gfilter2, size=(T1,V), mode="nearest")
        out_fft2 = x_fft * expanded_filter2
        out_ifft2 = torch.fft.irfft(out_fft2, dim=2, n=T)

        expanded_filter3 = F.interpolate(self.Gfilter3, size=(T1,V), mode="nearest")
        out_fft3 = x_fft * expanded_filter3
        out_ifft3 = torch.fft.irfft(out_fft3, dim=2, n=T)

        expanded_filter4 = F.interpolate(self.Gfilter4, size=(T1,V), mode="nearest")
        out_fft4 = x_fft * expanded_filter4
        out_ifft4 = torch.fft.irfft(out_fft4, dim=2, n=T)

        scale_pa = self.scale.softmax(dim=0)

        out_ifft_dy = out_ifft1 * scale_dy[0] + out_ifft2 * scale_dy[1] + out_ifft3 * scale_dy[2] + out_ifft4 * scale_dy[3]
        out_ifft_pa = out_ifft1 * scale_pa[0] + out_ifft2 * scale_pa[1] + out_ifft3 * scale_pa[2] + out_ifft4 * scale_pa[3]

        out_ifft = (out_ifft_dy + out_ifft_pa) / 2

        out_ifft = (out_ifft + x)/2

        return out_ifft * mask.unsqueeze(dim=-1)






class AttModule(nn.Module):
    def __init__(self, dilation, in_channel, out_channel, stage, alpha):
        super(AttModule, self).__init__()
        self.stage = stage
        self.alpha = alpha

        self.feed_forward = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 3, padding=dilation, dilation=dilation),
            nn.ReLU()
            )
        self.instance_norm = nn.InstanceNorm1d(out_channel, track_running_stats=False)
        self.att_layer = Linear_Attention(out_channel, out_channel, out_channel)
        
        self.conv_out = nn.Conv1d(out_channel, out_channel, 1)
        self.dropout = nn.Dropout()
        
    def forward(self, x, f, mask):

        out = self.feed_forward(x) #一个时间卷积
        if self.stage == 'encoder':
            q = self.instance_norm(out).permute(0, 2, 1)
            out = self.alpha * self.att_layer(q, q, q, mask).permute(0, 2, 1) + out
        else:
            assert f is not None
            q = self.instance_norm(out).permute(0, 2, 1)
            f = f.permute(0, 2, 1)
            out = self.alpha * self.att_layer(q, q, f, mask).permute(0, 2, 1) + out #linear transformer
       
        out = self.conv_out(out)
        out = self.dropout(out)

        return (x + out) * mask

class SFI(nn.Module):
    def __init__(self, in_channel, n_features):
        super().__init__()
        self.conv_s = nn.Conv1d(in_channel, n_features, 1) #19->64
        self.softmax = nn.Softmax(dim=-1)
        self.ff = nn.Sequential(nn.Linear(n_features, n_features),
                                nn.GELU(),
                                nn.Dropout(0.3),
                                nn.Linear(n_features, n_features)) #64—>64
        self.conv_fusion = nn.Conv1d(2 * n_features, n_features, 1)
        
    def forward(self, feature_s, feature_t, mask): #feature_s for（n,t,v) feature_t for (n,t,c)
        n, c, t, v = feature_s.shape
        feature_s = feature_s.permute(0, 3, 1, 2).contiguous().view(n, v * c, t)  # (n,8,t,v) -->(n,v*8,t)
        feature_s = self.conv_s(feature_s) #(n,v,t)->(n,c,t)
        feature_cross = self.conv_fusion(torch.cat([feature_t, feature_s], dim=1))
        feature_cross = feature_cross.permute(0, 2, 1) #(n,t,c）
        feature_cross = self.ff(feature_cross).permute(0, 2, 1) + feature_t

        return feature_cross * mask
    
class STI(nn.Module):
    def __init__(self, node, in_channel, n_features, out_channel, num_layers, SFI_layer, channel_masking_rate=0.3, alpha=1):
        super().__init__()
        self.SFI_layer = SFI_layer #（1,2,3,4,5,6,7,8,9）
        num_SFI_layers = len(SFI_layer) #9
        self.channel_masking_rate = channel_masking_rate
        self.dropout = nn.Dropout2d(p=channel_masking_rate) #0.3 dropout

        self.conv_in = nn.Conv2d(in_channel, (num_SFI_layers+1)*8, kernel_size=1) #64->80
        self.conv_t = nn.Conv1d(node * 8, n_features, 1)
        self.SFI_layers = nn.ModuleList(
            [SFI(node * 8, n_features) for i in range(num_SFI_layers)])
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, n_features, n_features, 'encoder', alpha) for i in 
                range(num_layers)]) #10层扩张注意力
        self.fft_layers = nn.ModuleList(
            [FFT_conv_filter(n_features, n_features) for i in range(num_layers)])


        self.conv_out = nn.Conv1d(n_features, out_channel, 1)

    def forward(self, x, mask):
        if self.channel_masking_rate > 0:
            x = self.dropout(x) #0.3

        count = 0
        x = self.conv_in(x) #c=64->80
        feature_s, feature_t = torch.split(x, (len(self.SFI_layers)*8, 8), dim=1) #（n,80,t,v）->(n,72,t,v)+(n,8,t,v)

        N, C, T ,V = feature_t.shape
        feature_t = feature_t.permute(0, 3, 1, 2).contiguous().view(N, V*C, T) #(n,8,t,v) -->(n,v*8,t)
        feature_st = self.conv_t(feature_t) #(n,v,t)->(n,64,t)

        for index, (layer, fft_layer) in enumerate(zip(self.layers, self.fft_layers)):
            if index in self.SFI_layer:
                feature_st =  self.SFI_layers[count](feature_s[:,count*8:(count+1)*8,:], feature_st, mask)
                count+=1
            feature_st = layer(feature_st, None, mask)
            feature_st = fft_layer(feature_st, mask)


        feature_st = self.conv_out(feature_st)
        return feature_st * mask
       
class Decoder(nn.Module):
    def __init__(self, in_channel, n_features, out_channel, num_layers, alpha=1):
        super().__init__()
        
        self.conv_in = nn.Conv1d(in_channel, n_features, 1)
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, n_features, n_features, 'decoder', alpha) for i in 
             range(num_layers)])
        self.conv_out = nn.Conv1d(n_features, out_channel, 1)

    def forward(self, x, fencoder, mask):
        feature = self.conv_in(x)
        for layer in self.layers:
            feature = layer(feature, fencoder, mask)
        out = self.conv_out(feature)
        
        return out, feature


    
class Model(nn.Module):
    """
    this model predicts both frame-level classes and boundaries.
    Args:
        in_channel: 
        n_feature: 64
        n_classes: the number of action classes
        n_layers: 10
    """

    def __init__(
        self,
        in_channel: int,
        n_features: int,
        n_classes: int,
        n_stages: int,
        n_layers: int,
        n_refine_layers: int,
        n_stages_asb: Optional[int] = None,
        n_stages_brb: Optional[int] = None,
        SFI_layer: Optional[int] = None,
        dataset: str = None,
        **kwargs: Any
    ) -> None:

        if not isinstance(n_stages_asb, int):
            n_stages_asb = n_stages

        if not isinstance(n_stages_brb, int):
            n_stages_brb = n_stages

        super().__init__()


        self.in_channel = in_channel
        if dataset == "LARA":
            node = 19
        elif dataset == "TCG-15":
            node = 17
        else:
            node = 25

        self.dataset = dataset
        self.logit_scale = nn.Parameter(torch.ones(2) * np.log(1 / 0.07))  # 2.6593

        self.SP = MultiScale_GraphConv(13, in_channel, n_features, dataset)   #multi-scale
        self.STI = STI(node, n_features, n_features, n_features, n_layers, SFI_layer)

        self.fft_filter = FFT_filter(n_features, n_features, joint_num=node)

        if dataset == "TCG-15":
            self.conv_freq = nn.Conv2d(n_features, 64, 1)

        self.conv_cls = nn.Conv1d(n_features, n_classes, 1) #cls head
        self.conv_bound = nn.Conv1d(n_features, 1, 1) #boundary head
        self.conv_feature = nn.Conv1d(n_features, 768, 1)  # feature head

        # action segmentation branch
        asb = [
            copy.deepcopy(Decoder(n_classes, n_features, n_classes, n_refine_layers, alpha=exponential_descrease(s))) for s in range(n_stages_asb - 1)
        ]
        conv_asb_feature = [nn.Conv1d(n_features, 768, 1) for s in range(n_stages_asb - 1)]
        # boundary regression branch
        brb = [
            SingleStageTCN(1, n_features, 1, n_refine_layers) for _ in range(n_stages_brb - 1)
        ]
        self.asb = nn.ModuleList(asb)
        self.brb = nn.ModuleList(brb)
        self.conv_asb_feature = nn.ModuleList(conv_asb_feature)

        self.activation_asb = nn.Softmax(dim=1)
        self.activation_brb = nn.Sigmoid()

    def forward(self, x: torch.Tensor, mask: torch.Tensor, joint_graph) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.SP(x, joint_graph) * mask.unsqueeze(3) # （n,c,t,v） spatial modeling

        x = self.fft_filter(x, mask) # frequency modeling

        if self.dataset == "TCG-15":
            freq_feature = self.conv_freq(x)
        else:
            freq_feature = x

        feature = self.STI(x, mask) #temporal modeling
        
        out_cls = self.conv_cls(feature) #cls
        out_bound = self.conv_bound(feature) #boundary
        out_feature = self.conv_feature(feature)  # feature
        
        if self.training:
            outputs_cls = [out_cls]
            outputs_bound = [out_bound]
            outputs_feature = [out_feature]

            for as_stage, conv_stage in zip(self.asb, self.conv_asb_feature):
                out_cls, feature = as_stage(self.activation_asb(out_cls) * mask, feature * mask, mask)
                out_feature = conv_stage(feature)
                outputs_cls.append(out_cls)
                outputs_feature.append(out_feature)

            for br_stage in self.brb:
                out_bound = br_stage(self.activation_brb(out_bound), mask)
                outputs_bound.append(out_bound)

            return (outputs_cls, outputs_bound, outputs_feature, freq_feature, self.logit_scale)
        else:
            for as_stage in self.asb:
                out_cls, _ = as_stage(self.activation_asb(out_cls)* mask, feature* mask, mask)

            for br_stage in self.brb:
                out_bound = br_stage(self.activation_brb(out_bound), mask)

            return (out_cls, out_bound)
