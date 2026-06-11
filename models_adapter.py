# modified from: https://github.com/openai/CLIP/blob/a9b1bf5920416aaeaec965c25dd9e8f98c864f16/clip/model.py

from operator import truediv
from typing import Tuple
from collections import OrderedDict
import math
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import (
    CLIP_VIT_B16_PATH,
    CLIP_VIT_L14_PATH,
    DWCONV3D_DISABLE_CUDNN,
    )


class TemporalFreqAdapter(nn.Module):
    def __init__(self, in_channels, adapter_channels, windows=(16, 8, 4), kernel_size=(3,3,3),skew_symmetric=False,use_stft=False,use_temporal_conv=True, use_frequency_conv=True, fusion_mode = 'concat'):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        if fusion_mode == 'gate':
            self.fc2 = nn.Linear(adapter_channels//2, in_channels)
        else:
            self.fc2 = nn.Linear(adapter_channels, in_channels)

        self.fusion_mode = fusion_mode
        self.windows = windows
        self.Ca = adapter_channels
        if use_temporal_conv and use_frequency_conv:
            Ct = adapter_channels // 2  # temporal half
            Cf = adapter_channels - Ct  # frequency half
        elif use_temporal_conv:
            Ct = adapter_channels
            Cf = 0
        elif use_frequency_conv:
            Ct = 0
            Cf = adapter_channels
        else:
            raise ValueError("Either use_temporal_conv or use_frequency_conv must be True")
        self.use_temporal_conv = use_temporal_conv
        self.use_frequency_conv = use_frequency_conv
        # Temporal branch: simple Conv3D over (T, H, W)
        if self.use_temporal_conv:
            self.temporal_conv = nn.Conv3d(
                in_channels=Ct,
                out_channels=Ct,
                kernel_size=kernel_size,
                padding=tuple(k//2 for k in kernel_size),
                groups=Ct  # depthwise
            )

        if self.use_frequency_conv:
            # Frequency branch: shared Conv3D over FFT
            self.freq_conv = nn.Conv3d(
                in_channels=2*Cf,
                out_channels=2*Cf,
                kernel_size=kernel_size,
                padding=tuple(k//2 for k in kernel_size),
                groups=2*Cf
            )
        if self.use_temporal_conv and self.use_frequency_conv:
            if self.fusion_mode == 'linear':
                self.fuse_proj = nn.Linear(Ct + Cf, Ct + Cf)
                self.fuse_activation = nn.GELU()
                nn.init.constant_(self.fuse_proj.bias, 0.)
            elif self.fusion_mode == 'gate':
                self.gate = nn.Parameter(torch.zeros(1, 1, 1, 1, min(Ct, Cf)))
            elif self.fusion_mode == 'concat':
                pass
            else:
                raise ValueError(f"Invalid fusion mode: {self.fusion_mode}")
       # init
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)
        if self.use_temporal_conv:
            nn.init.constant_(self.temporal_conv.bias, 0.)

        if self.use_frequency_conv:
            nn.init.constant_(self.freq_conv.bias, 0.)

    def forward(self, x, T):
        """
        x: (B*T, L, C)
        T: number of frames
        """       
        if T == 8:
            self.windows = [8,4,2]
        elif T == 32:
            self.windows = [32,16,8]
        elif T == 16:
            self.windows = [16,8,4]
        else:
            raise ValueError(f"Invalid number of frames: {T}")
            
        BT, L, C = x.shape
        B = BT // T
        Ca = self.Ca
        if self.use_temporal_conv and self.use_frequency_conv:
            Ct = Ca // 2  # temporal half
            Cf = Ca - Ct  # frequency half
        elif self.use_temporal_conv:
            Ct = Ca
            Cf = 0
        elif self.use_frequency_conv:
            Ct = 0
            Cf = Ca

        H = W = round(math.sqrt(L - 1))
        assert L - 1 == H * W

        x_id = x
        x = x[:, 1:, :]          # drop CLS
        x = self.fc1(x)          # (BT, N, Ca)
        x = x.view(B, T, H, W, Ca)

        # Split channels: temporal half, frequency half
        xt, xf = x[..., :Ct], x[..., Ct:]

        # ------------------
        # Temporal branch
        # ------------------
        if self.use_temporal_conv:
            xt = xt.permute(0, 4, 1, 2, 3)   # (B, Ct, T, H, W)
            xt = self.temporal_conv(xt)
            xt = xt.permute(0, 2, 3, 4, 1)   # (B, T, H, W, Ct)

        # ------------------
        # Frequency branch (multi-res FFT)
        # ------------------


        if self.use_frequency_conv:
            out_f = torch.zeros_like(xf)
            for w in self.windows:
                assert T % w == 0, f"T={T} not divisible by window={w}"
                chunks = T // w
                xf_chunk = xf.view(B, chunks, w, H, W, Cf)  # (B, chunks, w, H, W, Cf)

                X = torch.fft.rfft(xf_chunk, dim=2)         # (B, chunks, F, H, W, Cf)
                X = torch.view_as_real(X).contiguous()
                B_, chunks_, Freq, H_, W_, d, two = X.shape
                X = X.view(B_*chunks_, 2*d, Freq, H_, W_)
                Y = self.freq_conv(X)                       # (B*chunks, 2*Cf, F, H, W)

                Y = Y.view(B, chunks, Freq, H, W, d, 2)
                Y = torch.complex(Y[...,0], Y[...,1])
                y = torch.fft.irfft(Y, n=w, dim=2)          # (B, chunks, w, H, W, Cf)
                y = y.contiguous().view(B, T, H, W, d)

                out_f += y

            out_f = out_f / len(self.windows)

        # ------------------
        # Fuse
        # ------------------
        if self.use_temporal_conv and self.use_frequency_conv:
            if self.fusion_mode == "concat":
                out = torch.cat([xt, out_f], dim=-1)
            elif self.fusion_mode == "linear":
                fused = torch.cat([xt, out_f], dim=-1)
                out = self.fuse_activation(self.fuse_proj(fused))
            elif self.fusion_mode == "gate":
                # gated fusion: both branches same size
                min_c = min(xt.shape[-1], out_f.shape[-1])
                alpha = torch.sigmoid(self.gate)
                fused_half = alpha * xt[..., :min_c] + (1 - alpha) * out_f[..., :min_c]
                out = fused_half
            else:
                raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")
        elif self.use_temporal_conv:
            out = xt
        elif self.use_frequency_conv:
            out = out_f
        else:
            raise ValueError("Either use_temporal_conv or use_frequency_conv must be True")
        if self.fusion_mode == 'gate':
            out = out.reshape(BT, H*W, Ca//2)
        else:
            out = out.reshape(BT, H*W, Ca)

        out = self.fc2(out)

        x_id[:, 1:, :] += out
        return x_id

    
class STFT_dual_adapter(nn.Module):
    def __init__(self, in_channels, adapter_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.fc2 = nn.Linear(adapter_channels, in_channels)

        # Branch A: Conv3D (3,1,1) → temporal refinement only
        self.conv3d_temporal = nn.Conv3d(
            in_channels=2*adapter_channels,
            out_channels=2*adapter_channels,
            kernel_size=(3,1,1),
            padding=(1,0,0),
            groups=2*adapter_channels
        )

        # Branch B: Conv3D (3,3,1) → temporal + cross-frequency refinement
        self.conv3d_cross = nn.Conv3d(
            in_channels=2*adapter_channels,
            out_channels=2*adapter_channels,
            kernel_size=(3,3,1),
            padding=(1,1,0),
            groups=2*adapter_channels
        )

        # init
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)
        nn.init.constant_(self.conv3d_temporal.bias, 0.)
        nn.init.constant_(self.conv3d_cross.bias, 0.)
        nn.init.constant_(self.conv3d_temporal.weight, 0.)
        nn.init.constant_(self.conv3d_cross.weight, 0.)


    @staticmethod
    def safe_stft(x, n_fft, hop_length):
        assert x.ndim == 2, f"STFT expects 2D (batch,time), got {x.shape}"
        assert x.size(1) >= n_fft, f"time dim {x.size(1)} < n_fft {n_fft}"
        x = x.contiguous().to(torch.float32)   # avoid AMP issues
        return torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            return_complex=True
        )

    def forward(self, x, T):
        """
        x: (B*T, L, C) with CLS + patch tokens
        T: number of frames
        """
            
        BT, L, C = x.shape
        B = BT // T
        Ca = self.fc1.out_features

        H = W = round(math.sqrt(L - 1))
        assert L - 1 == H * W

        x_id = x
        x = x[:, 1:, :]          # remove CLS
        x = self.fc1(x)          # (BT, N, Ca)
        x = x.view(B, T, H, W, Ca)  # (B, T, H, W, Ca)

        out = torch.zeros_like(x)

        # STFT implementation - process entire sequence at once
        # Reshape to (B*H*W, T, Ca) for STFT processing
        x_flat = x.permute(0, 2, 3, 1, 4).reshape(B*H*W, T, Ca)  # (B*H*W, T, Ca)
        
        # STFT parameters - ensure multiple frames
        n_fft = min(32, T // 2)  # Use smaller FFT size to allow overlapping
        hop_length = max(1, n_fft // 4)  # 75% overlap
        
        # Ensure we have at least 2 frames
        if (T - n_fft) // hop_length + 1 < 2:
            n_fft = T // 4
            hop_length = max(1, n_fft // 4)
        
        window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
        
        # Apply STFT to all channels at once
        # Reshape to process all channels together
        x_reshaped = x_flat.permute(0, 2, 1).contiguous()  # (B*H*W, Ca, T)
        x_reshaped = x_reshaped.view(-1, T)  # (B*H*W*Ca, T)
        
        X = torch.stft(
            x_reshaped,  # (B*H*W*Ca, T)
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
            center=False
        )  # (B*H*W*Ca, Freq, Frames)
        
        # Reshape back to separate channels
        X = X.view(B*H*W, Ca, X.shape[1], X.shape[2])  # (B*H*W, Ca, Freq, Frames)
        X = X.permute(0, 2, 3, 1).contiguous()  # (B*H*W, Freq, Frames, Ca)
        
        # Reshape for Conv3D
        X = torch.view_as_real(X)  # (B*H*W, Freq, Frames, Ca, 2)
        X = X.permute(0, 3, 1, 2, 4).contiguous()  # (B*H*W, Ca, Freq, Frames, 2)
        X = X.view(B*H*W, 2*Ca, X.shape[2], X.shape[3], 1)  # Add spatial dim
        
        # Two Conv3D branches
        Y_temporal = self.conv3d_temporal(X)
        Y_cross    = self.conv3d_cross(X)

        # Fuse branches (simple sum)
        Y = Y_temporal + Y_cross
        
        # Inverse STFT
        Y = Y.view(B*H*W, 2*Ca, Y.shape[2], Y.shape[3], 1)
        Y = Y.permute(0, 2, 3, 1, 4).contiguous()  # (B*H*W, Freq, Frames, 2*Ca, 1)
        Y = Y[..., 0]  # Remove spatial dim
        Y = Y.view(B*H*W, Y.shape[1], Y.shape[2], Ca, 2)
        Y = torch.complex(Y[..., 0], Y[..., 1])  # (B*H*W, Freq, Frames, Ca)
        
        # Inverse STFT for all channels at once
        Y_reshaped = Y.permute(0, 3, 1, 2).contiguous()  # (B*H*W, Ca, Freq, Frames)
        Y_reshaped = Y_reshaped.view(-1, Y.shape[1], Y.shape[2])  # (B*H*W*Ca, Freq, Frames)
        
        y = torch.istft(
            Y_reshaped,  # (B*H*W*Ca, Freq, Frames)
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            length=T  # Ensure output length matches T
        )  # (B*H*W*Ca, T)
        
        # Reshape back to separate channels
        y = y.view(B*H*W, Ca, T)  # (B*H*W, Ca, T)
        y = y.permute(0, 2, 1).contiguous()  # (B*H*W, T, Ca)
        y = y.view(B, H, W, T, Ca).permute(0, 3, 1, 2, 4)  # (B, T, H, W, Ca)
        
        out = y
            
        # flatten back
        out = out.reshape(BT, H*W, Ca)
        out = self.fc2(out)

        # residual connection
        x_id[:, 1:, :] += out
        return x_id



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 adapter_width: int,
                 adapter_kernel_size: Tuple[int, int, int],
                 windows: Tuple[int, int, int],
                 adapter_pre_attn: bool,
                 adapter_pre_mlp: bool,
                 fusion_mode: str = 'concat',
                 use_temporal_conv: bool = True,
                 use_frequency_conv: bool = True,
                 use_stft: bool = False,
                 ) -> None:
        super().__init__()
        self.fusion_mode = fusion_mode
        self.use_temporal_conv = use_temporal_conv
        self.use_frequency_conv = use_frequency_conv
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

        adapter_class = functools.partial(
            TemporalFreqAdapter,
            in_channels=d_model,
            adapter_channels=adapter_width,
            windows=windows,
            kernel_size=adapter_kernel_size,
            fusion_mode=fusion_mode,
            use_temporal_conv=use_temporal_conv,
            use_frequency_conv=use_frequency_conv,
        )
        if use_stft:
            adapter_class = functools.partial(
                STFT_dual_adapter,
                in_channels=d_model,
                adapter_channels=adapter_width,
            )
        self.adapter_pre_attn = \
            adapter_class() if adapter_pre_attn else None
        self.adapter_pre_mlp = \
            adapter_class() if adapter_pre_mlp else None

    def attention(self, x: torch.Tensor, return_attn_weights: bool = False):
        B, L, C = x.size()
        H = self.attn.num_heads

        qkv = F.linear(x, weight=self.attn.in_proj_weight, bias=self.attn.in_proj_bias)
        qkv = qkv.view(B, L, H * 3, -1).permute(0, 2, 1, 3)
        q, k, v = qkv.split([H, H, H], dim=1)

        head_dim = q.shape[-1]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        attn_probs = torch.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_probs, v)
        out = attn_output.permute(0, 2, 1, 3).flatten(-2)
        out = self.attn.out_proj(out)

        if return_attn_weights:
            return out, attn_probs

        return out
        

    def forward(self,
                x: torch.Tensor,
                num_frames: int,
                return_attn_weights: bool = False
                ):
        if self.adapter_pre_attn is not None:
            x = self.adapter_pre_attn(x, num_frames)
        if return_attn_weights:
            attn_out, attn_weights = self.attention(self.ln_1(x), return_attn_weights=True)
        else:
            attn_out = self.attention(self.ln_1(x))
        x = x + attn_out
        if self.adapter_pre_mlp is not None:
            x = self.adapter_pre_mlp(x, num_frames)
        x = x + self.mlp(self.ln_2(x))
        if return_attn_weights:
            return x, attn_weights
        return x



class Transformer(nn.Module):
    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 adapter_width: int,
                 adapter_layers: int,
                 adapter_start_layer: int,
                 adapter_kernel_size: Tuple[int, int, int],
                 windows: Tuple[int, int, int],
                 adapter_pre_attn: bool,
                 adapter_pre_mlp: bool,
                 fusion_mode: str = 'concat',
                 use_temporal_conv: bool = True,
                 use_frequency_conv: bool = True,
                 use_stft: bool = False,
                 ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.adapter_layers = adapter_layers
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                d_model=width,
                n_head=heads,
                adapter_width=adapter_width,
                adapter_kernel_size=adapter_kernel_size,
                windows=windows,
                adapter_pre_attn=adapter_pre_attn and i >= layers - adapter_layers,
                adapter_pre_mlp=adapter_pre_mlp and i >= layers - adapter_layers,
                fusion_mode=fusion_mode,
                use_temporal_conv=use_temporal_conv,
                use_frequency_conv=use_frequency_conv,
                use_stft=use_stft,
            )
            for i in range(layers)
        ])

            
    def forward(self, x: torch.Tensor, num_frames: int, return_attn_weights: bool = False):
        attn_weights = [] if return_attn_weights else None
        for block in self.resblocks:
            if return_attn_weights:
                x, weights = block(x, num_frames, return_attn_weights=True)
                attn_weights.append(weights)
            else:
                x = block(x, num_frames)
        if return_attn_weights:
            return x, attn_weights
        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 input_resolution: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 num_classes: int,
                 adapter_width: int,
                 adapter_layers: int,
                 adapter_start_layer: int,
                 adapter_kernel_size: Tuple[int, int, int],
                 windows: Tuple[int, int, int],
                 adapter_pre_attn: bool,
                 adapter_pre_mlp: bool,
                 fusion_mode: str = 'concat',
                 use_temporal_conv: bool = True,
                 use_frequency_conv: bool = True,
                 use_stft: bool = False,
                 ):
        super().__init__()
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width,
            kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(
                (input_resolution // patch_size) ** 2 + 1, width
            )
        )
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads,
            adapter_width, adapter_layers, adapter_start_layer, adapter_kernel_size,
            windows, adapter_pre_attn, adapter_pre_mlp, fusion_mode, use_temporal_conv, use_frequency_conv, use_stft)

        self.ln_post = LayerNorm(width)

        for n, p in self.named_parameters():
          if 'adapter' not in n:
            p.requires_grad_(False)
            p.data = p.data.half()
        
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(width, num_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.constant_(self.fc.bias, 0.)


    def forward(self, x: torch.Tensor, return_features=False, attn_weights=False):
        B, T = x.size(0), x.size(2)
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        spatial_size = tuple(x.size()[2:])
        x = x.flatten(-2).permute(0, 2, 1)
        x = torch.cat([
            self.class_embedding.view(1, 1, -1).expand(x.shape[0], -1, -1), x
            ], dim=1)  # [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.view(B, T, x.size(1), x.size(2)).flatten(0, 1) # BT, L, D

        transformer_out = self.transformer(x, T, return_attn_weights=attn_weights)
        if attn_weights:
            x, attn_maps = transformer_out
        else:
            x = transformer_out

        x = x.contiguous().view(B, T, spatial_size[0] * spatial_size[1] + 1, x.size(-1))
        cls_emb = x[:, :, 1:, :].mean(dim=2)
        x = x[:, :, 0, :].mean(dim=1)

        x = self.ln_post(x)
        x = self.dropout(x)
        x = self.fc(x)

        if return_features and attn_weights:
            return x, cls_emb, attn_maps
        if return_features:
            return x, cls_emb
        if attn_weights:
            return x, attn_maps
        return x


def clip_vit_base_patch16_adapter12x384_ms(**kwargs):
    model = VisionTransformer(
        input_resolution=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        adapter_width=384,
        adapter_layers=12,
        adapter_start_layer=0,
        adapter_kernel_size=(3, 1, 1),
        windows=(32, 16, 8),
        adapter_pre_attn=False,
        adapter_pre_mlp=True,
        **kwargs,
    )
    assert CLIP_VIT_B16_PATH is not None, \
        'Please set CLIP_VIT_B16_PATH in configs.py'
    checkpoint = torch.jit.load(CLIP_VIT_B16_PATH, map_location='cpu')
    print(model.load_state_dict(checkpoint.visual.state_dict(), strict=False))
    return model

def clip_vit_base_patch16_adapter12x384_stft(**kwargs):
    model = VisionTransformer(
        input_resolution=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        adapter_width=384,
        adapter_layers=12,
        adapter_start_layer=0,
        adapter_kernel_size=(3, 1, 1),
        windows=(32, 16, 8),
        adapter_pre_attn=False,
        adapter_pre_mlp=True,
        use_stft=True,
        **kwargs,
    )
    assert CLIP_VIT_B16_PATH is not None, \
        'Please set CLIP_VIT_B16_PATH in configs.py'
    checkpoint = torch.jit.load(CLIP_VIT_B16_PATH, map_location='cpu')
    print(model.load_state_dict(checkpoint.visual.state_dict(), strict=False))
    return model

def clip_vit_base_patch16_adapter24x384_ms(**kwargs):
    model = VisionTransformer(
        input_resolution=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        adapter_width=384,
        adapter_layers=12,
        adapter_start_layer=0,
        adapter_kernel_size=(3, 1, 1),
        windows=(32, 16, 8),
        adapter_pre_attn=True,
        adapter_pre_mlp=True,
        **kwargs,
    )
    assert CLIP_VIT_B16_PATH is not None, \
        'Please set CLIP_VIT_B16_PATH in configs.py.'
    checkpoint = torch.jit.load(CLIP_VIT_B16_PATH, map_location='cpu')
    print(model.load_state_dict(checkpoint.visual.state_dict(), strict=False))
    return model

def clip_vit_large_patch14_adapter12x384_ms(**kwargs):
    model = VisionTransformer(
        input_resolution=224,
        patch_size=14,
        width=1024,
        layers=24,
        heads=16,
        adapter_width=384,
        adapter_layers=24,
        adapter_start_layer=0,
        adapter_kernel_size=(3, 1, 1),
        windows=(16, 8, 4),
        adapter_pre_attn=False,
        adapter_pre_mlp=True,
        **kwargs,
    )
    assert CLIP_VIT_L14_PATH is not None, \
        'Please set CLIP_VIT_L14_PATH in configs.py.'
    checkpoint = torch.jit.load(CLIP_VIT_L14_PATH, map_location='cpu')
    print(model.load_state_dict(checkpoint.visual.state_dict(), strict=False))
    return model
