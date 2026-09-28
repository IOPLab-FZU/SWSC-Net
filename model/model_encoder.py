import torch
from torch import nn
from torch.nn import functional as F
from mamba_ssm import Mamba


class Encoder(nn.Module):

    def __init__(self, network, wavelet_stage=4):
        super(Encoder, self).__init__()
        self.network = network
        if 'segformer' not in self.network:
            raise ValueError('SWSC Encoder only supports segformer backbones, e.g. segformer-mit_b1.')
        if wavelet_stage not in {0, 1, 2, 3, 4}:
            raise ValueError('wavelet_stage must be one of {0, 1, 2, 3, 4}; 0 means raw image.')

        self.wavelet_stage = wavelet_stage
        from .segformer import Segformer_baseline
        self.cnn = Segformer_baseline(backbone=self.network.split('-')[-1])
        self.fine_tune()

    def forward(self, imageA, imageB):
        """
        Args:
            imageA, imageB: [B, 3, H, W]

        Returns:
            feat1, feat2: SegFormer stage-4 high-level features.
            shallow1, shallow2: raw image or selected stage-1/2/3/4 features for WSRM.
        """
        shallow_feats1, feat1_stage3 = self.cnn.segformer.stage_123(imageA)
        shallow_feats2, feat2_stage3 = self.cnn.segformer.stage_123(imageB)
        feat1 = self.cnn.segformer.stage_4(feat1_stage3)
        feat2 = self.cnn.segformer.stage_4(feat2_stage3)
        if self.wavelet_stage == 0:
            shallow1 = imageA
            shallow2 = imageB
        elif self.wavelet_stage in {1, 2, 3}:
            shallow1 = shallow_feats1[self.wavelet_stage - 1]
            shallow2 = shallow_feats2[self.wavelet_stage - 1]
        else:
            shallow1 = feat1
            shallow2 = feat2
        return feat1, feat2, shallow1, shallow2

    def fine_tune(self, fine_tune=True):
        for p in self.cnn.parameters():
            p.requires_grad = fine_tune


def _dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_ll = x1 + x2 + x3 + x4
    x_hl = -x1 - x2 + x3 + x4
    x_lh = -x1 + x2 - x3 + x4
    x_hh = x1 - x2 - x3 + x4
    return torch.cat((x_ll, x_hl, x_lh, x_hh), dim=0)


def _iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch = int(in_batch / (r ** 2))
    out_height, out_width = r * in_height, r * in_width
    x1 = x[0:out_batch, :, :, :] / 2
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2
    h = torch.zeros([out_batch, in_channel, out_height, out_width], dtype=x.dtype, device=x.device)
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return h


class HaarDWT(nn.Module):
    def forward(self, x):
        return _dwt_init(x)


class HaarIWT(nn.Module):
    def forward(self, x):
        return _iwt_init(x)


class HighFreqMamba(nn.Module):
    """Apply SSM/Mamba to high-frequency wavelet bands."""

    def __init__(self, dim):
        super(HighFreqMamba, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(dim)
        self.ssm = Mamba(d_model=dim, d_state=32, d_conv=4, expand=2)
        self.post = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x
        x = self.pre(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = self.ssm(x)
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return self.post(x) + residual


class WSRM(nn.Module):
    """
    Wavelet Structural Refinement Module.

    DWT band order follows this file's Haar implementation:
        LL, HL, LH, HH

    wavelet_bands can be any underscore combination of ll/lh/hl/hh, or all.
    The selected bands are refined by Mamba; unselected bands bypass refinement.
    """

    BAND_INDEX = {
        'll': 0,
        'hl': 1,
        'lh': 2,
        'hh': 3,
    }

    def __init__(self, in_dim, hidden_dim, wavelet_bands='lh_hl_hh'):
        super(WSRM, self).__init__()
        self.wavelet_bands = wavelet_bands
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        self.dwt = HaarDWT()
        self.iwt = HaarIWT()
        self.high_mamba = HighFreqMamba(hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
        )
        self._selected_band_indices()

    def _selected_band_indices(self):
        if self.wavelet_bands == 'all':
            return [0, 1, 2, 3]
        names = [name for name in self.wavelet_bands.split('_') if name]
        if not names:
            raise ValueError('wavelet_bands must select at least one band.')
        selected = []
        for name in names:
            if name not in self.BAND_INDEX:
                raise ValueError('wavelet_bands must be a non-empty underscore combination of ll/lh/hl/hh, or all.')
            idx = self.BAND_INDEX[name]
            if idx not in selected:
                selected.append(idx)
        return selected

    def forward(self, x):
        x = self.input_proj(x)
        _, _, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

        b = x.size(0)
        x_dwt = self.dwt(x)
        bands = [
            x_dwt[:b],
            x_dwt[b:b * 2],
            x_dwt[b * 2:b * 3],
            x_dwt[b * 3:b * 4],
        ]
        selected = self._selected_band_indices()
        selected_feat = torch.cat([bands[idx] for idx in selected], dim=0)
        selected_feat = self.high_mamba(selected_feat)
        selected_chunks = selected_feat.chunk(len(selected), dim=0)
        for idx, feat in zip(selected, selected_chunks):
            bands[idx] = feat

        x = self.iwt(torch.cat(bands, dim=0))
        x = x[:, :, :h, :w]
        return self.output_proj(x)


class SRB(nn.Module):
    """
    Semantic Refinement Block.
    """
    def __init__(self, in_dim=2048, hidden_dim=512):
        super(SRB, self).__init__()
        self.input_proj = nn.Sequential(
            nn.Identity() if in_dim == hidden_dim else nn.Conv2d(in_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        self.context3 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        self.context7 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=3, dilation=3, groups=hidden_dim, bias=False),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.input_proj(x)
        tmp = x + self.context3(x)
        return tmp + self.context7(tmp)


class DCR(nn.Module):
    """
    Difference-Consistency Relation Encoding.
    """
    def __init__(self, hidden_dim=512):
        super(DCR, self).__init__()

        in_dim = hidden_dim * 4 + 1

        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim)
        )

    def forward(self, f1, f2):
        diff = torch.abs(f2 - f1)
        prod = f1 * f2
        sim = F.cosine_similarity(f1, f2, dim=1, eps=1e-6).unsqueeze(1)

        relation = torch.cat([
            f1,
            f2,
            diff,
            1.0 - sim,
            prod
        ], dim=1)

        relation = self.proj(relation)

        return relation


class TransformerBlock(nn.Module):
    """
    Transformer block for relation-guided visual refinement.
    """
    def __init__(self, hidden_dim=512, num_heads=8, ffn_dim=2048, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.dropout1(y)
        return x + self.ffn(self.norm2(x))


class DCIB(nn.Module):
    """
    Difference-Consistency Interaction Block.
    """
    def __init__(self, hidden_dim=512, num_heads=8, ffn_dim=2048, dropout=0.1):
        super(DCIB, self).__init__()
        self.dcr = DCR(hidden_dim)
        self.block = TransformerBlock(hidden_dim, num_heads, ffn_dim, dropout)

    def _forward_block(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.block(tokens)
        return tokens.transpose(1, 2).reshape(b, c, h, w)

    def forward(self, f1, f2):
        relation = self.dcr(f1, f2)
        f1 = self._forward_block(f1 + relation)
        f2 = self._forward_block(f2 + relation)
        return f1, f2


class SWSCEncoder(nn.Module):
    """
    SWSC visual encoder with SRB, DCIB, and WSRM.
    """
    def __init__(self, in_dim=2048, hidden_dim=512, num_heads=8, n_layers=2,
                 ffn_dim=2048, dropout=0.1, feat_size=16, wavelet_in_dim=320,
                 wavelet_bands='lh_hl_hh'):
        super(SWSCEncoder, self).__init__()
        self.h_embedding = nn.Embedding(feat_size, int(in_dim / 2))
        self.w_embedding = nn.Embedding(feat_size, int(in_dim / 2))
        self.srb = SRB(in_dim, hidden_dim)
        self.dcib = nn.ModuleList([
            DCIB(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.wsrm = WSRM(wavelet_in_dim, hidden_dim, wavelet_bands=wavelet_bands)
        self.wavelet_gate = nn.Parameter(torch.tensor(-2.2))
        nn.init.xavier_uniform_(self.h_embedding.weight)
        nn.init.xavier_uniform_(self.w_embedding.weight)

    def forward(self, feat1, feat2, shallow1=None, shallow2=None):
        batch, c, h, w = feat1.shape
        pos_h = torch.arange(h, device=feat1.device)
        pos_w = torch.arange(w, device=feat1.device)
        embed_h = self.w_embedding(pos_h)
        embed_w = self.h_embedding(pos_w)
        pos_embedding = torch.cat([
            embed_w.unsqueeze(0).repeat(h, 1, 1),
            embed_h.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1)
        pos_embedding = pos_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1, 1)
        feat1 = feat1 + pos_embedding
        feat2 = feat2 + pos_embedding
        v1 = self.srb(feat1)
        v2 = self.srb(feat2)
        for dcib in self.dcib:
            v1, v2 = dcib(v1, v2)

        if shallow1 is not None and shallow2 is not None:
            w1 = self.wsrm(shallow1)
            w2 = self.wsrm(shallow2)
            if w1.shape[-2:] != v1.shape[-2:]:
                w1 = F.interpolate(w1, size=v1.shape[-2:], mode='bilinear', align_corners=False)
                w2 = F.interpolate(w2, size=v2.shape[-2:], mode='bilinear', align_corners=False)
            gate = torch.sigmoid(self.wavelet_gate)
            v1 = v1 + gate * w1
            v2 = v2 + gate * w2

        return v1, v2
