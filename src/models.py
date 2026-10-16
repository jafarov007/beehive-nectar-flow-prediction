"""
Uc mimari (ayni arayuz: girdi (B, T, F) -> cikti (B,)):
  * CausalCNN         : causal + dilated + residual (TCN)
  * LSTMModel         : 2 katmanli LSTM
  * CausalTransformer : causal mask'li Transformer encoder
Parametre butcesi ~49k (iso-parametre). Hepsinin basligi: Linear(h,32)-ReLU-Linear(32,1).
"""
import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

N_FEATURES = 9

# Pencereye gore CNN konfigurasyonu (C kanal, dilation listesi) -- parametre ~49-50k kalacak sekilde
CNN_CFG = {
    24: (51, [1, 2, 4]),                 # RF = 1+2*2*7  = 29  >= 24
    72: (40, [1, 2, 4, 8, 16]),          # RF = 1+2*2*31 = 125 >= 72
    168: (36, [1, 2, 4, 8, 16, 32]),     # RF = 1+2*2*63 = 253 >= 168
}
CNN_CFG_12H = (51, [1, 2, 4])            # 12 saatlik adimlarda: gunluk=2 adim, haftalik=14 adim; RF=29 >= 14
LSTM_H = 60
TR_D, TR_FF, TR_L, TR_HEADS = 44, 88, 3, 4


def receptive_field(k, dilations, convs_per_block=2):
    return 1 + convs_per_block * (k - 1) * sum(dilations)


class CausalBlock(nn.Module):
    def __init__(self, c, k, d, dropout):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv1 = nn.Conv1d(c, c, k, dilation=d)
        self.conv2 = nn.Conv1d(c, c, k, dilation=d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.drop(F.relu(self.conv1(F.pad(x, (self.pad, 0)))))   # sola padding => causal
        h = self.drop(F.relu(self.conv2(F.pad(h, (self.pad, 0)))))
        return F.relu(h + x)                                         # residual


def _head(h):
    return nn.Sequential(nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, 1))


class CausalCNN(nn.Module):
    def __init__(self, in_features=N_FEATURES, channels=36, dilations=(1, 2, 4, 8, 16, 32), kernel=3, dropout=0.1):
        super().__init__()
        self.kernel, self.dilations = kernel, list(dilations)
        self.inp = nn.Conv1d(in_features, channels, 1)
        self.blocks = nn.Sequential(*[CausalBlock(channels, kernel, d, dropout) for d in dilations])
        self.head = _head(channels)

    def forward(self, x):
        h = self.blocks(self.inp(x.transpose(1, 2)))
        return self.head(h[:, :, -1]).squeeze(-1)

    @property
    def rf(self):
        return receptive_field(self.kernel, self.dilations)


class LSTMModel(nn.Module):
    def __init__(self, in_features=N_FEATURES, hidden=LSTM_H, layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = _head(hidden)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class CausalTransformer(nn.Module):
    def __init__(self, in_features=N_FEATURES, d=TR_D, heads=TR_HEADS, layers=TR_L, ff=TR_FF, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(in_features, d)
        self.pos = PositionalEncoding(d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = _head(d)

    def forward(self, x):
        T = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)   # gelecegi gormez
        h = self.enc(self.pos(self.inp(x)), mask=mask, is_causal=True)
        return self.head(h[:, -1]).squeeze(-1)


ARCHS = ["cnn", "lstm", "transformer"]


def make_model(arch, W=168, dropout=0.1, in_features=N_FEATURES, grid="1h"):
    if arch == "cnn":
        c, dil = CNN_CFG_12H if grid == "12h" else CNN_CFG[W]
        return CausalCNN(in_features, c, dil, dropout=dropout)
    if arch == "lstm":
        return LSTMModel(in_features, dropout=dropout)
    if arch == "transformer":
        return CausalTransformer(in_features, dropout=dropout)
    raise ValueError(arch)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ------------------------------------------------------------------ MAC sayaci (hook tabanli)
def count_macs(model, T, in_features=N_FEATURES):
    """Bir dizi (batch=1, uzunluk T) icin carpma-toplama sayisi. Bias/aktivasyon/LayerNorm sayilmaz.
    Conv1d, Linear, LSTM ve MultiheadAttention modullerinin gercek tensor sekillerinden hesaplanir."""
    total = {"v": 0}
    hooks = []

    def conv_hook(m, i, o):
        total["v"] += m.in_channels * m.out_channels * m.kernel_size[0] * o.shape[-1] // m.groups

    def lin_hook(m, i, o):
        total["v"] += int(o.numel() // o.shape[-1]) * m.in_features * m.out_features

    def lstm_hook(m, i, o):
        Tn = i[0].shape[1]
        for l in range(m.num_layers):
            I = m.input_size if l == 0 else m.hidden_size
            total["v"] += 4 * m.hidden_size * (I + m.hidden_size) * Tn

    def mha_hook(m, i, o):
        q = i[0]
        Tn, d = q.shape[1], q.shape[2]
        total["v"] += 3 * Tn * d * d + Tn * Tn * d + Tn * Tn * d + Tn * d * d   # QKV, QK^T, AV, W_O

    for mod in model.modules():
        if isinstance(mod, nn.Conv1d):
            hooks.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.LSTM):
            hooks.append(mod.register_forward_hook(lstm_hook))
        elif isinstance(mod, nn.MultiheadAttention):
            hooks.append(mod.register_forward_hook(mha_hook))
        elif isinstance(mod, nn.Linear):
            if "out_proj" in getattr(mod, "_name", ""):
                continue
            hooks.append(mod.register_forward_hook(lin_hook))
    model.train()   # eval+no_grad'da Transformer "fast path" alt-modul hook'larini atlar -> eksik sayim olur
    with torch.no_grad():
        model(torch.zeros(1, T, in_features))
    for h in hooks:
        h.remove()
    return total["v"]


def measured_receptive_field(model, T=400, trials=8):
    """Son cikti adiminin girdideki en eski bagimli adimi (gradyan != 0)."""
    model.eval()
    best = 0
    for s in range(trials):
        torch.manual_seed(s)
        x = torch.randn(1, T, N_FEATURES, requires_grad=True)
        model(x).sum().backward()
        g = x.grad.abs().sum(dim=(0, 2))
        nz = torch.nonzero(g > 0).flatten()
        best = max(best, T - int(nz.min()))
    return best


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for arch in ARCHS:
        m = make_model(arch, 168)
        print(f"{arch:12s} params={count_params(m):6d}  MAC(T=168)={count_macs(m, 168):,}")
    m = make_model("cnn", 168)
    print("CNN RF teorik =", m.rf, " olculen =", measured_receptive_field(m))
