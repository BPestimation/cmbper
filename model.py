import torch as th
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
import math
import time


class BLSTM(nn.Module):
    def __init__(self, dim, layers=2, bi=True):
        super().__init__()
        self.lstm = nn.LSTM(bidirectional=bi, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = None
        if bi:
            self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x, hidden)
        if self.linear:
            x = self.linear(x)
        return x, hidden


def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)


class Model(nn.Module):
    def __init__(self,
                 chin=1,
                 chout=4,
                 hidden=48,
                 depth=4,
                 kernel_size=5,
                 stride=2,
                 causal=False,
                 growth=2,
                 max_hidden=10_000,
                 normalize=False,
                 glu=True,
                 rescale=0.1,
                 floor=1e-3,
                 eps=1e-8, 
                 calibration=False,
                 use_temperature_alpha=False):
        super().__init__()

        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.causal = causal
        self.floor = floor
        self.normalize = normalize
        self.eps = eps
        self.calibration = calibration
        self.use_temperature_alpha = use_temperature_alpha
        self.tmps = nn.Parameter(th.zeros(3))
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        activation = nn.GLU(1) if glu else nn.ReLU()
        ch_scale = 2 if glu else 1

        for index in range(depth):
            encode = []
            encode += [
                weight_norm(nn.Conv1d(chin, hidden, kernel_size, stride), name='weight'),
                nn.ReLU(),
                weight_norm(nn.Conv1d(hidden, hidden * ch_scale, 1), name='weight'),
                activation,
                nn.BatchNorm1d(hidden),
            ]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            decode += [
                nn.BatchNorm1d(hidden),
                weight_norm(nn.Conv1d(hidden, ch_scale * hidden, 1), name='weight'),
                activation,
                weight_norm(nn.ConvTranspose1d(hidden, chout, kernel_size, stride), name='weight'),
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            chout = hidden
            chin = hidden
            hidden = min(int(growth * hidden), max_hidden)
        self.lstm = BLSTM(chin, bi=not causal)
        if rescale:
            rescale_module(self, reference=rescale)

    def valid_length(self, length):
        length = math.ceil(length)
        for _ in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for _ in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length))
        return int(length)

    @property
    def total_stride(self):
        return self.stride ** self.depth

    def compute_epistemic_uncertainty(self, alpha, beta, v):
        epis = (beta / (v * (alpha - 1))) ** 0.5
        return epis

    def compute_aleatoric_uncertainty(self, alpha, beta):
        alea = (beta / ((alpha - 1))) ** 0.5
        return alea

    def compute_prediction_and_uncertainty(self, mix, aleatoric=False):
        out = self.forward(mix)
        gamma, v, alpha, beta = out.chunk(4,1)
        pred = gamma
        epis = self.compute_epistemic_uncertainty(alpha, beta, v)
        if aleatoric:
            alea = self.compute_aleatoric_uncertainty(alpha, beta)
            return pred, epis, alea
        else:
            return pred, epis

    def forward(self, mix):
        if mix.dim() == 2:
            mix = mix.unsqueeze(1)

        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            mix = mix / (self.floor + std)
        else:
            std = 1
        length = mix.shape[-1]
        x = mix
        pad_l = (self.valid_length(length) - length) // 2
        pad_r = (self.valid_length(length) - length) - pad_l

        x = F.pad(x, (pad_l, pad_r))
        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = x.permute(1, 2, 0)
        for decode in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[..., :x.shape[-1]]
            x = decode(x)

        x = x[..., :length]
        x = std * x

        a, b, c, d = x.chunk(4, 1)
        gamma = a
        v = F.softplus(b)
        alpha = F.softplus(c) 
        beta = F.softplus(d)
        if self.calibration:
            v = F.softplus(self.tmps[0]) * v
            beta = F.softplus(self.tmps[1]) * beta
            if self.use_temperature_alpha:
                alpha = F.softplus(self.tmps[2]) * alpha
        v = v + self.eps
        alpha_plus_one = alpha + 1 + self.eps
        beta = beta + self.eps
        x = th.cat((gamma, v, alpha_plus_one, beta), dim=1)

        return x