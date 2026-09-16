"""The learned gain model: single GRU layer, band-gain output (float32).

Architecture (documented in docs/research/learned-model.md):
    input   : 2*N_BANDS features per frame (log10 band RMS: mixture + noise)
    layer   : one GRU, hidden units = HIDDEN (32)
    output  : N_BANDS gains in [0,1] via sigmoid on a linear readout
    apply   : gains are expanded to the 65 bins (banding.py) and multiply the
              mixture spectrum inside the SAME WOLA chain as the classical
              algorithm (spectral_subtraction.SpectralSubtraction gain_hook).

PyTorch GRU gate math (nn.GRU) and WHY the export keeps biases separate:

    r_t = sigmoid(W_ir x_t + b_ir + W_hr h + b_hr)     # both b_ir, b_hr added
    z_t = sigmoid(W_iz x_t + b_iz + W_hz h + b_hz)     # both b_iz, b_hz added
    n_t = tanh( W_in x_t + b_in + r_t * (W_hn h + b_hn))  # b_hn scaled by r
    h_t = (1 - z_t) * n_t + z_t * h

bias_ih and bias_hh therefore CANNOT be summed into one vector: for the n
gate PyTorch applies b_in unscaled and b_hn scaled by r_t. Summing them
first inserts a spurious r*b_in term (observed as a ~1e-3 systematic gap).
The numpy forward in infer.py replicates the equations EXACTLY; a test
asserts torch==numpy agreement to 3e-3 (float32 activation rounding over
a short window - the machinery is identical, the tolerance is precision,
not a second algorithm). Training is float32 throughout; quantized int8 is
a later, separate step (the classical DSP->firmware sequencing discipline).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .constants import HIDDEN, N_BANDS


class GruGainNet(nn.Module):
    """Float32 GRU gain estimator (training object)."""

    def __init__(self, n_features: int, hidden: int = HIDDEN,
                 n_bands: int = N_BANDS) -> None:
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_bands)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None):
        """x: (T, F) or (B, T, F); returns (gains, last_hidden_state)."""
        out, h = self.gru(x, h)
        return torch.sigmoid(self.out(out)), h


@dataclass
class LearnedWeights:
    """Float32 weights in the export schema (what infer.py consumes).

    gate order of the stacked (3H, *) rows is (reset r, update z, new n),
    matching PyTorch's weight_ih_l0/weight_hh_l0 layout. b_ih and b_hh are
    kept SEPARATE because of the n-gate's reset-scaled hidden bias (see the
    module docstring) - a combined bias vector is not semantically equal.
    """

    w_ir: np.ndarray   # (3H, F)  gate input weights
    w_hr: np.ndarray   # (3H, H)  gate hidden weights
    b_ih: np.ndarray   # (3H,)   input biases (b_ir, b_iz, b_in)
    b_hh: np.ndarray   # (3H,)   hidden biases (b_hr, b_hz, b_hn)
    out_w: np.ndarray  # (B, H)
    out_b: np.ndarray  # (B,)
    n_features: int
    hidden: int
    n_bands: int

    @property
    def total_params(self) -> int:
        return (self.w_ir.size + self.w_hr.size + self.b_ih.size
                + self.b_hh.size + self.out_w.size + self.out_b.size)


def export_weights(net: GruGainNet) -> LearnedWeights:
    """Copy torch state into the flat float32 export schema (biases separate)."""
    gru = net.gru
    w_ir = gru.weight_ih_l0.detach().cpu().numpy().astype(np.float32)
    w_hr = gru.weight_hh_l0.detach().cpu().numpy().astype(np.float32)
    b_ih = gru.bias_ih_l0.detach().cpu().numpy().astype(np.float32)
    b_hh = gru.bias_hh_l0.detach().cpu().numpy().astype(np.float32)
    ow = net.out.weight.detach().cpu().numpy().astype(np.float32)
    ob = net.out.bias.detach().cpu().numpy().astype(np.float32)
    return LearnedWeights(
        w_ir=w_ir, w_hr=w_hr, b_ih=b_ih, b_hh=b_hh, out_w=ow, out_b=ob,
        n_features=gru.input_size, hidden=gru.hidden_size,
        n_bands=ow.shape[0],
    )


def gru_forward_numpy(w: LearnedWeights, x: np.ndarray,
                      h: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Numpy GRU forward, exactly PyTorch's nn.GRU equations (1 layer).

    x: (T, F) -> (T, B) sigmoid gains + final h (H,). Gate order (r, z, n)
    matches torch's stacked weights; bias semantics match the module docstring
    (b_ih added unscaled everywhere, b_hh scaled by r in the n gate only).
    """
    H = w.hidden
    x = np.asarray(x, dtype=np.float64)
    if h is None:
        h = np.zeros(H, dtype=np.float64)
    else:
        h = np.asarray(h, dtype=np.float64).reshape(H)
    Wr, Wz, Wn = w.w_ir[:H], w.w_ir[H : 2 * H], w.w_ir[2 * H :]
    Ur, Uz, Un = w.w_hr[:H], w.w_hr[H : 2 * H], w.w_hr[2 * H :]
    bir, biz, bin_ = w.b_ih[:H], w.b_ih[H : 2 * H], w.b_ih[2 * H :]
    bhr, bhz, bhn = w.b_hh[:H], w.b_hh[H : 2 * H], w.b_hh[2 * H :]
    gains = np.empty((x.shape[0], w.n_bands), dtype=np.float64)
    for t in range(x.shape[0]):
        xt = x[t]
        r = 1.0 / (1.0 + np.exp(-(Wr @ xt + Ur @ h + bir + bhr)))
        z = 1.0 / (1.0 + np.exp(-(Wz @ xt + Uz @ h + biz + bhz)))
        n_ = np.tanh(Wn @ xt + bin_ + r * (Un @ h + bhn))
        h = (1.0 - z) * n_ + z * h
        gains[t] = 1.0 / (1.0 + np.exp(-(w.out_w @ h + w.out_b)))
    return gains, h