"""PyTorch implementation of BP waveform reconstruction network.

Architecture inferred from actual trained weights in bp_weights.mat
(Stage1_Waveform_Net_Safe_V7_ALLMAXMIN.mat).

Key differences from build_BP_Stage1_Model1.m reference:
- RAMS attention (spatial conv7 + Avg/Max/Range channel MLP)
- ASPP bridge module (3 dilated conv branches + fusion)
- No PositionEncodingLayer
- ResPath skip connections from RAMS outputs

Input:  (B, 1, 256)  normalized pulse wave  [0, 1]
Output: (B, 1, 256)  reconstructed BP waveform [0, 1]
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat


# ===========================================================================
# Building blocks
# ===========================================================================

class MultiResBlock(nn.Module):
    """Multi-resolution convolution block matching MATLAB addMultiResBlock.

    Three parallel branches (3x1, 5x1, 7x1 conv + BN + ReLU), concatenated
    along channel dim, plus 1x1 residual shortcut.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        b1_ch = out_ch // 3
        b2_ch = out_ch // 3
        b3_ch = out_ch - b1_ch - b2_ch

        self.c1 = nn.Conv1d(in_ch, b1_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(b1_ch)
        self.c2 = nn.Conv1d(in_ch, b2_ch, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(b2_ch)
        self.c3 = nn.Conv1d(in_ch, b3_ch, 7, padding=3)
        self.bn3 = nn.BatchNorm1d(b3_ch)
        self.sc = nn.Conv1d(in_ch, out_ch, 1)
        self.scbn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b1 = self.relu(self.bn1(self.c1(x)))
        b2 = self.relu(self.bn2(self.c2(x)))
        b3 = self.relu(self.bn3(self.c3(x)))
        return torch.cat([b1, b2, b3], dim=1) + self.scbn(self.sc(x))


class RAMSAttention(nn.Module):
    """RAMS: Range-Aware Multi-pooling Spatial-Channel attention.

    Spatial:  channel-mean -> Conv(1->2,7) -> softmax -> mean -> (B,1,L) gate
    Channel:  AvgPool, MaxPool, Range each through
              FC(in->in/16->in) -> summed -> sigmoid -> (B,C,1) gate
    Output = x * channel_gate * spatial_gate
    """

    def __init__(self, channels: int):
        super().__init__()
        r = max(2, channels // 16)

        # Spatial conv operates on channel-pooled features: Conv1d(1, 2, 7)
        # MATLAB weight stored as [k=7, out=2] -> 2D (7, 2)
        self.sp_conv = nn.Conv1d(1, 2, 7, padding=3)

        self.mlp_avg_1 = nn.Conv1d(channels, r, 1)
        self.mlp_avg_2 = nn.Conv1d(r, channels, 1)
        self.mlp_max_1 = nn.Conv1d(channels, r, 1)
        self.mlp_max_2 = nn.Conv1d(r, channels, 1)
        self.mlp_rng_1 = nn.Conv1d(channels, r, 1)
        self.mlp_rng_2 = nn.Conv1d(r, channels, 1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Spatial gate: pool channels -> conv -> softmax -> average 2 outputs
        x_pooled = x.mean(dim=1, keepdim=True)            # (B, 1, L)
        sp = torch.softmax(self.sp_conv(x_pooled), dim=-1)  # (B, 2, L)
        sp = sp.mean(dim=1, keepdim=True)                 # (B, 1, L)

        # Channel gate: three pooling strategies
        gap = x.mean(dim=-1, keepdim=True)
        gmp = x.max(dim=-1, keepdim=True).values
        rng = (x.max(dim=-1).values - x.min(dim=-1).values).unsqueeze(-1)

        ch = (self.mlp_avg_2(self.relu(self.mlp_avg_1(gap))) +
              self.mlp_max_2(self.relu(self.mlp_max_1(gmp))) +
              self.mlp_rng_2(self.relu(self.mlp_rng_1(rng))))
        ch = self.sigmoid(ch)
        return x * ch * sp


class ASPPBridge(nn.Module):
    """Atrous Spatial Pyramid Pooling bridge.

    3 parallel branches (1x1, 3x3/d=2, 3x3/d=4) each 1024->256,
    concatenated to 768, fused 768->1024 via 1x1.
    """

    def __init__(self, in_ch: int = 1024, mid_ch: int = 256):
        super().__init__()
        self.b1_conv = nn.Conv1d(in_ch, mid_ch, 1)
        self.b1_bn = nn.BatchNorm1d(mid_ch)
        self.b2_conv = nn.Conv1d(in_ch, mid_ch, 3, padding=2, dilation=2)
        self.b2_bn = nn.BatchNorm1d(mid_ch)
        self.b3_conv = nn.Conv1d(in_ch, mid_ch, 3, padding=4, dilation=4)
        self.b3_bn = nn.BatchNorm1d(mid_ch)
        self.fusion_conv = nn.Conv1d(mid_ch * 3, in_ch, 1)
        self.fusion_bn = nn.BatchNorm1d(in_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b1 = self.relu(self.b1_bn(self.b1_conv(x)))
        b2 = self.relu(self.b2_bn(self.b2_conv(x)))
        b3 = self.relu(self.b3_bn(self.b3_conv(x)))
        return self.relu(self.fusion_bn(
            self.fusion_conv(torch.cat([b1, b2, b3], dim=1))))


class ResPath(nn.Module):
    """Residual path for UNet skip connections (MATLAB addResPath).

    N serial (1x1->BN->ReLU->3x3->BN->ReLU) blocks + 1x1 residual shortcut.
    """

    def __init__(self, in_ch: int, out_ch: int, n_blocks: int = 1):
        super().__init__()
        self.n_blocks = n_blocks
        self.sc = nn.Conv1d(in_ch, out_ch, 1)
        for i in range(1, n_blocks + 1):
            setattr(self, f"conv1_{i}", nn.Conv1d(
                in_ch if i == 1 else out_ch, out_ch, 1))
            setattr(self, f"bn1_{i}", nn.BatchNorm1d(out_ch))
            setattr(self, f"conv3_{i}", nn.Conv1d(out_ch, out_ch, 3, padding=1))
            setattr(self, f"bn2_{i}", nn.BatchNorm1d(out_ch))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = x
        for i in range(1, self.n_blocks + 1):
            out = self.relu(getattr(self, f"bn1_{i}")(
                getattr(self, f"conv1_{i}")(out)))
            out = self.relu(getattr(self, f"bn2_{i}")(
                getattr(self, f"conv3_{i}")(out)))
        return out + self.sc(x)


# ===========================================================================
# Full network
# ===========================================================================

class BPWaveformNet(nn.Module):
    """MultiResUNet + RAMS attention + ASPP bridge.

    Matches Stage1_Waveform_Net_Safe_V7_ALLMAXMIN.mat actual trained weights.
    """

    def __init__(self):
        super().__init__()

        self.input_proj = nn.Conv1d(1, 32, 1)

        # --- Encoder (4 levels) ---
        self.enc1 = MultiResBlock(32, 64)
        self.rams1 = RAMSAttention(64)
        self.pool1 = nn.MaxPool1d(2, 2)

        self.enc2 = MultiResBlock(64, 128)
        self.rams2 = RAMSAttention(128)
        self.pool2 = nn.MaxPool1d(2, 2)

        self.enc3 = MultiResBlock(128, 256)
        self.rams3 = RAMSAttention(256)
        self.pool3 = nn.MaxPool1d(2, 2)

        self.enc4 = MultiResBlock(256, 512)
        self.rams4 = RAMSAttention(512)
        self.pool4 = nn.MaxPool1d(2, 2)

        # --- Bridge ---
        self.bridge_base = MultiResBlock(512, 1024)
        self.rams_bridge = RAMSAttention(1024)
        self.bridge_aspp = ASPPBridge(1024, 256)

        # --- Decoder ---
        self.up4 = nn.ConvTranspose1d(1024, 512, 2, 2)
        self.res4 = ResPath(512, 256, 1)
        self.dec4 = MultiResBlock(512 + 256, 512)
        self.rams_dec4 = RAMSAttention(512)

        self.up3 = nn.ConvTranspose1d(512, 256, 2, 2)
        self.res3 = ResPath(256, 128, 2)
        self.dec3 = MultiResBlock(256 + 128, 256)
        self.rams_dec3 = RAMSAttention(256)

        self.up2 = nn.ConvTranspose1d(256, 128, 2, 2)
        self.res2 = ResPath(128, 64, 3)
        self.dec2 = MultiResBlock(128 + 64, 128)
        self.rams_dec2 = RAMSAttention(128)

        self.up1 = nn.ConvTranspose1d(128, 64, 2, 2)
        self.res1 = ResPath(64, 32, 4)
        self.dec1 = MultiResBlock(64 + 32, 64)
        self.rams_dec1 = RAMSAttention(64)

        # --- Output ---
        self.final_conv = nn.Conv1d(64, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_proj(x)

        e1 = self.rams1(self.enc1(x))
        e2 = self.rams2(self.enc2(self.pool1(e1)))
        e3 = self.rams3(self.enc3(self.pool2(e2)))
        e4 = self.rams4(self.enc4(self.pool3(e3)))

        b = self.bridge_aspp(self.rams_bridge(
            self.bridge_base(self.pool4(e4))))

        d4 = self.rams_dec4(self.dec4(
            torch.cat([self.up4(b), self.res4(e4)], 1)))
        d3 = self.rams_dec3(self.dec3(
            torch.cat([self.up3(d4), self.res3(e3)], 1)))
        d2 = self.rams_dec2(self.dec2(
            torch.cat([self.up2(d3), self.res2(e2)], 1)))
        d1 = self.rams_dec1(self.dec1(
            torch.cat([self.up1(d2), self.res1(e1)], 1)))

        return self.sigmoid(self.final_conv(d1))


# ===========================================================================
# Weight loading helpers
# ===========================================================================

def _t_conv(w: np.ndarray) -> np.ndarray:
    """MATLAB conv [k,in,out] -> PyTorch [out,in,k].
    k=1 weights stored as [in,out] -> [out,in,1].
    """
    if w.ndim == 2:
        return w.T[:, :, np.newaxis].copy()
    return np.transpose(w, (2, 1, 0)).copy()


def _t_tconv(w: np.ndarray) -> np.ndarray:
    """MATLAB tconv [k,out,in] -> PyTorch [in,out,k]."""
    return np.transpose(w, (2, 1, 0)).copy()


def _t_spconv(w: np.ndarray) -> np.ndarray:
    """MATLAB sp_conv weight [k=7,out=2] (in=1 implicit) -> PyTorch [2,1,7]."""
    # (7, 2) -> (7, 1, 2) -> (2, 1, 7)
    return w.reshape(7, 1, 2).transpose(2, 1, 0).copy()


def _s(v: np.ndarray) -> np.ndarray:
    """Squeeze to 1D."""
    return np.squeeze(v).copy()


def _conv(all_p, sd, pt_key, ml_key):
    """Load conv weight+bias."""
    sd[f"{pt_key}.weight"] = torch.from_numpy(_t_conv(all_p[f"{ml_key}__Weights"]))
    sd[f"{pt_key}.bias"] = torch.from_numpy(_s(all_p[f"{ml_key}__Bias"]))


def _tconv(all_p, sd, pt_key, ml_key):
    """Load transposed conv."""
    sd[f"{pt_key}.weight"] = torch.from_numpy(_t_tconv(all_p[f"{ml_key}__Weights"]))
    sd[f"{pt_key}.bias"] = torch.from_numpy(_s(all_p[f"{ml_key}__Bias"]))


def _bn(all_p, sd, pt_key, ml_key):
    """Load BatchNorm weight+bias+running stats."""
    sd[f"{pt_key}.weight"] = torch.from_numpy(_s(all_p[f"{ml_key}__Scale"]))
    sd[f"{pt_key}.bias"] = torch.from_numpy(_s(all_p[f"{ml_key}__Offset"]))
    sd[f"{pt_key}.running_mean"] = torch.from_numpy(_s(all_p[f"{ml_key}__TrainedMean"]))
    sd[f"{pt_key}.running_var"] = torch.from_numpy(_s(all_p[f"{ml_key}__TrainedVariance"]))


def _mrb(all_p, sd, pt, ml):
    """Load MultiResBlock: 3 conv+bn branches + shortcut."""
    for sfx, bn in [("c1", "bn1"), ("c2", "bn2"), ("c3", "bn3")]:
        _conv(all_p, sd, f"{pt}.{sfx}", f"{ml}_{sfx}")
        _bn(all_p, sd, f"{pt}.{bn}", f"{ml}_{bn}")
    _conv(all_p, sd, f"{pt}.sc", f"{ml}_sc")
    _bn(all_p, sd, f"{pt}.scbn", f"{ml}_scbn")


def _rams(all_p, sd, pt, ml):
    """Load RAMSAttention: sp_conv + 3 channel MLPs."""
    w_key = f"{ml}_P_Sp_conv7__Weights"
    b_key = f"{ml}_P_Sp_conv7__Bias"
    sd[f"{pt}.sp_conv.weight"] = torch.from_numpy(_t_spconv(all_p[w_key]))
    # Bias is a scalar float -> broadcast to [2]
    sd[f"{pt}.sp_conv.bias"] = torch.tensor(
        [float(all_p[b_key])] * 2, dtype=torch.float32)
    for pool in ["Avg", "Max", "Rng"]:
        p = pool.lower()
        _conv(all_p, sd, f"{pt}.mlp_{p}_1", f"{ml}_P_Ch_mlp{pool}1")
        _conv(all_p, sd, f"{pt}.mlp_{p}_2", f"{ml}_P_Ch_mlp{pool}2")


def _aspp(all_p, sd, pt, ml):
    """Load ASPPBridge."""
    for b in ["b1", "b2", "b3"]:
        _conv(all_p, sd, f"{pt}.{b}_conv", f"{ml}_{b}_conv")
        _bn(all_p, sd, f"{pt}.{b}_bn", f"{ml}_{b}_bn")
    _conv(all_p, sd, f"{pt}.fusion_conv", f"{ml}_fusion_conv")
    _bn(all_p, sd, f"{pt}.fusion_bn", f"{ml}_fusion_bn")


def _respath(all_p, sd, pt, ml, n):
    """Load ResPath with n blocks."""
    for i in range(1, n + 1):
        base = f"{ml}_{i}"
        _conv(all_p, sd, f"{pt}.conv1_{i}", f"{base}_1x1")
        _bn(all_p, sd, f"{pt}.bn1_{i}", f"{base}_bn1")
        _conv(all_p, sd, f"{pt}.conv3_{i}", f"{base}_3x3")
        _bn(all_p, sd, f"{pt}.bn2_{i}", f"{base}_bn2")
    _conv(all_p, sd, f"{pt}.sc", f"{ml}_sc")


# ===========================================================================
# Main loader
# ===========================================================================

def load_bp_network(weights_path: str) -> BPWaveformNet:
    """Create BPWaveformNet and load MATLAB-extracted weights.

    Args:
        weights_path: Path to bp_weights.mat (from extract_weights.m)

    Returns:
        BPWaveformNet in eval mode with all weights loaded.
    """
    mat = loadmat(weights_path, simplify_cells=True)
    w = mat["weights"]
    st = mat.get("state", {})
    all_p = dict(w)
    if st and hasattr(st, "keys"):
        all_p.update(st)

    model = BPWaveformNet()
    sd = model.state_dict()

    # Input projection (Conv1d(1->32,1) - weight squeezed to (32,))
    sd["input_proj.weight"] = torch.from_numpy(
        all_p["input_projection__Weights"].copy().reshape(32, 1, 1))
    sd["input_proj.bias"] = torch.from_numpy(
        _s(all_p["input_projection__Bias"]))

    # Encoder
    _mrb(all_p, sd, "enc1", "enc1")
    _rams(all_p, sd, "rams1", "rams1")
    _mrb(all_p, sd, "enc2", "enc2")
    _rams(all_p, sd, "rams2", "rams2")
    _mrb(all_p, sd, "enc3", "enc3")
    _rams(all_p, sd, "rams3", "rams3")
    _mrb(all_p, sd, "enc4", "enc4")
    _rams(all_p, sd, "rams4", "rams4")

    # Bridge
    _mrb(all_p, sd, "bridge_base", "bridge_base")
    _rams(all_p, sd, "rams_bridge", "rams_bridge")
    _aspp(all_p, sd, "bridge_aspp", "bridge_aspp")

    # Decoder
    _tconv(all_p, sd, "up4", "up4")
    _respath(all_p, sd, "res4", "res4", 1)
    _mrb(all_p, sd, "dec4", "dec4")
    _rams(all_p, sd, "rams_dec4", "rams_dec4")

    _tconv(all_p, sd, "up3", "up3")
    _respath(all_p, sd, "res3", "res3", 2)
    _mrb(all_p, sd, "dec3", "dec3")
    _rams(all_p, sd, "rams_dec3", "rams_dec3")

    _tconv(all_p, sd, "up2", "up2")
    _respath(all_p, sd, "res2", "res2", 3)
    _mrb(all_p, sd, "dec2", "dec2")
    _rams(all_p, sd, "rams_dec2", "rams_dec2")

    _tconv(all_p, sd, "up1", "up1")
    _respath(all_p, sd, "res1", "res1", 4)
    _mrb(all_p, sd, "dec1", "dec1")
    _rams(all_p, sd, "rams_dec1", "rams_dec1")

    # Output (Conv1d(64->1,1) - weight squeezed to (64,), bias is scalar)
    sd["final_conv.weight"] = torch.from_numpy(
        np.asarray(all_p["final_conv__Weights"]).copy().reshape(1, 64, 1))
    fb = all_p["final_conv__Bias"]
    sd["final_conv.bias"] = torch.tensor([float(fb)])

    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


# ===========================================================================
# High-level inference wrapper
# ===========================================================================

class BPInference:
    """Convenience wrapper for BP waveform inference.

    Usage:
        bp = BPInference("bp_matlab/bp_weights.mat")
        waveform_mmhg = bp.predict(pulse_wave_256)  # np[256] -> np[256] in mmHg
    """

    def __init__(self, weights_path: str):
        mat = loadmat(weights_path, simplify_cells=True)
        self.x_min = float(mat["x_min"])
        self.x_max = float(mat["x_max"])
        self.y_min = float(mat["y_min"])
        self.y_max = float(mat["y_max"])
        self.x_rng = (self.x_max - self.x_min) + 1e-8
        self.y_rng = self.y_max - self.y_min
        self.model = load_bp_network(weights_path)

        # Sanity check: verify output is not NaN on zero input
        zero_out = self.predict(np.zeros(256, dtype=np.float32))
        if np.any(np.isnan(zero_out)):
            raise RuntimeError(
                "BP network sanity check failed: NaN in output. "
                "Check weight loading or bp_weights.mat integrity."
            )

    def predict(self, wave_256: np.ndarray) -> np.ndarray:
        """Run inference on a 256-point pulse wave.

        Args:
            wave_256: float32 [256], clean pulse wave resampled to 50 Hz

        Returns:
            float32 [256], reconstructed BP waveform in mmHg
        """
        x = np.asarray(wave_256, dtype=np.float32)
        x_norm = (x - self.x_min) / self.x_rng
        print(f"[BPInference] input range=[{float(np.min(x)):.4f}, {float(np.max(x)):.4f}]  "
              f"x_min={self.x_min:.4f} x_max={self.x_max:.4f}  "
              f"norm_range=[{float(np.min(x_norm)):.4f}, {float(np.max(x_norm)):.4f}]")
        x_t = torch.from_numpy(x_norm).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            y_t = self.model(x_t)

        y = y_t.squeeze().numpy()
        print(f"[BPInference] sigmoid_out range=[{float(np.min(y)):.6f}, {float(np.max(y)):.6f}]  "
              f"y_min={self.y_min:.2f} y_max={self.y_max:.2f}")
        return y * self.y_rng + self.y_min
