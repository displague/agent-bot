"""
quantize.py — Per-channel int8 quantization for PersonaPlex LM weights.

NOTE (empirical): On PersonaPlex/Moshi the VRAM reduction is ~0.2 GB rather than
the theoretical ~6-9 GB, because Moshi's large weight matrices live in fused-ops
layers (gating/linear_in/linear_out/out_proj) that must be skipped to avoid
breaking the model. Only small residual layers get replaced. A bitsandbytes- or
GGUF-based approach would be needed for meaningful VRAM reduction.

Usage:
    from quantize import quantize_model_after_load
    lm = quantize_model_after_load(lm, quantize_type="8bit", device="cuda")

Ported from ComfyUI_PersonaPlexMF/quantize.py (squarewulf, MIT License) with
minor adaptations for agent-bot conventions.
"""

import gc
from typing import Optional

import torch
import torch.nn as nn


def quantize_model_after_load(
    model: nn.Module,
    quantize_type: str,
    device: str = "cuda",
) -> nn.Module:
    """Quantize a model after loading weights and move it to *device*.

    Args:
        model: Loaded model (may be on CPU or CUDA).
        quantize_type: ``"8bit"``, ``"4bit"``, or ``""``/``"none"`` to skip.
        device: Target device string.

    Returns:
        The (possibly quantized) model on *device*.
    """
    if quantize_type in ("", "none", "None", None):
        return model.to(device)

    if not torch.cuda.is_available():
        raise RuntimeError("Quantization requires a CUDA-capable GPU.")

    print(f"[quantize] Applying {quantize_type} quantization (PyTorch native)…")

    # Quantize on CPU to avoid OOM during weight conversion
    model = model.to("cpu")
    num_replaced = _quantize_inplace(model, quantize_type)

    if num_replaced == 0:
        print("[quantize] Warning: no layers were quantized — check skip_patterns")
    else:
        print(f"[quantize] {num_replaced} Linear layers replaced with QuantizedLinear")

    model = model.to(device)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Rough size estimate
    buffers = sum(b.numel() for b in model.buffers())
    params = sum(p.numel() for p in model.parameters())
    size_gb = (buffers * 1 + params * 2) / (1024 ** 3)
    print(f"[quantize] Estimated model size after quantization: ~{size_gb:.1f} GB")

    return model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _QuantizedLinear(nn.Module):
    """Linear layer with per-channel int8 weights and fp16 scales.

    Weights are stored as int8; scales are fp16 vectors of length out_features.
    The forward pass dequantizes on-the-fly: weight_fp = weight_int8 * scale.
    Per-channel (rather than per-tensor) quantization preserves more accuracy.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 quantize_type: str = "8bit"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantize_type = quantize_type

        self.register_buffer("weight_quantized",
                             torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("weight_scale",
                             torch.zeros(out_features, dtype=torch.float16))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)

    @property
    def weight(self) -> torch.Tensor:
        """Dequantize weights on-the-fly for compatibility with direct .weight access."""
        w = self.weight_quantized.float() * self.weight_scale.unsqueeze(1).float()
        # Ensure we return the same dtype as bias (usually float16/bfloat16)
        if self.bias is not None:
            w = w.to(self.bias.dtype)
        return w

    @classmethod
    def from_linear(cls, linear: nn.Linear, quantize_type: str = "8bit") -> "_QuantizedLinear":
        q = cls(linear.in_features, linear.out_features,
                bias=linear.bias is not None, quantize_type=quantize_type)

        weight = linear.weight.data.float()
        # Per-channel: one scale per output neuron
        weight_max = weight.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)

        int_range = 127 if quantize_type == "8bit" else 7  # "4bit" uses reduced range
        scale = weight_max / int_range
        w_q = (weight / scale).round().clamp(-int_range, int_range).to(torch.int8)

        q.weight_quantized.copy_(w_q)
        q.weight_scale.copy_(scale.squeeze().half())
        if linear.bias is not None:
            q.bias.data.copy_(linear.bias.data.half())

        return q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize on-the-fly (cheap: just a multiply + cast)
        weight = self.weight_quantized.float() * self.weight_scale.unsqueeze(1).float()
        weight = weight.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return nn.functional.linear(x, weight, bias)

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"bias={self.bias is not None}, quant={self.quantize_type}")


# Layers whose weights are accessed directly by fused ops or embedding look-ups —
# replacing them with QuantizedLinear would break the model.
_SKIP_PATTERNS = [
    "embed", "emb", "lm_head", "out_proj", "output",
    "norm", "gating",
]
# Note: "out_proj" restored to skip list. "linear_in" and "linear_out" remain candidates.
_MIN_LAYER_SIZE = 1024  # don't bother quantizing tiny layers


def _quantize_inplace(model: nn.Module, quantize_type: str) -> int:
    """Replace eligible nn.Linear layers with _QuantizedLinear in-place."""
    replacements = []
    module_map = dict(model.named_modules())

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(p in name.lower() for p in _SKIP_PATTERNS):
            continue
        if module.in_features * module.out_features < _MIN_LAYER_SIZE:
            continue

        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent = module_map[parts[0]]
            attr = parts[1]
        else:
            parent = model
            attr = name
        replacements.append((parent, attr, module))

    for parent, attr, module in replacements:
        setattr(parent, attr, _QuantizedLinear.from_linear(module, quantize_type))
        del module

    if replacements:
        gc.collect()

    return len(replacements)
