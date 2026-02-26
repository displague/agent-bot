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

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None


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

    if bnb is None:
        print("[quantize] bitsandbytes not found. Falling back to native PyTorch quantization (slow/VRAM-heavy)…")
    else:
        print(f"[quantize] Applying {quantize_type} quantization (bitsandbytes)…")

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

    # Accurate size estimate based on bits-per-parameter
    total_bytes = 0
    for name, module in model.named_modules():
        if isinstance(module, _BNBLinear4bit):
            total_bytes += (module.in_features * module.out_features * 0.5)
        elif isinstance(module, (_BNBLinear8bit, _QuantizedLinear)):
            total_bytes += (module.in_features * module.out_features * 1.0)
        elif isinstance(module, nn.Linear):
            total_bytes += (module.in_features * module.out_features * 2.0) # assume fp16/bf16
            
    # Add buffers (norm, embeddings, etc.)
    for b in model.buffers():
        total_bytes += b.numel() * 2 # assume 16-bit
        
    size_gb = total_bytes / (1024 ** 3)
    print(f"[quantize] Estimated model size after quantization: ~{size_gb:.1f} GB")

    return model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _BNBLinear4bit(nn.Module):
    """Linear layer using bitsandbytes 4-bit NF4 quantization."""
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        
        # We initialize the bnb layer
        self.bnb_layer = bnb.nn.Linear4bit(
            self.in_features, 
            self.out_features, 
            bias=linear.bias is not None,
            compute_dtype=torch.bfloat16,
            quant_type="nf4",
            double_quant=True
        )
        # Load weights into the bnb layer
        self.bnb_layer.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            self.bnb_layer.bias.data.copy_(linear.bias.data)

    @property
    def weight(self) -> torch.Tensor:
        """Dequantize weights on-the-fly for direct access by multi_linear fused ops."""
        # Use bnb's internal dequantization — works on GPU and CPU (for recent versions)
        return bnb.functional.dequantize_4bit(
            self.bnb_layer.weight.data, 
            self.bnb_layer.weight.quant_state
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bnb_layer(x)


class _BNBLinear8bit(nn.Module):
    """Linear layer using bitsandbytes 8-bit quantization."""
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        
        self.bnb_layer = bnb.nn.Linear8bitLt(
            self.in_features, 
            self.out_features, 
            bias=linear.bias is not None,
            has_fp16_weights=False
        )
        self.bnb_layer.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            self.bnb_layer.bias.data.copy_(linear.bias.data)

    @property
    def weight(self) -> torch.Tensor:
        # 8-bit dequantization logic
        if self.bnb_layer.weight.device.type == 'cuda':
            # bitsandbytes 8bit doesn't have a simple functional dequantize like 4bit,
            # but we can get the float weight via the .CB property or similar if needed.
            # For now, cast should work for compatibility checks.
            return self.bnb_layer.weight.data.to(torch.float32)
        return self.bnb_layer.weight.data.to(torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bnb_layer(x)


class _QuantizedLinear(nn.Module):
    """Fallback native PyTorch linear layer with per-channel int8 weights."""
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
        if self.bias is not None:
            w = w.to(self.bias.dtype)
        return w

    @classmethod
    def from_linear(cls, linear: nn.Linear, quantize_type: str = "8bit") -> "_QuantizedLinear":
        q = cls(linear.in_features, linear.out_features,
                bias=linear.bias is not None, quantize_type=quantize_type)

        weight = linear.weight.data.float()
        weight_max = weight.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)

        int_range = 127 if quantize_type == "8bit" else 7
        scale = weight_max / int_range
        w_q = (weight / scale).round().clamp(-int_range, int_range).to(torch.int8)

        q.weight_quantized.copy_(w_q)
        q.weight_scale.copy_(scale.squeeze().half())
        if linear.bias is not None:
            q.bias.data.copy_(linear.bias.data.half())

        return q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight_quantized.float() * self.weight_scale.unsqueeze(1).float()
        weight = weight.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return nn.functional.linear(x, weight, bias)


# Layers whose weights are accessed directly by fused ops or embedding look-ups —
# replacing them with QuantizedLinear would break the model.
_SKIP_PATTERNS = [
    "embed", "emb", "lm_head", "out_proj", "output",
    "norm", "gating", "linear_in", "linear_out",
]
_MIN_LAYER_SIZE = 1024  # don't bother quantizing tiny layers


def _quantize_inplace(model: nn.Module, quantize_type: str) -> int:
    """Replace eligible nn.Linear layers with a quantized version in-place."""
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
        if bnb is not None:
            if quantize_type == "4bit":
                setattr(parent, attr, _BNBLinear4bit(module))
            else:
                setattr(parent, attr, _BNBLinear8bit(module))
        else:
            setattr(parent, attr, _QuantizedLinear.from_linear(module, quantize_type))
        del module

    if replacements:
        gc.collect()

    return len(replacements)
