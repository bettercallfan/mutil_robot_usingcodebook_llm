"""LLM V-adapter for MazeBots0 communication vectors."""

from __future__ import annotations

import glob
import os
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


def _is_cuda_graph_capturing() -> bool:
    """Return True when the current CUDA stream is recording a graph.

    ``torch.utils.checkpoint`` internally saves / restores RNG state, which
    is unsupported during CUDA graph capture.  We detect capture and skip
    checkpoint so the adapter can be safely traced.
    """
    if not torch.cuda.is_available():
        return False
    if hasattr(torch.cuda, "is_current_stream_capturing"):
        return torch.cuda.is_current_stream_capturing()
    return False


try:
    from transformers import AutoConfig, AutoModel
except Exception:  # pragma: no cover - stub mode still works without transformers
    AutoConfig = None
    AutoModel = None


MODEL_CANDIDATES = [
    "/nfsdat1/home/rfchenslm/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/895c3a49bc3fa70a340399125c650a463535e71c",
    "/nfsdat1/home/rfchenslm/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
    "/nfsdat1/home/rfchenslm/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
]

_BRIDGE_CACHE: Dict[Tuple[str, str, int], nn.Module] = {}


def _has_weights(path: str) -> bool:
    index_file = os.path.join(path, "model.safetensors.index.json")
    if not os.path.exists(index_file):
        return False
    # Accept any shard count (2, 4, …)
    return len(glob.glob(os.path.join(path, "model-00001-of-*.safetensors"))) > 0


def _pick_model_path(model_path: Optional[str]) -> Optional[str]:
    if model_path:
        return model_path

    for candidate in MODEL_CANDIDATES:
        if _has_weights(candidate):
            return candidate

    return None


class UpMLP(nn.Module):
    def __init__(
        self,
        v_dim: int = 64,
        hidden_dim: int = 512,
        n_tokens: int = 8,
        llm_dim: int = 4096,
    ):
        super().__init__()

        self.n_tokens = n_tokens
        self.llm_dim = llm_dim

        out_dim = n_tokens * llm_dim

        self.net = nn.Sequential(
            nn.Linear(v_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, v: Tensor) -> Tensor:
        squeeze_batch = v.dim() == 2
        if squeeze_batch:
            v = v.unsqueeze(1)

        batch_size, bot_count, _ = v.shape

        z = self.net(v)
        z = z.reshape(batch_size, bot_count, self.n_tokens, self.llm_dim)

        if squeeze_batch:
            z = z.squeeze(1)

        return z


class EmbeddingAligner(nn.Module):
    """Align pseudo-token embeddings to the LLM token embedding distribution.

    During initialisation the module reads the LLM's input embedding matrix and
    records its per-dimension mean and standard deviation.  At forward time each
    pseudo-token vector is independently standardised to zero-mean unit-variance
    and then re-scaled to match the LLM's embedding statistics.
    """

    def __init__(self, embed_weight: Tensor):
        super().__init__()
        with torch.no_grad():
            w = embed_weight.float()
            mean = w.mean(dim=0)  # [llm_dim]
            std = w.std(dim=0, unbiased=False).clamp_min(1e-6)
        self.register_buffer("embed_mean", mean)
        self.register_buffer("embed_std", std)

    def extra_repr(self) -> str:
        return f"llm_dim={self.embed_mean.numel()}"

    def forward(self, z: Tensor) -> Tensor:
        # z: [..., llm_dim]
        z_f32 = z.float()
        z_std = z_f32.std(dim=-1, keepdim=True).clamp_min(1e-6)
        z_mean = z_f32.mean(dim=-1, keepdim=True)
        z_norm = (z_f32 - z_mean) / z_std
        return (z_norm * self.embed_std + self.embed_mean).to(dtype=z.dtype)


class DownMLP(nn.Module):
    def __init__(
        self,
        v_dim: int = 64,
        hidden_dim: int = 512,
        n_tokens: int = 8,
        llm_dim: int = 4096,
    ):
        super().__init__()

        self.n_tokens = n_tokens
        self.llm_dim = llm_dim

        in_dim = n_tokens * llm_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, v_dim),
        )

    def forward(self, h: Tensor) -> Tensor:
        squeeze_batch = h.dim() == 3
        if squeeze_batch:
            h = h.unsqueeze(1)

        batch_size, bot_count, token_count, hidden_dim = h.shape

        if token_count != self.n_tokens:
            raise ValueError(f"expected n_tokens={self.n_tokens}, got {token_count}")
        if hidden_dim != self.llm_dim:
            raise ValueError(f"expected llm_dim={self.llm_dim}, got {hidden_dim}")

        v_hat = self.net(h.reshape(batch_size, bot_count, token_count * hidden_dim))

        if squeeze_batch:
            v_hat = v_hat.squeeze(1)

        return v_hat


class FrozenLLMBridge(nn.Module):
    def __init__(self, model_path: Optional[str], dtype: torch.dtype, hidden_layer: int):
        super().__init__()

        if AutoConfig is None or AutoModel is None:
            raise ImportError("transformers is required for bridge_mode='llama'")

        self.model_path = _pick_model_path(model_path)
        if self.model_path is None:
            raise FileNotFoundError("no local LLaMA weights found")

        config = AutoConfig.from_pretrained(self.model_path, local_files_only=True)

        self.model = AutoModel.from_pretrained(
            self.model_path,
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

        # SDPA's causal-mask shortcut calls torch.all(...) on CUDA tensors, which is not
        # safe under CUDA graph capture. Eager attention avoids that path.
        setattr(self.model.config, "_attn_implementation", "eager")

        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.hidden_layer = hidden_layer

    def get_embed_weight(self) -> Tensor:
        """Return the token embedding matrix ``[vocab_size, llm_dim]``."""
        return self.model.get_input_embeddings().weight.detach()

    def forward(self, z_seq: Tensor) -> Tensor:
        bridge_param = next(self.model.parameters())
        bridge_dtype = bridge_param.dtype

        z_seq_in = z_seq.to(dtype=bridge_dtype)

        outputs = self.model(
            inputs_embeds=z_seq_in,
            return_dict=True,
            use_cache=False,
        )
        return outputs.last_hidden_state.to(dtype=z_seq.dtype)


class IdentityBridge(nn.Module):
    def forward(self, z_seq: Tensor) -> Tensor:
        return z_seq


class LLMVAdapter(nn.Module):
    def __init__(
        self,
        model_path: Optional[str] = None,
        n_tokens: int = 8,
        v_dim: int = 64,
        llm_dim: int = 4096,
        hidden_dim: int = 512,
        hidden_layer: int = -1,
        output_scale: float = 0.05,
        dtype: Optional[torch.dtype] = None,
        bridge_mode: str = "auto",
    ):
        super().__init__()

        if dtype is None:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.n_tokens = n_tokens
        self.llm_dim = llm_dim
        self.hidden_layer = hidden_layer
        self.bridge_chunk_size = 1
        self.bridge_mode = bridge_mode

        self.up_mlp = UpMLP(
            v_dim=v_dim,
            hidden_dim=hidden_dim,
            n_tokens=n_tokens,
            llm_dim=llm_dim,
        )

        if bridge_mode == "stub":
            self.bridge = IdentityBridge()
            self.bridge_name = "stub"
        else:
            resolved_path = _pick_model_path(model_path)
            if resolved_path is None:
                if bridge_mode == "llama":
                    raise FileNotFoundError("no local LLaMA weights found")

                self.bridge = IdentityBridge()
                self.bridge_name = "stub"

            else:
                cache_key = (resolved_path, str(dtype), hidden_layer)
                bridge = _BRIDGE_CACHE.get(cache_key)

                if bridge is None:
                    bridge = FrozenLLMBridge(resolved_path, dtype=dtype, hidden_layer=hidden_layer)
                    _BRIDGE_CACHE[cache_key] = bridge

                self.bridge = bridge
                self.bridge_name = "llama"

        if self.bridge_name == "llama":
            self.aligner = EmbeddingAligner(self.bridge.get_embed_weight())
        else:
            self.aligner = nn.Identity()

        self.down_mlp = DownMLP(
            v_dim=v_dim,
            hidden_dim=hidden_dim,
            n_tokens=n_tokens,
            llm_dim=llm_dim,
        )

        self.last_aux: Dict[str, Tensor] = {}

    def forward(self, v_old: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        squeeze_batch = v_old.dim() == 2
        if squeeze_batch:
            v_old = v_old.unsqueeze(1)

        batch_size, bot_count, _ = v_old.shape

        z = self.up_mlp(v_old)
        z = self.aligner(z)
        z_seq = z.reshape(batch_size, bot_count * self.n_tokens, self.llm_dim)
        if batch_size <= self.bridge_chunk_size or _is_cuda_graph_capturing():
            h_seq = self.bridge(z_seq)
        else:
            h_seq_chunks = []
            for z_seq_chunk in z_seq.split(self.bridge_chunk_size, dim=0):
                h_seq_chunks.append(checkpoint(self.bridge, z_seq_chunk))
            h_seq = torch.cat(h_seq_chunks, dim=0)
        h = h_seq.reshape(batch_size, bot_count, self.n_tokens, self.llm_dim)
        v_hat = self.down_mlp(h)

        rec_loss = (v_hat - v_old.detach()).square().mean()

        aux = {
            "rec_loss": rec_loss,
            "v_old_norm": v_old.detach().norm(dim=-1).mean(),
            "z_norm": z.detach().norm(dim=-1).mean(),
            "v_hat_norm": v_hat.detach().norm(dim=-1).mean(),
            "bridge_active": torch.tensor(1 if self.bridge_name == "llama" else 0, device=v_hat.device),
        }

        self.last_aux = aux

        if squeeze_batch:
            v_hat = v_hat.squeeze(1)

        return v_hat, aux
