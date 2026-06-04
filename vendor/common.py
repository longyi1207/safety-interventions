"""Shared LM utilities (vendored from nla_rsa_study/src/common.py for standalone repo)."""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def get_device(prefer: str | None = None) -> torch.device:
    """Return torch.device for mps, cuda, or cpu (in that order unless prefer set)."""
    order = ["mps", "cuda", "cpu"]
    if prefer is not None:
        prefer = prefer.lower()
        if prefer in ("auto", "default", ""):
            prefer = None
        elif prefer not in order:
            raise ValueError(f"Unknown device {prefer!r}; expected one of {order + ['auto']}")
        else:
            order = [prefer] + [d for d in order if d != prefer]

    for name in order:
        if name == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if name == "cpu":
            return torch.device("cpu")
    return torch.device("cpu")


def resolve_torch_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    """Map config dtype string to torch.dtype; fall back float16 on MPS if bf16 unsupported."""
    dtype_name = dtype_name.lower()
    if dtype_name in ("bfloat16", "bf16"):
        if device.type == "mps":
            supported = getattr(torch.backends.mps, "is_bfloat16_supported", lambda: False)
            if not supported():
                logger.warning("MPS bf16 unsupported; falling back to float16")
                return torch.float16
        return torch.bfloat16
    if dtype_name in ("float16", "fp16", "half"):
        return torch.float16
    if dtype_name in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"Unknown dtype {dtype_name!r}")


def model_device(model: torch.nn.Module) -> torch.device:
    """Device of first parameter (works with device_map models)."""
    return next(model.parameters()).device


def load_causal_lm(model_name: str, device: torch.device, dtype: torch.dtype | None = None):
    """Load HF causal LM; use device_map on CUDA for fast load + generate."""
    from transformers import AutoModelForCausalLM

    load_kwargs: dict = {"trust_remote_code": True}
    if device.type == "cuda":
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["device_map"] = "cuda"
        load_kwargs["attn_implementation"] = "sdpa"
    else:
        load_kwargs["torch_dtype"] = dtype or torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if device.type != "cuda":
        model = model.to(device)
    return model.eval()


def load_prompts_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL prompts; skip blank lines."""
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prompt_message(row: dict[str, Any]) -> str:
    """User message text for chat template (emotion_50 schema)."""
    return row.get("chat_user_message") or row["text"]


def build_chat_inputs(
    tokenizer: Any,
    message: str,
    add_generation_prompt: bool = True,
) -> dict[str, Any]:
    """Apply chat template; return input_ids tensor [1, T] and raw id list."""
    messages = [{"role": "user", "content": message}]
    out = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=False,
    )
    if isinstance(out, list) and out and isinstance(out[0], int):
        input_ids_list = list(out)
    elif hasattr(out, "input_ids"):
        input_ids_list = list(out["input_ids"])
    elif isinstance(out, str):
        input_ids_list = tokenizer.encode(out, add_special_tokens=False)
    else:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        input_ids_list = tokenizer.encode(text, add_special_tokens=False)
    input_ids = torch.tensor([input_ids_list], dtype=torch.long)
    return {"input_ids": input_ids, "input_ids_list": input_ids_list}


def get_extraction_index(
    input_ids: list[int] | torch.Tensor,
    position: str = "last_prompt_token",
) -> int:
    """Return token index for hidden-state extraction."""
    if isinstance(input_ids, torch.Tensor):
        seq_len = int(input_ids.shape[-1])
    else:
        seq_len = len(input_ids)

    if seq_len == 0:
        raise ValueError("empty input_ids")

    if position in ("last_prompt_token", "pre_assistant_generation"):
        return seq_len - 1

    raise ValueError(
        f"Unknown extraction position {position!r}; "
        "expected last_prompt_token or pre_assistant_generation"
    )


def resolve_hf_checkpoint(repo_id_or_path: str | Path) -> Path:
    """Local path with nla_meta.yaml; download HF snapshot if needed."""
    p = Path(repo_id_or_path)
    if p.is_dir() and (p / "nla_meta.yaml").exists():
        return p.resolve()
    if p.is_dir():
        return p.resolve()
    from huggingface_hub import snapshot_download

    snap = snapshot_download(str(repo_id_or_path))
    return Path(snap)


def release_gpu(*objs: Any) -> None:
    """Drop model/tokenizer refs and empty device cache between eval steps."""
    for obj in objs:
        if obj is not None:
            del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def cleanup_mps(*objs: Any) -> None:
    """Alias for release_gpu (optional objects to delete first)."""
    release_gpu(*objs)
