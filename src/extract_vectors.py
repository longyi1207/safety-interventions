"""Extract contrast vectors (refusal, evil, harmfulness, qualities)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
NLA_SRC = ROOT.parent / "nla_rsa_study" / "src"
if str(NLA_SRC) not in sys.path:
    sys.path.insert(0, str(NLA_SRC))

from common import build_chat_inputs, cleanup_mps, get_device, load_causal_lm, model_device, resolve_torch_dtype  # noqa: E402

from .chat_extract import extract_assistant_mean_hidden
from .config_loader import load_config, repo_root
from .harmbench_data import load_evil_contrast_conversations, load_harmful_behaviors, load_harmless_xstest
from .vectors import normalize_layers, save_vectors

logger = logging.getLogger(__name__)


@torch.no_grad()
def extract_mean_hidden_prompt(
    model, tokenizer, messages: list[str], device, n_layers: int
) -> torch.Tensor:
    """Last user token (refusal axis)."""
    sums = None
    count = 0
    for msg in tqdm(messages, desc="extract", leave=False):
        inp = build_chat_inputs(tokenizer, msg)
        ids = inp["input_ids"].to(device)
        idx = len(inp["input_ids_list"]) - 1
        out = model(input_ids=ids, output_hidden_states=True)
        stacked = torch.stack(
            [out.hidden_states[i + 1][0, idx].float().cpu() for i in range(n_layers)]
        )
        sums = stacked if sums is None else sums + stacked
        count += 1
    return sums / max(count, 1)


def contrast_prompt_vectors(pos, neg, model, tokenizer, device):
    n_layers = model.config.num_hidden_layers
    pos_m = extract_mean_hidden_prompt(model, tokenizer, pos, device, n_layers)
    neg_m = extract_mean_hidden_prompt(model, tokenizer, neg, device, n_layers)
    diff = pos_m - neg_m
    vecs = {i: diff[i] for i in range(n_layers)}
    return normalize_layers(vecs), pos_m, neg_m


def contrast_assistant_vectors(pos_convos, neg_convos, model, tokenizer, device):
    n_layers = model.config.num_hidden_layers
    pos_m = extract_assistant_mean_hidden(model, tokenizer, pos_convos, device, n_layers)
    neg_m = extract_assistant_mean_hidden(model, tokenizer, neg_convos, device, n_layers)
    diff = pos_m - neg_m
    vecs = {i: diff[i] for i in range(n_layers)}
    norm = normalize_layers(vecs)
    meta_proj = {}
    for i in range(n_layers):
        d = norm[i]
        meta_proj[i] = {
            "pos_proj": float(torch.dot(pos_m[i].float(), d).item()),
            "neg_proj": float(torch.dot(neg_m[i].float(), d).item()),
            "separation": float(torch.dot(pos_m[i].float(), d).item() - torch.dot(neg_m[i].float(), d).item()),
        }
    return norm, meta_proj


def extract_refusal(model, tokenizer, device, n_pairs: int = 50):
    harmful = load_harmful_behaviors(limit=n_pairs)
    harmless = load_harmless_xstest(limit=n_pairs)
    n = min(len(harmful), len(harmless))
    logger.info("Refusal extraction: %d harmful + %d harmless (last user token)", n, n)
    vecs, pos_m, neg_m = contrast_prompt_vectors(
        harmful[:n], harmless[:n], model, tokenizer, device
    )
    harmless_proj = {i: float(torch.dot(neg_m[i].float(), vecs[i]).item()) for i in vecs}
    harmful_proj = {i: float(torch.dot(pos_m[i].float(), vecs[i]).item()) for i in vecs}
    meta = {
        "method": "harmful_vs_harmless_last_user_token",
        "rfa_harmless_mean": harmless_proj,
        "harmful_proj_mean": harmful_proj,
        "n_harmful": n,
        "n_harmless": n,
    }
    return vecs, meta


def extract_evil(model, tokenizer, device, n_pairs: int = 40, same_system: bool = True):
    pos, neg = load_evil_contrast_conversations(limit=n_pairs, same_system=same_system)
    logger.info(
        "Evil extraction (Chen-style): %d pairs, assistant-token mean, evil vs neutral system",
        len(pos),
    )
    vecs, layer_meta = contrast_assistant_vectors(pos, neg, model, tokenizer, device)
    # Best layer = max pos/neg separation on axis
    best_layer = max(layer_meta, key=lambda l: layer_meta[l]["separation"])
    meta = {
        "method": "chen_assistant_token_contrast",
        "same_system_extraction": True,
        "n_pairs": len(pos),
        "layer_separation": layer_meta,
        "best_layer": int(best_layer),
        "best_separation": layer_meta[best_layer]["separation"],
    }
    logger.info("Evil best layer L%s separation=%.3f", best_layer, meta["best_separation"])
    return vecs, meta


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=repo_root() / "configs/qwen7b_harmbench.yaml")
    p.add_argument("--axes", nargs="+", default=["refusal", "evil"])
    p.add_argument("--same-system", action="store_true", default=True)
    p.add_argument("--dual-system", action="store_true", help="Extract with EVIL_SYSTEM on pos branch")
    p.add_argument("--n-pairs", type=int, default=50)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--merge", action="store_true", help="Merge into existing checkpoint (keep other axes)")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["model"].get("device", "mps"))
    dtype = resolve_torch_dtype(cfg["model"]["dtype"], device)

    out = args.out or Path(cfg.get("paths", {}).get("vectors", "outputs/vectors/qwen7b_vectors.pt"))
    if not out.is_absolute():
        out = repo_root() / out

    payload = {"model": cfg["model"]["name"], "axes": {}, "metadata": {}}
    if args.merge and out.exists():
        import torch as _torch

        payload = _torch.load(out, map_location="cpu", weights_only=False)
        logger.info("Merging into existing %s", out)

    logger.info("Loading %s on %s", cfg["model"]["name"], device)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    model = load_causal_lm(cfg["model"]["name"], device, dtype)

    extractors = {
        "refusal": lambda m, t, d: extract_refusal(m, t, d, args.n_pairs),
        "evil": lambda m, t, d: extract_evil(m, t, d, args.n_pairs, same_system=not args.dual_system),
    }

    for axis in args.axes:
        if axis not in extractors:
            raise ValueError(f"Unknown axis {axis}; have {list(extractors)}")
        logger.info("Extracting axis: %s", axis)
        vecs, meta = extractors[axis](model, tokenizer, device)
        payload["axes"][axis] = vecs
        meta = dict(meta)
        if axis == "evil" and args.dual_system:
            meta["extraction_mode"] = "dual_system_evil_vs_neutral"
        payload["metadata"][axis] = meta

    save_vectors(out, payload)
    logger.info("Saved %s", out)
    cleanup_mps()


if __name__ == "__main__":
    main()
