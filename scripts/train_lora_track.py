#!/usr/bin/env python3
"""LoRA training for D2-ER (Type-A) or D3a-ENT (Type-B entanglement)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
NLA_SRC = ROOT.parent / "nla_rsa_study" / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(NLA_SRC))

from common import cleanup_mps, get_device, load_causal_lm, model_device, resolve_torch_dtype  # noqa: E402

from src.hooks import EditSpec, InterventionStack, register_intervention_hooks, remove_hooks  # noqa: E402
from src.mandatory_fuse import MandatoryFuse, register_mandatory_fuse_hooks, remove_fuse_hooks  # noqa: E402
from src.vectors import load_vectors  # noqa: E402

logger = logging.getLogger(__name__)


class ChatJsonlDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_len: int):
        self.rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        messages = self.rows[idx]["messages"]
        user_msgs = [m for m in messages if m["role"] != "assistant"]
        prompt_text = self.tokenizer.apply_chat_template(
            user_msgs, tokenize=False, add_generation_prompt=True
        )
        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        ids = self.tokenizer(
            full_text, truncation=True, max_length=self.max_len, return_tensors="pt"
        )["input_ids"].squeeze(0)
        plen = len(self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
        labels = ids.clone()
        labels[: min(plen, labels.shape[0] - 1)] = -100
        fz = bool(self.rows[idx].get("fuse_zero", False))
        return ids, labels, fz


def collate(batch):
    max_l = max(x[0].shape[0] for x in batch)
    pad = 0
    ids, labels, mask, fuse_zero = [], [], [], []
    for i, lab, fz in batch:
        pad_len = max_l - i.shape[0]
        ids.append(F.pad(i, (0, pad_len), value=pad))
        labels.append(F.pad(lab, (0, pad_len), value=-100))
        mask.append(torch.cat([torch.ones(i.shape[0]), torch.zeros(pad_len)]))
        fuse_zero.append(fz)
    return (
        torch.stack(ids),
        torch.stack(labels),
        torch.stack(mask).bool(),
        torch.tensor(fuse_zero, dtype=torch.bool),
    )


def ce_loss(model, input_ids, labels, attn_mask):
    out = model(input_ids=input_ids, attention_mask=attn_mask)
    logits = out.logits[:, :-1, :].float().contiguous()
    tgt = labels[:, 1:].contiguous()
    return F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1), ignore_index=-100)


def load_d3a_mix(data_dir: Path, harmful_frac: float, seed: int) -> list[dict]:
    benign = [json.loads(l) for l in (data_dir / "d3a_benign_train.jsonl").read_text().splitlines() if l.strip()]
    harmful = [json.loads(l) for l in (data_dir / "d3a_harmful_refuse_train.jsonl").read_text().splitlines() if l.strip()]
    import random

    rng = random.Random(seed)
    n_h = int(len(benign) * harmful_frac / max(1 - harmful_frac, 1e-6))
    n_h = min(n_h, len(harmful))
    mix = benign + rng.sample(harmful, n_h)
    rng.shuffle(mix)
    tmp = data_dir / "_d3a_mix_train.jsonl"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in mix:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return mix


def load_d3c_fuse_mix(data_dir: Path, seed: int) -> Path:
    """Shuffle benign (fuse on) + harmful refuse (fuse zero at train time)."""
    import random

    path = data_dir / "d3c_fuse_train.jsonl"
    if not path.exists():
        load_d3a_mix(data_dir, 0.35, seed)
        return data_dir / "_d3a_mix_train.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(rows)
    tmp = data_dir / "_d3c_fuse_mix_train.jsonl"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return tmp


def rfa_stack_from_payload(payload: dict, layer: int, scale: float) -> InterventionStack:
    meta = payload.get("metadata", {})
    rfa = meta.get("refusal", {}).get("rfa_harmless_mean", {})
    coeff = float(rfa.get(layer, rfa.get(str(layer), 0.0))) * scale
    return InterventionStack(
        edits=[EditSpec(type="rfa_r", axis="refusal", layer=layer, harmless_mean_coeff=coeff)],
        vectors=payload["axes"],
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument(
        "--track",
        choices=["d2_er", "d3a_ent", "d3c_fuse", "d3c_fuse_v3c", "d3c_fuse_v3d"],
        required=True,
    )
    p.add_argument("--config", type=Path, default=ROOT / "configs/d3_lora_train.yaml")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    data_dir = args.data_dir or ROOT / cfg["paths"]["data_dir"]
    max_steps = args.max_steps or cfg["train"]["max_steps"]
    device = get_device("cuda" if torch.cuda.is_available() else "auto")
    dtype = resolve_torch_dtype(cfg["model"]["dtype"], device)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.track == "d2_er":
        train_path = data_dir / "d2_er_train.jsonl"
    elif args.track.startswith("d3c_fuse"):
        train_path = load_d3c_fuse_mix(data_dir, cfg["data"]["seed"])
    else:
        load_d3a_mix(data_dir, cfg["train"]["harmful_fraction_d3a"], cfg["data"]["seed"])
        train_path = data_dir / "_d3a_mix_train.jsonl"

    ds = ChatJsonlDataset(train_path, tokenizer, cfg["train"]["max_seq_len"])
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collate)

    model = load_causal_lm(cfg["model"]["name"], device, dtype)
    lora = LoraConfig(
        r=cfg["train"]["lora_r"],
        lora_alpha=cfg["train"]["lora_alpha"],
        lora_dropout=cfg["train"]["lora_dropout"],
        target_modules=cfg["train"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    hidden = model.config.hidden_size
    fuse = None
    fuse_init_scale = cfg["train"].get("fuse_init_scale", 0.02)

    fuse_layer = cfg["train"].get("fuse_layer", cfg["train"]["rfa_layer"])
    lambda_kill = cfg["train"].get("lambda_fuse_kill", 0.2)
    p_zero = cfg["train"].get("fuse_zero_prob", 0.0)
    fuse_warmup = cfg["train"].get("fuse_warmup_steps", 200)
    fuse_kill_margin = cfg["train"].get("fuse_kill_margin", 1.0)
    fuse_loss_cap = cfg["train"].get("fuse_loss_cap", 15.0)
    if args.track.startswith("d3c_fuse"):
        fuse = MandatoryFuse(hidden, rank=cfg["train"].get("fuse_rank", 64), init_scale=fuse_init_scale)
        fuse = fuse.to(model_device(model), dtype=torch.float32)

    model.print_trainable_parameters()
    model.train()
    lora_params = [p for p in model.parameters() if p.requires_grad]
    opt_groups = [{"params": lora_params, "lr": cfg["train"]["lr"]}]
    if fuse is not None:
        opt_groups.append(
            {"params": fuse.parameters(), "lr": cfg["train"].get("fuse_lr", 3e-5)}
        )
    opt = torch.optim.AdamW(opt_groups)
    params = [p for g in opt_groups for p in g["params"]]

    payload = load_vectors(ROOT / cfg["paths"]["vectors"]) if args.track == "d3a_ent" else None
    rfa_stack = None
    if args.track == "d3a_ent":
        rfa_stack = rfa_stack_from_payload(
            payload, cfg["train"]["rfa_layer"], cfg["train"]["rfa_scale"]
        )

    out_dir = ROOT / cfg["paths"]["adapters_dir"] / args.track
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    accum = cfg["train"]["grad_accum"]
    opt.zero_grad()
    pbar = tqdm(total=max_steps, desc=args.track)

    use_tagged_fuse_zero = args.track == "d3c_fuse_v3d"

    while step < max_steps:
        for input_ids, labels, attn, fuse_zero_flags in loader:
            input_ids = input_ids.to(model_device(model))
            labels = labels.to(model_device(model))
            attn = attn.to(model_device(model))

            if args.track == "d2_er":
                loss = ce_loss(model, input_ids, labels, attn) / accum
            elif args.track == "d3a_ent":
                loss_clean = ce_loss(model, input_ids, labels, attn)
                handles = register_intervention_hooks(model, rfa_stack)
                loss_rfa = ce_loss(model, input_ids, labels, attn)
                remove_hooks(handles)
                lam = cfg["train"]["lambda_ent"]
                loss = (loss_clean - lam * loss_rfa) / accum
            else:
                import random

                if use_tagged_fuse_zero:
                    force_zero = step >= fuse_warmup and bool(fuse_zero_flags[0].item())
                    handles = register_mandatory_fuse_hooks(
                        model, fuse, fuse_layer, force_zero=force_zero
                    )
                    loss = ce_loss(model, input_ids, labels, attn) / accum
                    remove_fuse_hooks(handles)
                else:
                    handles = register_mandatory_fuse_hooks(
                        model, fuse, fuse_layer, force_zero=False
                    )
                    loss_on = ce_loss(model, input_ids, labels, attn)
                    remove_fuse_hooks(handles)
                    loss = loss_on / accum
                    if step >= fuse_warmup and random.random() < p_zero:
                        handles = register_mandatory_fuse_hooks(
                            model, fuse, fuse_layer, force_zero=True
                        )
                        with torch.no_grad():
                            loss_zero = ce_loss(model, input_ids, labels, attn)
                        remove_fuse_hooks(handles)
                        if torch.isfinite(loss_zero) and torch.isfinite(loss_on):
                            gap = torch.clamp(
                                loss_zero - loss_on - fuse_kill_margin,
                                min=0.0,
                                max=fuse_loss_cap,
                            )
                            loss = (loss_on + lambda_kill * gap) / accum

            if not torch.isfinite(loss):
                logger.warning("step=%s non-finite loss, skip backward", step)
                opt.zero_grad(set_to_none=True)
                step += 1
                pbar.update(1)
                if step >= max_steps:
                    break
                continue
            loss.backward()
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, cfg["train"].get("max_grad_norm", 1.0))
                opt.step()
                opt.zero_grad()

            if step % cfg["train"]["logging_steps"] == 0:
                logger.info("step=%s loss=%.4f", step, loss.item() * accum)

            if step > 0 and step % cfg["train"]["save_steps"] == 0:
                model.save_pretrained(out_dir / f"checkpoint-{step}")

            step += 1
            pbar.update(1)
            if step >= max_steps:
                break

    pbar.close()
    model.save_pretrained(out_dir)
    if fuse is not None:
        torch.save(fuse.state_dict(), out_dir / "mandatory_fuse.pt")
    (out_dir / "train_meta.json").write_text(
        json.dumps({"track": args.track, "max_steps": max_steps, "config": str(args.config)}, indent=2)
    )
    logger.info("Saved adapter %s", out_dir)
    cleanup_mps()


if __name__ == "__main__":
    main()
