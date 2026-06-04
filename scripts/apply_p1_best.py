#!/usr/bin/env python3
"""Apply P1 sweep winners to config + run C0/C1/C2/C4 on dev 20."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT.parent / "nla_rsa_study" / ".venv" / "bin" / "python"
SWEEP = ROOT / "outputs/ablations/p1_sweep_results.json"
CFG = ROOT / "configs/qwen7b_harmbench.yaml"


def main():
    if not SWEEP.exists():
        print("Missing sweep results; run src.sweep_p1 first", file=sys.stderr)
        sys.exit(1)

    with open(SWEEP) as f:
        s = json.load(f)
    br, be = s["best_refusal"], s["best_evil"]

    with open(CFG) as f:
        cfg = yaml.safe_load(f)

    cfg["interventions"]["default_layer"] = br["layer"]
    cfg["interventions"]["alpha_star"] = {"refusal": 1.0, "evil": be["alpha"]}
    cfg["interventions"]["combinations"]["C1"]["edits"][0]["layer"] = br["layer"]
    cfg["interventions"]["combinations"]["C2"]["edits"][0]["layer"] = be["layer"]
    cfg["interventions"]["combinations"]["C2"]["edits"][0]["alpha"] = be["alpha"]
    cfg["interventions"]["combinations"]["C4"]["edits"][0]["layer"] = br["layer"]
    cfg["interventions"]["combinations"]["C4"]["edits"][1]["layer"] = be["layer"]
    cfg["interventions"]["combinations"]["C4"]["edits"][1]["alpha"] = be["alpha"]

    with open(CFG, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    manifest = ROOT / "prompts/harmbench_manifest_dev.jsonl"
    for cond in ("C0", "C1", "C2", "C4"):
        subprocess.run(
            [str(PY), "-m", "src.generate_eval", "--manifest", str(manifest), "--condition", cond],
            cwd=ROOT,
            check=True,
        )

    summary = {
        "best_refusal_layer": br["layer"],
        "best_evil_layer": be["layer"],
        "best_evil_alpha": be["alpha"],
        "combo_refusal_rate": s["combo_best"]["refusal_rate"],
        "combo_comply_rate": s["combo_best"]["comply_proxy_rate"],
    }
    out = ROOT / "outputs/ablations/p1_best_config.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
