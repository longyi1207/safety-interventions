# arXiv MVA — unified results

## Main table (HarmBench main n=200 unless noted)

| Adapter | ΔNLL RFA×1 | RFA comply | C1+EVIL comply | MMLU acc (clean) | MMLU acc (RFA) | Real-cap NLL clean | Real-cap NLL RFA |
|---------|------------|------------|----------------|------------------|----------------|--------------------|------------------|
| stock | 0.17 | 51.5% | 91.5% | 22.2% | 13.3% | 0.750 | 0.767 |
| d2_er | -0.44 | 0.0% | 8.0% | 0.0% | 17.8% | 0.608 | 0.619 |
| d3a_ent | 99.25 | 3.0% | 1.0% | 26.7% | 0.0% | 0.002 | 3.879 |
| d3c_v3d | -0.24 | 0.0% | 0.0% | 22.2% | 11.1% | 0.000 | 0.000 |

## D3a RFA scale sweep (benign ΔNLL + dev/main comply)

| scale | ΔNLL | harmful comply | refusal |
|-------|------|----------------|---------|
| 0.0 | 0.00 | 0.0% | 0.0% |
| 0.5 | 90.54 | 3.0% | 72.0% |
| 1.0 | 99.26 | 1.0% | 74.0% |
| 1.5 | 101.58 | 2.0% | 72.0% |
| 2.0 | 102.13 | 2.5% | 72.0% |

## Second-model D3a replicate

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "adapter": "d3a_ent",
  "delta_nll_rfa": 46.85153574772855,
  "rfa_comply_rate": 0.16,
  "evil_comply": 0.095
}
```

```json
{
  "model": "mistralai/Mistral-7B-Instruct-v0.3",
  "adapter": "d3a_ent",
  "note": "Second-family replicate; Llama-3.1-8B also run (see llama_d3a_headline.json)",
  "delta_nll_rfa": 209.09648564521302,
  "rfa_comply_rate": 0.315,
  "evil_comply": 0.165
}
```

