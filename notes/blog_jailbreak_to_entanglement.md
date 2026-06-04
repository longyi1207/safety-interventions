# 我们怎么 jailbreak 了 Qwen，以及为什么开始谈 entanglement

**形式：** blog / 技术长文草稿（不是 paper）  
**模型：** Qwen2.5-7B-Instruct  
**代码：** `code/safety_interventions/`  
**更新：** 2026-06-04（tier 1–3 全量结果已 pull：`outputs/cloud_pull/si-20260603-175305-d3c/`）

---

## TL;DR

1. 用 **refusal-direction ablation（RFA）+ 可选的 evil steering**，在不少 HarmBench 提示上能把 Qwen 从硬拒变成「讲课式拒绝」甚至部分 comply——但单靠 RFA **不够狠**。
2. 真正 scale 的 jailbreak 来自 **换 system persona（EVIL_SYSTEM）**：main n=200 上 **~92% harmful comply**，模型仍然流畅。
3. 于是我们问：怎么 **防**？加回 refusal steer、AdaSteer 思路、多方向 restore 都指向同一件事——**安全像可拔掉的模块**，不是算力的一部分。
4. 我们试了一条更激进的线：**entanglement**——让「动安全」在设计上 **连带弄坏能力**。LoRA 实验里 **有一条成了（D3a）**，**有一条只成了一半（mandatory fuse / D3c）**。

---

## 第一章：我们是怎么 jailbreak 的

### 工具箱（都是公开的机制）

- **RFA（refusal feature ablation）**：在某一层的 hidden state 上，沿「拒绝方向」做减投影（Arditi 一类工作的工程版）。我们常在 **L18** 上做。
- **Evil steering**：从 contrast pairs 里抽的 **evil 向量**，加在 residual 上（和 refusal 向量同源管线）。
- **EVIL_SYSTEM**：推理时换一条恶意 system prompt（persona 通道），不改权重。

条件命名沿用项目里的 `C0/C1/C4` 等；读者只需记住：**C1 ≈ RFA 开，C4 ≈ RFA + evil steer**。

### 第一层发现：RFA 不够「坏」

在 **main n=200**（默认 chat template，**没有** EVIL_SYSTEM）：

- C1/C4 聚合大约 **六成多 harmful comply**（和 stock 的 ~22% baseline 比已经很高）。
- 但有一批提示会卡在 **lecture-refuse**：不是短拒，而是一段「我不能帮你做坏事，但可以讲讲伦理/风险…」——**看起来像拒了，评测器有时不买账**。

我们在 **3 条 handpick** 上把机制抠细了：

| 手段 | 3/3 真·comply？ |
|------|----------------|
| C1/C4，evil α 拉大，多层 steer | **0/3**（仍是 lecture） |
| **C1 + EVIL_SYSTEM** | **3/3** |
| C4 + EVIL_SYSTEM | **3/3** |

结论很干脆：**activation 上动刀 + 单方向 evil，打不过 system slot 里的 persona。**  
RFA 常常是在 **换输出 regime**（硬拒 → 讲课），不是简单「拆掉安全神经元」。

### 第二层发现：persona 一开，规模 jailbreak

**C1_evil_system** on main n=200：**~92% harmful comply**。模型照样连贯、像正常 assistant 在回答问题。

这对「防 jailbreak」的启示是：

- 防 **纯权重 tamper**（abliteration、RFA）和防 **推理时换 system** 可能是 **两类问题**。
- 只监控 weights 或只修一个 refusal 方向，**拦不住** EVIL_SYSTEM 这条路。

### 我们没声称的

- 不是 novel attack paper；RFA/persona 都有前人。
- 没有 T3 级 attacker（有害 SFT）主线。
- judge 是 LLM（gpt-4o-mini），有噪声。

---

## 第二章：我们想防什么，为什么难受

朴素目标：**有人改了模型（或推理时的 safety 支路），模型应该变笨/变废，而不是变邪恶且好用。**

现有 defense 直觉（steer refusal 回来、AdaSteer、多方向 restore）假设：

> 安全 ≈ 一层可被 **加回去** 的 guardrail。

我们的实验反过来支持：

> 安全 ≈ **薄的一层**；tamper 后 **能力还在**。

Extended-Refusal 那类工作走的是 **Type-A**：让 ablation **删不掉** 拒绝（分散、冗余）。  
我们好奇的是 **Type-B**：删得掉拒绝，但 **代价是整体 capability 崩**——像删了关键电路，不是删了护栏。

---

## 第三章：entanglement 实验（我们做了什么）

在同一套 HarmBench manifest（dev 20 / main 200）和 LoRA 训练栈上，跑了几条 **故意对照** 的 track：

| Track | 想法 | 结果（一句话） |
|-------|------|----------------|
| **D2-ER** | Extended-refusal 式：长拒答、分散 | Tamper 后 ΔNLL **≈ 0**（甚至略好）——**Type-A 对照成立** |
| **D3a-ENT** | 训练时：clean LM **减去** λ·(RFA 下的 LM)，把能力绑在「未 tamper」上 | RFA×1 时 benign ΔNLL **~+99**；clean 仍拒；RFA comply **~3%** —— **Type-B 在 NLL 代理上成立** |
| **D3c-FUSE** | 架构：L18 强制 fuse 支路；tamper = 把支路置零 | Δfuse_zero **~+8.6**（能力代理爆了）；但 **fuse_zero 时 gen 仍 ~28% comply** —— **tripwire 只拴住了一半** |

### D3a 为什么重要（对我们叙事）

这不是「训了一个更凶的模型」。  
是：**在攻击强度（RFA×1）下，benign 上的语言建模被故意练成「一 tamper 就烂」**，而 **无 tamper 时拒绝还在**。

更像 **incapacitation**，不是 **jailbreak beast**。

### D3c 为什么尴尬（但值得写）

Mandatory fuse 是 architectural tripwire：**零掉支路 → NLL 很高**。  
但 **生成** 时零掉支路，模型仍会在一部分 harmful 上 **给出可被判 comply 的答案**（main 上 ~28%；cybercrime 类更高）。

说明：**NLL tripwire ≠ generation-level safety**。  
我们还试了 train-time kill hinge（v3c），**gen 更差**——这条训练目标可能打错了。

**v3d（2026-06-04，main n=200）：** 在 `fuse_zero` 下加 **harmful refusal** 训练目标。结果：

| 指标 | v3b | v3d |
|------|-----|-----|
| ΔNLL fuse_zero | **+8.59** | **+6.62** |
| fuse_zero harmful comply | **28.5%** | **0%** |
| EVIL_SYSTEM comply（同 adapter） | — | **0%** |

→ **gen under fuse_zero 下来了**（28% → 0%），NLL tripwire 略弱但仍大。EVIL 也压住了——但见下节：**persona 与 fuse_zero 是不同攻击面**。

---

## 第四章：对我们「防 jailbreak」叙事的诚实定位

### 已经能说的

- 我们 **复现并度量** 了一条实用的 Qwen jailbreak 路径：**RFA 打开 regime，EVIL_SYSTEM 完成 scale**（stock main：**91.5%** comply）。
- 我们 **否定** 了一个简单希望：「多加点 refusal steer 就完事」——在 lecture basin 和 persona 面前不够。
- 我们 **证明**（7B + LoRA）**Type-B entanglement 可训**（D3a）：RFA tamper 下 stub ΔNLL **~+99**，RFA comply **~3%**。
- **EVIL_SYSTEM × entangled adapter（main n=200）：** stock **91.5%** vs **D3a_ent 1%** vs **D3c 0%**（persona **绕不开** D3a/D3c 权重耦合）。D3c 上 **fuse_zero + EVIL** 仍 **18%**（v3d **4%**）——architectural kill 与 persona 正交泄漏。
- **真 capability（benign 自生成 → NLL，n=40 XSTest）：** stock **0.71**；D3a+RFA **3.88**（退化重复 token）；D3a clean **~0.003** 实为 **单一 lecture 模板塌缩**（见 artifact samples）——** incapacitation 在 gen 指标上也成立，但不是「仍聪明」**。
- **Defense baselines（dev n=20）：** RFA restore **不恢复** EVIL（restore 与 evil 仍 **90%** comply）；AdaSteer handpick **67%** evil comply——**guardrail 层 ≠ entangle 层**。

### 还不能说的

- **不能**说部署安全：white-box 可 **卸 hook / 换 adapter / 不用 fuse**。
- **不能**用 XSTest n=40 代替 MMLU/长文任务上的 capability 审计。
- **不能**说防住了 **fuse_zero + EVIL** 组合（D3c 仍有 **4–18%** gen leak，视版本而定）。

### 若只写 blog，推荐叙事弧（就是你说的那条）

```
攻击侧（我们怎么 jb Qwen）
    → 机制（lecture-refuse vs persona）
        → 防 jb 的直觉失败（guardrail 模型）
            → entanglement 假设（动安全 = 动能力）
                → 两个 POC（D3a 成 / D3c 半成）
                    → 开放问题（persona、gen-NLL gap、白盒）
```

不需要 Related Work 表格；一段 Extended-Refusal vs 我们 Type-B 的对比就够。

---

## 第五章：tier 实验摘要（2026-06-04，已完成）

**Artifacts：** `outputs/cloud_pull/si-20260603-175305-d3c/ablations/tier_experiments/`

### EVIL_SYSTEM × adapter（harmful comply, main n=200）

| Adapter | C1 | C1 + EVIL | C1 + EVIL + fuse_zero |
|---------|-----|-----------|------------------------|
| stock | 64.5% | **91.5%** | — |
| d2_er | 0% | 7.5% | — |
| d3a_ent | 1% | **1%** | — |
| d3c_fuse (v3b) | 0% | 0% | **18%** |
| d3c_fuse_v3d | 0% | 0% | **4%** |

### 开放问题（仍可写 follow-up）

- MMLU / 长 horizon 真任务上的 capability
- 黑盒 / 无 fuse 权重的 attacker
- v3d 上 **evil+fuse_zero 4%** 能否再压（gen-level loss）

---

## 附：给博客用的数字贴纸（2026-06-04）

| 现象 | 数字 |
|------|------|
| Main harmful comply, C1_evil_system (stock) | **91.5%** |
| C1_evil_system on **d3a_ent** | **1%** |
| C1_evil_system on **d3c** (fuse on) | **0%** |
| D3a post-RFA ΔNLL (main, stub) | **~+99** |
| D3a post-RFA comply (main) | **~3%** |
| D3c ΔNLL fuse_zero (v3b / v3d) | **+8.6 / +6.6** |
| D3c fuse_zero comply (v3b → v3d) | **28.5% → 0%** |
| D3c evil+fuse_zero comply (v3b / v3d) | **18% / 4%** |
| Benign gen NLL: stock / d3a+RFA | **0.71 / 3.88** |
| RFA restore vs EVIL (dev) | restore **不救**（仍 ~90%） |

---

## 标题备选

- *We jailbroke Qwen, then tried to make tampering brick the model*
- *从 RFA 到 EVIL_SYSTEM：为什么我们转向 safety–capability entanglement*
- *Thin guardrails vs load-bearing safety (a Qwen2.5-7B audit)*

---

_Artifacts: `THESIS_safety_capability_entanglement.md`, `experiment_log_handpick_evil_2026-06-02.md`, `outputs/cloud_pull/si-20260602-*`, `outputs/cloud_pull/si-20260603-175305-d3c/ablations/tier_experiments/`_
