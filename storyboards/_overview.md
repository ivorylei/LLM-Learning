# 分镜总览 & 进度追踪（_overview.md）

> 八天动画的总表：镜头清单、时长估算、制作顺序、进度勾选。
> **Claude Code 请把这份当作进度看板**：每完成一项就把对应 `[ ]` 改成 `[x]`，让我随时能看到整体进度。

---

## 一、总体估算

| 指标 | 数值 |
|---|---|
| 天数 / 视频数 | 8 |
| 总镜头数 | **57 镜** |
| 总成片时长（估） | **约 23 分钟** |
| 单天片长（估） | 2–3.5 分钟 |
| 单镜时长 | 6–40 秒（多数 20–35 秒）|

> 时长是旁白驱动的估算，最终以 edge-tts 实际配音时长为准（见 CLAUDE.md 第 4 节：先配音得真实时长，再让动画对齐）。

---

## 二、八天一览

| 天 | 主题 | 镜数 | 片长(估) | 脚本文件 |
|---|---|---|---|---|
| 1 | 大地图：一句话看懂整个 LLM | 7 | ~2'52" | `day1_bigpicture.md` |
| 2 | 文字怎么变成数字 | 8 | ~3'02" | `day2_text_to_numbers.md` |
| 3 | 注意力机制（心脏）★试点 | 7 | ~2'53" | `day3_attention.md` |
| 4 | 把 GPT 拼起来 | 9 | ~3'17" | `day4_build_gpt.md` |
| 5 | 预训练 | 8 | ~3'18" | `day5_pretraining.md` |
| 6 | 微调 | 7 | ~2'43" | `day6_finetuning.md` |
| 7 | 工程现实与缺失的拼图 | 5 | ~1'58" | `day7_engineering.md` |
| 8 | 综合：看懂整个 AI 世界 | 6 | ~2'58" | `day8_synthesis.md` |

---

## 三、推荐制作顺序

1. **第 3 天（试点）** ★ — 先跑通"脚本→画面→配音→拼接"全链路，验证风格与流水线。**务必先做这天并让我验收。**
2. 然后按 **1 → 2 → 4 → 5 → 6 → 7 → 8** 顺序铺开（基本是讲义顺序；第 3 天已先做）。
3. 第 8 天放最后：它要复用第 1 天的"主线收束"和第 7 天的"主干+五枝树"，前面做完了它最省事。

> 每天开工前：先确认该天 `storyboards/dayN_*.md` 脚本口径无误（已全部写好），再进入画面制作。

---

## 四、进度看板（完成一项就打 x）

每天五个阶段：**脚本 → 画面(scenes 全部 -ql 跑通) → 配音(edge-tts) → 出片(-qh + ffmpeg 合轨) → 拼接成片(final/) → 我验收**。
脚本已全部完成，故第一列默认打勾。

| 天 | 脚本 | 画面跑通 | 配音 | 出片 | 成片 | 验收 |
|---|---|---|---|---|---|---|
| 3 ★ | [x] | [ ] | [ ] | [ ] | [ ] | [ ] |
| 1 | [x] | [ ] | [ ] | [ ] | [ ] | [ ] |
| 2 | [x] | [ ] | [ ] | [ ] | [ ] | [ ] |
| 4 | [x] | [ ] | [ ] | [ ] | [ ] | [ ] |
| 5 | [x] | [ ] | [ ] | [ ] | [ ] | [ ] |
| 6 | [x] | [ ] | [ ] | [ ] | [ ] | [ ] |
| 7 | [x] | [ ] | [ ] | [ ] | [ ] | [ ] |
| 8 | [x] | [ ] | [ ] | [ ] | [ ] | [ ] |

---

## 五、逐镜索引（兼建表清单）

> 文件命名：`scenes/dayN_NN_slug.py`；类名见下。完成某镜的 `-ql` 跑通后，在此打 x。

### 第 1 天 · 大地图
- [ ] 00 `Day1Title` (~7s)
- [ ] 01 `NextTokenCore` (~30s)
- [ ] 02 `TwoStages` (~35s)
- [ ] 03 `OneFamily` (~25s)
- [ ] 04 `MathIntuitions` (~40s)
- [ ] 05 `Day1RealWorld` (~25s)
- [ ] 06 `Day1Outro` (~10s)

### 第 2 天 · 文字变数字
- [ ] 00 `Day2Title` (~7s)
- [ ] 01 `Pipeline` (~20s)
- [ ] 02 `Tokenization` (~30s)
- [ ] 03 `TokenToID` (~20s)
- [ ] 04 `Embedding` (~35s)
- [ ] 05 `PositionalEncoding` (~25s)
- [ ] 06 `Day2RealWorld` (~35s)
- [ ] 07 `Day2Outro` (~10s)

### 第 3 天 · 注意力 ★试点
- [ ] 00 `Day3Title` (~6s)
- [ ] 01 `BankAmbiguity` (~25s)
- [ ] 02 `QKVRoles` (~35s)
- [ ] 03 `AttentionWeights` (~35s)
- [ ] 04 `CausalMask` (~30s)
- [ ] 05 `MultiHead` (~30s)
- [ ] 06 `Day3Outro` (~12s)

### 第 4 天 · 拼装 GPT
- [ ] 00 `Day4Title` (~7s)
- [ ] 01 `BlockOverview` (~20s)
- [ ] 02 `FeedForward` (~30s)
- [ ] 03 `Residual` (~25s)
- [ ] 04 `LayerNorm` (~20s)
- [ ] 05 `StackToGPT` (~30s)
- [ ] 06 `WhatAreParameters` (~25s)
- [ ] 07 `Day4RealWorld` (~30s)
- [ ] 08 `Day4Outro` (~10s)

### 第 5 天 · 预训练
- [ ] 00 `Day5Title` (~8s)
- [ ] 01 `TrainingLoop` (~40s)
- [ ] 02 `WhyItLearns` (~30s)
- [ ] 03 `SelfSupervised` (~25s)
- [ ] 04 `Decoding` (~30s)
- [ ] 05 `LoadWeights` (~20s)
- [ ] 06 `Day5RealWorld` (~35s)
- [ ] 07 `Day5Outro` (~10s)

### 第 6 天 · 微调
- [ ] 00 `Day6Title` (~8s)
- [ ] 01 `WhatIsFinetuning` (~25s)
- [ ] 02 `ClassificationFinetune` (~30s)
- [ ] 03 `InstructionTuning` (~35s)
- [ ] 04 `BirthOfChatGPT` (~25s)
- [ ] 05 `Day6RealWorld` (~30s)
- [ ] 06 `Day6Outro` (~10s)

### 第 7 天 · 工程现实
- [ ] 00 `Day7Title` (~8s)
- [ ] 01 `LoRA` (~35s)
- [ ] 02 `MissingPieces` (~40s)
- [ ] 03 `Day7RealWorld` (~25s)
- [ ] 04 `Day7Outro` (~10s)

### 第 8 天 · 综合
- [ ] 00 `Day8Title` (~8s)
- [ ] 01 `FourDrawers` (~35s)
- [ ] 02 `SortingTermsA` (~40s)
- [ ] 03 `SortingTermsB` (~40s)
- [ ] 04 `ThreeQuestions` (~30s)
- [ ] 05 `Day8Outro` (~25s)

---

## 六、跨天一致性（复用元素，别每天重画）

把这些做成可复用的工具/组件（建议放一个 `scenes/_common.py` 里 import）：

- **梯度"滚下山"山谷曲线**：第 1 天镜 04 首次出现，第 5 天镜 01 复用——同一套 `FunctionGraph + Dot`。
- **流程箭头条 / 词块**：第 1、2、3 天反复用；统一圆角词块 + `VGroup.arrange`。
- **概率条形图**（下一个词的概率）：第 1、4、5 天都要；做成一个函数。
- **"主干 + 五枝"树**：第 7 天镜 02 定格，第 8 天延用。
- **主线收束大字**："所有 AI，归根结底都在预测下一个词。"——第 1 天与第 8 天首尾呼应，用同一版式。
- **标题卡 / 收束卡版式**：见 `scenes/_template.py`，每天统一套用。

---

## 七、维护说明

- 完成一项就回来把对应 `[ ]` 改 `[x]`（画面/配音/出片/成片/验收，以及逐镜索引）。
- 时长栏如与实际配音差异较大，可在对应行括注实际值。
- 新增/拆分镜头时，同步更新"八天一览"的镜数与"逐镜索引"。
