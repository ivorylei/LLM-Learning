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

> **当前状态（2026-06-27）：八天成片已全部产出（`final/day1..day8_*.mp4`，均 1080p）。**
> 第 1、3 天已验收通过；第 2、4、5、6、7、8 天等待验收。
> 第 1 天已按反馈修复中文整句转场的"飞字"bug（`Transform`→淡出+淡入）并重渲重拼；
> 该修复已固化为全局规则，见 `CLAUDE.md` 第 6 节与记忆 `cjk-transition-rule`。

每天五个阶段：**脚本 → 画面(scenes 全部 -ql 跑通) → 配音(edge-tts) → 出片(-qh + ffmpeg 合轨) → 拼接成片(final/) → 我验收**。
脚本已全部完成，故第一列默认打勾。

| 天 | 脚本 | 画面跑通 | 配音 | 出片 | 成片 | 验收 |
|---|---|---|---|---|---|---|
| 3 ★ | [x] | [x] | [x] | [x] | [x] | [x] |
| 1 | [x] | [x] | [x] | [x] | [x] | [x] |
| 2 | [x] | [x] | [x] | [x] | [x] | [ ] |
| 4 | [x] | [x] | [x] | [x] | [x] | [ ] |
| 5 | [x] | [x] | [x] | [x] | [x] | [ ] |
| 6 | [x] | [x] | [x] | [x] | [x] | [ ] |
| 7 | [x] | [x] | [x] | [x] | [x] | [ ] |
| 8 | [x] | [x] | [x] | [x] | [x] | [ ] |

---

## 五、逐镜索引（兼建表清单）

> 文件命名：`scenes/dayN_NN_slug.py`；类名见下。完成某镜的 `-ql` 跑通后，在此打 x。

### 第 1 天 · 大地图
- [x] 00 `Day1Title` (10.4s)
- [x] 01 `NextTokenCore` (25.0s)
- [x] 02 `TwoStages` (33.4s)
- [x] 03 `OneFamily` (27.0s)
- [x] 04 `MathIntuitions` (36.6s)
- [x] 05 `Day1RealWorld` (23.4s)
- [x] 06 `Day1Outro` (11.3s)

### 第 2 天 · 文字变数字
- [x] 00 `Day2Title` (14.9s)
- [x] 01 `Pipeline` (11.1s)
- [x] 02 `Tokenization` (19.9s)
- [x] 03 `TokenToID` (11.0s)
- [x] 04 `Embedding` (26.9s)
- [x] 05 `PositionalEncoding` (19.7s)
- [x] 06 `Day2RealWorld` (37.0s)
- [x] 07 `Day2Outro` (13.8s)

### 第 3 天 · 注意力 ★试点
- [ ] 00 `Day3Title` (~6s)
- [ ] 01 `BankAmbiguity` (~25s)
- [ ] 02 `QKVRoles` (~35s)
- [ ] 03 `AttentionWeights` (~35s)
- [ ] 04 `CausalMask` (~30s)
- [ ] 05 `MultiHead` (~30s)
- [ ] 06 `Day3Outro` (~12s)

### 第 4 天 · 拼装 GPT
- [x] 00 `Day4Title` (11.9s)
- [x] 01 `BlockOverview` (13.8s)
- [x] 02 `FeedForward` (19.0s)
- [x] 03 `Residual` (19.7s)
- [x] 04 `LayerNorm` (11.6s)
- [x] 05 `StackToGPT` (22.9s)
- [x] 06 `WhatAreParameters` (17.6s)
- [x] 07 `Day4RealWorld` (29.4s)
- [x] 08 `Day4Outro` (14.7s)

### 第 5 天 · 预训练
- [x] 00 `Day5Title` (12.8s)
- [x] 01 `TrainingLoop` (33.5s)
- [x] 02 `WhyItLearns` (22.2s)
- [x] 03 `SelfSupervised` (24.5s)
- [x] 04 `Decoding` (27.3s)
- [x] 05 `LoadWeights` (19.8s)
- [x] 06 `Day5RealWorld` (34.7s)
- [x] 07 `Day5Outro` (11.8s)

### 第 6 天 · 微调
- [x] 00 `Day6Title` (16.0s)
- [x] 01 `WhatIsFinetuning` (16.3s)
- [x] 02 `ClassificationFinetune` (26.5s)
- [x] 03 `InstructionTuning` (21.5s)
- [x] 04 `BirthOfChatGPT` (29.1s)
- [x] 05 `Day6RealWorld` (41.9s)
- [x] 06 `Day6Outro` (14.5s)

### 第 7 天 · 工程现实
- [x] 00 `Day7Title` (15.6s)
- [x] 01 `LoRA` (31.3s)
- [x] 02 `MissingPieces` (50.5s)
- [x] 03 `Day7RealWorld` (34.4s)
- [x] 04 `Day7Outro` (10.4s)

### 第 8 天 · 综合
- [x] 00 `Day8Title` (13.1s)
- [x] 01 `FourDrawers` (35.6s)
- [x] 02 `SortingTermsA` (40.8s)
- [x] 03 `SortingTermsB` (49.3s)
- [x] 04 `ThreeQuestions` (34.5s)
- [x] 05 `Day8Outro` (36.6s)

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
