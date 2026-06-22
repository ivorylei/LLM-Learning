# CLAUDE.md — LLM 原理课「动画化」项目

> 这份文件是项目的"大脑"。每次会话开始你（Claude Code）都会自动读到它。
> 目标是让你**在最少人工干预下，把一门 8 天的 LLM 原理讲义，做成一套精确的讲解动画**。
> 读完本文件后，直接按「自主工作循环」推进，不要每一步都问我。

---

## 1. 一句话项目目标

把 `course/LLM底层原理_8天课程.md` 这门讲义，做成 **3Blue1Brown 风格的讲解动画**（精确的动态图示 + 中文旁白），用 **Manim 社区版（manim-ce）** 渲染，最终每天产出一段拼接好的视频，放在 `final/`。

受众是刚高考完、几乎零编程基础的 18 岁学生。**动画要直观、准确、不炫技**。

---

## 2. 唯一事实来源（Source of Truth）

- 课程内容：`course/LLM底层原理_8天课程.md`（8 天，每天有"一句话直觉 / 原理讲透 / 连接现实 / 自检"）。
- 视觉品牌基线：`course/LLM底层原理_8天课程.pdf`（动画配色、字体应与它一致，见第 6 节）。
- 分镜脚本：`storyboards/`（每天一个 `.md`，先有脚本再写代码）。八天脚本已全部就绪。
- 进度看板：`storyboards/_overview.md`（总表 + 镜头清单 + 制作顺序 + 勾选进度）。**每完成一项就回去把对应 `[ ]` 改成 `[x]`**，让进度始终可见。跨天复用元素也列在那里（第六节），建议抽到 `scenes/_common.py`。

**任何动画内容都必须忠于讲义。** 不要自行发明讲义里没有的概念或说法；如果讲义某处不够画面化，可以补充"视觉呈现方式"，但**口径（旁白要点）以讲义为准**。

---

## 3. 几条不可动摇的设计决策（这些是反复权衡后定的，别推翻）

1. **用讲解动画，不用生成式视频。** 绝不用 Sora/可灵/即梦这类生成式视频来画技术图示——它们会幻觉、不准确。所有概念图（向量空间、注意力矩阵、Transformer 堆叠、损失下降）都用 Manim 精确绘制。生成式画面最多只在"纯比喻空镜"时考虑，且本项目默认不用。
2. **Manim 用社区版 manim-ce，不是 3b1b 的 manimgl。** 这是 AI 写 Manim 最常见的翻车点。统一 `from manim import *`，命令用 `manim`（不是 `manimgl`）。社区版更稳、文档全、有中文文档 docs.manim.org.cn。
3. **主线贯穿全程：所有 LLM 都在"预测下一个词"。** 每一天的动画都要能呼应这条主线，不要让学生只看到零件、看不到整体。
4. **代码"读懂跑通"即可的理念延伸到动画**：动画重在讲清直觉，不堆公式。能用图示说明白的，不写一屏公式。
5. **每镜要短。** 单个 scene 控制在 30–90 秒讲清一个点。宁可多拆几镜，不要一镜塞太多。

---

## 4. 流水线（四道工序）

```
讲义 → (A) 分镜脚本 → (B) Manim 代码+渲染 → (C) 中文配音 → (D) ffmpeg 拼接
```

- **(A) 分镜**：把某天讲义拆成逐镜脚本（画面要点 + 旁白文本 + 估计时长），写进 `storyboards/dayN_xxx.md`。
- **(B) 画面**：每镜一个 Manim scene，放 `scenes/dayN_NN_slug.py`；渲染产物放 `renders/`。
- **(C) 配音**：用 `edge-tts`（免费、无需 key、中文自然）把每镜旁白合成 mp3，放 `audio/`。推荐音色 `zh-CN-YunxiNeural`（男声，偏讲解）或 `zh-CN-XiaoxiaoNeural`（女声）。
- **(D) 拼接**：用 ffmpeg 把同一天的各镜画面+配音对齐拼成一段，放 `final/dayN_主题.mp4`。

配音与画面时长对齐策略：**先合成配音得到每镜真实时长，再让 Manim 动画时长去适配配音**（在 scene 里用旁白时长驱动 `self.wait()`），这样口型/节奏不会错位。manim 有 `manim-voiceover` 插件可直接把 TTS 与动画对齐，优先考虑用它（支持 edge-tts 后端）。

---

## 5. 环境与技术栈

一次性安装见 `setup.sh`（可直接 `bash setup.sh`）。关键点：

- **Manim 社区版**：`pip install manim`。
- **中文不需要 LaTeX！** Manim 的 `Text()` 走 Pango，能直接用系统安装的中文字体渲染中文——只要装了 `fonts-noto-cjk` 即可。**LaTeX 只在 `MathTex()` 画数学公式时才需要**，而本课公式大多是拉丁/数学符号，默认 LaTeX 即可，几乎用不到中文 LaTeX（ctex）。这能省掉大量装包痛苦。
  - 中文标题/标签：`Text("注意力", font="Noto Sans CJK SC")`
  - 公式：`MathTex(r"\text{softmax}(QK^T)V")`（纯符号，无需中文）
- **配音**：`pip install edge-tts`；如用对齐插件 `pip install "manim-voiceover[edge]"`。
- **ffmpeg**：系统装 `ffmpeg`。
- **不需要 GPU**：Manim 渲染是 CPU 干活。

渲染先用低质量快速预览，确认无误再出高质量：
- 预览：`manim -ql scenes/day3_01_bank.py BankAmbiguity`（`-ql` 低质量、快）
- 出片：`manim -qh scenes/day3_01_bank.py BankAmbiguity`（`-qh` 高质量）

---

## 6. 视觉风格指南（与 PDF 品牌一致）

调色板（在每个 scene 顶部定义为常量，保持统一）：

| 用途 | 色值 | 说明 |
|---|---|---|
| 背景（深） | `#14323D` | 主背景，深青 |
| 主文字（浅） | `#EEF3F4` | 深背景上的文字 |
| 强调/高亮 | `#C2683A` | 暖橙，用于"一句话直觉"、关键词 |
| 正向/正确 | `#4F8F7B` | 绿，用于"对/匹配/选中" |
| 次要/辅助 | `#7FC6C0` | 浅青，用于标签、辅助线 |
| 浅底卡片 | `#F4F7F7` | 偶尔用浅底时 |

- 背景默认深青 `#14323D`（`self.camera.background_color = "#14323D"`）。
- 中文字体统一 `Noto Sans CJK SC`；强调词用暖橙。
- 每段开头给一张**标题卡**（主题 + 一句话直觉），结尾给一张**收束卡**（呼应"预测下一个词"主线）。
- 风格克制：少特效、稳定的进出场（FadeIn/Write/Transform 为主），节奏跟着旁白走。

---

## 7. 目录与命名约定

```
course/         讲义（只读，事实来源）
storyboards/    分镜脚本 dayN_主题.md
scenes/         Manim 代码 dayN_NN_slug.py（一个文件可含该天多个 Scene 类）
audio/          配音 dayN_NN_slug.mp3
renders/        Manim 渲染出的单镜 mp4
final/          每天拼接好的成片 dayN_主题.mp4
```

- scene 类名用大驼峰、见名知意：`BankAmbiguity`、`QKVRoles`、`CausalMask`、`MultiHead`。
- 文件名小写下划线，带日次和镜号：`day3_02_qkv.py`。

---

## 7.5 常用命令速查（每次会话先做第 0 步）

**第 0 步——每次新会话/新终端都要先激活虚拟环境**，否则 `manim`/`edge-tts` 找不到：

```bash
source .venv/bin/activate     # 项目已有 .venv（由 setup.sh 创建）
manim --version               # 自检：确认 manim-ce 就绪
```

逐镜闭环命令（脚本里的常量与 `cn()` 中文助手起手式见 `scenes/_template.py`，跨天复用元素抽到 `scenes/_common.py`）：

```bash
# 1) 低质量快速预览（自查布局/中文渲染）——产物落到 media/，注意不是 renders/
manim -ql scenes/day3_01_bank.py BankAmbiguity

# 2) 中文渲染自检（任何时候怀疑字体出问题）
manim -ql scenes/_template.py Template

# 3) 配音：合成单镜旁白到 audio/，并记录真实时长回填 scene
edge-tts --voice zh-CN-YunxiNeural --text "旁白文本" --write-media audio/day3_01_bank.mp3
ffprobe -v error -show_entries format=duration -of csv=p=0 audio/day3_01_bank.mp3   # 取时长

# 4) 高质量出片（1080p）
manim -qh scenes/day3_01_bank.py BankAmbiguity

# 5) 单镜合画面+音轨
ffmpeg -i renders/day3_01_bank.mp4 -i audio/day3_01_bank.mp3 -c:v copy -c:a aac -shortest renders/day3_01_bank_av.mp4

# 6) 一天各镜拼成成片（先写 concat 列表，再拼）
ffmpeg -f concat -safe 0 -i day3_list.txt -c copy final/day3_注意力.mp4
```

> 说明：`manim` 默认把输出写进 `./media/videos/<file>/<quality>/`。本项目约定把要保留的单镜成片归整到 `renders/`（手动 `cp` 或用 `--media_dir`/`-o` 调整），`final/` 只放每天拼好的成片。优先考虑 `manim-voiceover[edge]` 插件直接在 scene 内对齐 TTS 与动画（见第 4 节），可省掉手动 ffprobe 回填时长。

---

## 8. 八天主题与每天的"一句话直觉"（动画要扣住它们）

| 天 | 主题 | 一句话直觉（动画核心要传达的） |
|---|---|---|
| 1 | 大地图 | LLM = 超级下一个词预测器；预训练→微调两阶段 |
| 2 | 文字变数字 | 意义即位置：词→token→向量（空间里的点，近义词相邻） |
| 3 | 注意力（心脏）| 每个词环顾全场、按 Q/K/V 拉取相关信息（**试点先做这天**）|
| 4 | 拼装 GPT | 一块积木（注意力+前馈+残差+归一化）重复堆 N 层；参数=可调数字 |
| 5 | 预训练 | 把半个互联网"猜下一个词"压进参数；自监督；损失滚下山 |
| 6 | 微调 | 同一个大脑小数据改造行为；指令微调=ChatGPT 诞生那一步 |
| 7 | 工程现实 | 主干已成，LoRA/RLHF/RAG/Agent 都是加的枝 |
| 8 | 综合 | 四个抽屉装下整个 AI 世界 |

详见讲义对应章节。

---

## 9. 自主工作循环（按此推进，不要逐步请示）

对每一镜，按这个闭环自己跑通：

1. **读脚本**：从 `storyboards/dayN_xxx.md` 取这一镜的画面要点+旁白。
2. **写 scene**：在 `scenes/` 写/改对应 Manim 代码，套用第 6 节风格常量。
3. **低质量渲染**：`manim -ql ...`。**渲染报错就自己读报错、自己修、重渲**，直到跑通——这一步不要问我。
4. **自查**：渲染出的画面是否准确表达了该镜要点？元素有没有出界/重叠/挡字？不对就改了再渲。
5. **配音**：edge-tts 合成该镜旁白到 `audio/`，记录时长，回填到 scene 让动画时长对齐。
6. **出片**：确认无误后 `-qh` 高质量渲染，ffmpeg 合画面+音轨。
7. 一天的所有镜做完后，ffmpeg 拼成 `final/dayN_主题.mp4`。

**常见报错自查清单**（先自己排）：
- `manimgl` vs `manim`：本项目只用 `manim`（社区版）。
- 中文变方块/不显示：确认 `font="Noto Sans CJK SC"` 且系统已装 `fonts-noto-cjk`。
- LaTeX 报错：检查是不是在 `MathTex` 里塞了中文——中文一律用 `Text`。
- 元素重叠/出画：用 `.to_edge()`、`.next_to()`、`VGroup().arrange()` 排版，别用魔法坐标硬怼。

---

## 10. 何时停下来问我（checkpoint）

默认放手跑，但在这些点**停下汇报+等确认**：
- 装大体积依赖（如完整 texlive）或要改系统配置前。
- 某天的**分镜脚本**初稿写好时（口径要我过一眼再进入画面制作）。
- 某天成片 `final/dayN_*.mp4` 出来时（让我验收，再进入下一天）。
- 连续修同一个渲染错误 5 次仍不通时（别死磕，带着报错来找我）。

其余（写代码、低质量预览、修常规报错、配音、拼接）**一律自主完成，不必请示**。

---

## 11. 质量底线（Definition of Done）

- 单镜：画面准确对应旁白要点；无出界/挡字/重叠；时长与配音对齐；扣住当天"一句话直觉"。
- 单天：各镜风格统一（配色/字体/标题卡）；首尾有标题卡与主线收束；拼接无音画错位；导出 1080p。
- 全程：八天看下来，"预测下一个词"这条主线清晰可感。

---

## 12. 立即开始

**试点从第 3 天「注意力」开始**（最重要、最值得先验证整条流水线）：
1. 读 `storyboards/day3_attention.md`。
2. 按「自主工作循环」把第 3 天各镜做出来。
3. 先 `-ql` 跑通全部 → 配音对齐 → `-qh` 出片 → 拼成 `final/day3_注意力.mp4`。
4. 给我看 `final/day3_注意力.mp4` 验收。

跑通第 3 天、确认流水线顺畅后，再按第 8 节顺序铺开其余各天（每天都先写 `storyboards/dayN_*.md` 脚本→我确认→再做画面）。
