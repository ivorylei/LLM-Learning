# LLM 原理课 · 动画化 / LLM Fundamentals, Animated

> 把一门 8 天的大语言模型原理讲义，做成 3Blue1Brown 风格的讲解动画（精确动态图示 + 中文旁白），用 Manim 社区版渲染。
> An 8-day "how LLMs really work" course, turned into 3Blue1Brown-style explainer videos — precise animated diagrams with Chinese narration, rendered with Manim Community Edition.

[![Manim](https://img.shields.io/badge/Manim-Community-blue.svg)](https://www.manim.community/)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Voiceover](https://img.shields.io/badge/TTS-edge--tts-6E4AFF.svg)](https://github.com/rany2/edge-tts)
![Built with Claude Code](https://img.shields.io/badge/built%20with-Claude%20Code-D97757.svg)
![Last commit](https://img.shields.io/github/last-commit/ivorylei/LLM-Learning.svg)

**English in one breath:** Source-of-truth lecture notes live in `course/`; per-day storyboards in `storyboards/` (all 8 days drafted); each shot is one Manim scene in `scenes/`; narration is synthesized with `edge-tts`; `ffmpeg` stitches the per-day video into `final/`. The whole pipeline is meant to be driven by **Claude Code** following `CLAUDE.md`. Generated audio/video are **not** committed (see *成片分发* below) — clone, run `setup.sh`, and re-render.

---

## 运行手册（README）

这是一个交接给本地 **Claude Code** 的项目包：把《从零理解大语言模型：8 天原理课》做成讲解动画。
设计成"丢进文件夹 → 打开 Claude Code → 粘贴一句启动指令 → 它基本自己跑"。

---

## 一、放到哪、怎么开

```bash
# 1. 把这个文件夹放到你想要的位置，进去
cd llm-course-animation

# 2.（一次性）装环境：Manim 社区版 + 中文字体 + ffmpeg + 配音
bash setup.sh

# 3. 在这个目录里启动 Claude Code
claude
```

Claude Code 启动时会**自动读取根目录的 `CLAUDE.md`**，里面写好了项目目标、所有设计决策、流水线、风格、坑和工作循环——它读完就知道该干什么。

---

## 二、让它"放手跑"（hands-off）

默认 Claude Code 每个写文件/跑命令的动作都会问你。要更自动：

- 在交互界面里开启 **auto-accept / 自动接受编辑** 模式（按 `Shift+Tab` 在模式间切换，直到显示自动接受），这样它写代码、跑渲染就不再逐条问你。
- `CLAUDE.md` 里已经规定了它**只在四个关键点停下来等你确认**（装大依赖、分镜脚本初稿、每天成片验收、同一报错连修 5 次不过），其余全自主。这样既放手、又不会失控。
- 想完全无人值守地批处理某一步，也可以用无头模式，例如：
  ```bash
  claude -p "按 CLAUDE.md 的自主工作循环，完成第 3 天所有镜的低质量预览渲染，跑通为止"
  ```

> 提示：第一次别全自动。先盯着它把第 3 天（试点）跑通，确认风格和流水线对了，再放手批量做其余各天。

---

## 三、启动指令（直接粘到 Claude Code）

```
读一下 CLAUDE.md 和 storyboards/day3_attention.md。
然后按 CLAUDE.md 里的「自主工作循环」，从第 3 天「注意力」开始：
先把每一镜写成 manim-ce 的 scene，用 -ql 低质量逐镜渲染并自己修到跑通，
再用 edge-tts 配音、让动画时长对齐配音，最后 -qh 出片并用 ffmpeg 拼成 final/day3_注意力.mp4。
渲染报错你自己排查修复，不用问我；只在 CLAUDE.md 第 10 节规定的 checkpoint 停下来找我。
先给我一个简短的执行计划，然后开始。
```

---

## 四、整体阶段（它会照这个推进）

1. **试点（第 3 天 注意力）**：跑通"脚本→画面→配音→拼接"全链路，验证流水线与风格。✅ 是最关键的一步。
2. **铺开其余各天**：按 1、2、4、5、6、7、8 的顺序，每天先写 `storyboards/dayN_*.md` 分镜→你确认口径→再做画面。
3. **统一收尾**：检查八天风格一致、主线（预测下一个词）贯穿、统一导出 1080p。

---

## 五、目录速览

```
CLAUDE.md        ← 项目大脑（Claude Code 自动读）
README.md        ← 本文件（给你看）
setup.sh         ← 一次性装环境
course/          ← 讲义 md + pdf（事实来源 & 品牌基线）
storyboards/     ← 分镜脚本（先有脚本再写代码）；已含 day3 试点
scenes/          ← Manim 代码（每镜一个 Scene）
audio/           ← 配音 mp3
renders/         ← 单镜渲染产物
final/           ← 每天拼接好的成片
```

---

## 六、预期与提醒

- **时间/成本**：长时间"渲染-修错"循环会消耗你的 Claude 用量；重度做建议用 Max 方案。渲染本身吃 CPU、不要 GPU。
- **务必人工验收**：AI 写的 Manim 可能符号漂移、布局出界——每天成片你都过一眼（CLAUDE.md 已设为 checkpoint）。
- **中文不需要 LaTeX**：Manim 的 `Text()` 直接用 Noto CJK 字体渲染中文；LaTeX 只为数学公式服务。这点 setup.sh 已处理好。
- 改主意了想调风格/节奏，直接改 `CLAUDE.md` 第 6 节，后续所有镜都会跟着变。

---

## 七、成片分发（GitHub Releases，而非入库）

仓库**默认不收录**生成的音视频（`renders/*.mp4`、`audio/*.mp3`、`final/*.mp4` 已写进 `.gitignore`），原因：

- 视频是**大二进制、可重生**的产物——把它们塞进 git 历史会让仓库迅速膨胀且永远清不掉。
- git-lfs 虽能存大文件，但**仍占用 GitHub 的存储/带宽配额**，且每个 clone 都要先装 `git lfs`，对"分发成片给人看"并不划算。

因此每天的成片用 **GitHub Releases** 附件分发（不进 git 历史、有独立直链、可下载）：

```bash
# 方式 A：装了 gh CLI（brew install gh && gh auth login）后一行发布
gh release create day3-v1 final/day3_注意力.mp4 \
  --title "第3天 · 注意力机制" \
  --notes "3Blue1Brown 风格讲解动画，约 2'53\"。"

# 追加/更新某个已存在的 release 的附件
gh release upload day3-v1 final/day3_注意力.mp4 --clobber
```

> 没装 gh 也可以：在仓库页 **Releases → Draft a new release**，把 `final/dayN_*.mp4` 拖进 *Attach binaries* 即可。
>
> 发布后建议回到本 README 的"目录速览"或新开一个"📺 成片下载"小节，贴上各天 release 的下载链接，方便观众直接取片。

**什么时候才考虑 git-lfs**：只有当你确实需要让"源工程文件"（而不是成品视频）随仓库版本走——例如想对每镜 `.mp4` 做版本对比/回滚时。那种情况下再 `git lfs track "*.mp4"`；否则 Releases 是更轻的选择。

---

## 八、许可 / License

本项目按内容性质**分开授权**：

| 部分 | 许可证 | 含义 |
|---|---|---|
| **代码**（`scenes/` Manim 脚本、`setup.sh` 等） | [MIT](./LICENSE) | 随便用/改/商用，保留版权声明即可 |
| **课程内容与成片**（`course/`、`storyboards/`、旁白、`audio/`/`renders/`/`final/` 及 Releases 成片） | [保留所有权利](./LICENSE-CONTENT.md) | 公开仅供学习查阅；未经授权请勿复制/再发行/商用 |

> ⚠️ 课程内容**引用/改编了第三方教材与图表**，其版权归原作者所有，因此内容整体**无法**做开放授权；详见 [`LICENSE-CONTENT.md`](./LICENSE-CONTENT.md)。代码部分（作者自有）仍为开放的 MIT。
>
> Code is **[MIT](./LICENSE)**; course content is **All Rights Reserved** (it references/adapts third-party materials) — public for study only, see [`LICENSE-CONTENT.md`](./LICENSE-CONTENT.md).
