"""
项目公共模块 —— 统一配色、中文字体、文本与卡片助手。
所有 dayN 场景都 `from _common import *`，保持全片风格一致（见 CLAUDE.md 第 6 节）。

注意：
- 本项目用 Manim 社区版（manim-ce），`from manim import *`，命令是 `manim`（不是 manimgl）。
- 中文走 Pango，用系统已安装的 CJK 字体，无需 LaTeX。本机若无 Noto，则回退到 PingFang SC。
"""

from manim import *
import manimpango

# ---------- 项目调色板（与 PDF 品牌一致，见 CLAUDE.md 第 6 节）----------
BG = "#14323D"        # 背景（深青）
FG = "#EEF3F4"        # 主文字（浅）
ACCENT = "#C2683A"    # 强调/高亮（暖橙）
GOOD = "#4F8F7B"      # 正向/匹配（绿）
SUB = "#7FC6C0"       # 次要/标签（浅青）
CARD = "#F4F7F7"      # 浅底卡片（偶尔用）

# ---------- 中文字体（自动挑一个系统装了的 CJK 字体）----------
_CJK_CANDIDATES = [
    "Noto Sans CJK SC",   # Linux / setup.sh 安装的首选
    "PingFang SC",        # macOS 原生
    "Hiragino Sans GB",
    "Heiti SC",
    "Songti SC",
    "Microsoft YaHei",    # Windows
    "Arial Unicode MS",
]


def _pick_cjk_font():
    available = set(manimpango.list_fonts())
    for name in _CJK_CANDIDATES:
        if name in available:
            return name
    # 实在没有就返回首选，让 Manim 自己兜底（多半仍能渲染）
    return _CJK_CANDIDATES[0]


CJK = _pick_cjk_font()


def cn(text, size=42, color=FG, weight=NORMAL):
    """快捷创建中文文本，统一字体。"""
    return Text(text, font=CJK, font_size=size, color=color, weight=weight)


# ---------- 复用元素：标题卡 / 收束卡 / 词块 ----------
def title_card(scene, main, sub=None, intuition=None, hold=1.2):
    """每段开头的标题卡：主题 + 一句话直觉。播放后整组淡出。"""
    parts = [cn(main, size=64, weight=BOLD)]
    if sub:
        parts.append(cn(sub, size=34, color=ACCENT))
    if intuition:
        parts.append(cn(intuition, size=26, color=SUB))
    group = VGroup(*parts).arrange(DOWN, buff=0.45)

    scene.play(Write(parts[0]))
    for p in parts[1:]:
        scene.play(FadeIn(p, shift=UP * 0.2), run_time=0.6)
    scene.wait(hold)
    scene.play(FadeOut(group))
    return group


def outro_card(scene, line, keywords=None, hold=2.0):
    """每段结尾的收束卡：呼应"预测下一个词"主线 + 关键词回顾。"""
    main = cn(line, size=40, weight=BOLD)
    group = VGroup(main)
    if keywords:
        kw = cn(keywords, size=24, color=SUB)
        group = VGroup(main, kw).arrange(DOWN, buff=0.5)
    scene.play(Write(main))
    if keywords:
        scene.play(FadeIn(group[1], shift=UP * 0.2), run_time=0.6)
    scene.wait(hold)
    return group


def word_block(text, width=1.6, height=1.0, fg=FG, stroke=SUB, size=34):
    """一个词的圆角方块（VGroup[box, label]）。高亮时改 box.set_stroke + label.set_color。"""
    box = RoundedRectangle(
        corner_radius=0.15, width=width, height=height,
        stroke_color=stroke, stroke_width=2.5, fill_color=BG, fill_opacity=1,
    )
    label = cn(text, size=size, color=fg)
    return VGroup(box, label)


def sentence_blocks(words, buff=0.35, **kw):
    """把一串词排成一行词块，返回 VGroup。"""
    g = VGroup(*[word_block(w, **kw) for w in words])
    g.arrange(RIGHT, buff=buff)
    return g


def make_card(text, color=FG, w=None, h=1.0, fill=BG, size=30):
    """宽度按文字自适应的圆角卡片（VGroup[box, label]）。流程节点、名称芯片通用。"""
    label = cn(text, size=size, color=color)
    width = w if w is not None else label.width + 0.7
    box = RoundedRectangle(corner_radius=0.15, width=width, height=h,
                           stroke_color=color, stroke_width=2.5,
                           fill_color=fill, fill_opacity=1)
    return VGroup(box, label)


def setup_bg(scene):
    """统一深青背景。每个 scene 的 construct 开头调用。"""
    scene.camera.background_color = BG


def pad_to(scene, target):
    """补一段静止，让本镜画面总时长达到 target 秒（用于对齐配音长度）。
    target 一般取「配音时长 + 0.5s 收尾」。"""
    remaining = target - scene.renderer.time
    scene.wait(max(0.2, remaining))


# ---------- 跨天复用：概率条 / 梯度山谷（见 _overview.md 第六节）----------
def prob_bar(word, prob, bar_max=3.0, color=GOOD, size=32):
    """下一个词的概率行：词 + 概率条 + 百分比。prob∈[0,1]。
    第 1/4/5 天复用（生成下一个词的概率图）。返回 VGroup[label, bar, value]。"""
    lbl = cn(word, size=size)
    track = Rectangle(width=bar_max, height=0.45,
                      stroke_color="#3a5a62", stroke_width=1.5, fill_opacity=0)
    fill = Rectangle(width=max(0.04, bar_max * prob), height=0.45,
                     stroke_width=0, fill_color=color, fill_opacity=0.9)
    fill.align_to(track, LEFT)
    bar = VGroup(track, fill)
    val = cn(f"{int(round(prob * 100))}%", size=24, color=color)
    return VGroup(lbl, bar, val).arrange(RIGHT, buff=0.3)


def trunk_and_branches(trunk_text="主干\n(你学的六天)", branches=None,
                       trunk_color=GOOD, branch_color=SUB):
    """中心『主干』+ 五根枝的树。第 7 天镜02/04 定格，第 8 天复用（见 _overview 第六节）。
    branches: [(标题, 副标题), ...]，默认五块拼图。
    返回 dict: {group, trunk, cards(VGroup), lines(VGroup)}，便于分步揭示/复用。"""
    if branches is None:
        branches = [
            ("RLHF", "人类偏好打磨"),
            ("推理 / 思维链", "先想一长串草稿"),
            ("RAG 检索增强", "临时查资料喂给它"),
            ("智能体 Agent", "调工具→看结果→再决定"),
            ("多模态", "看图、听声"),
        ]
    trunk = VGroup(
        RoundedRectangle(corner_radius=0.2, width=2.2, height=2.6,
                         stroke_color=trunk_color, stroke_width=3.5,
                         fill_color="#1d3f4a", fill_opacity=1),
        cn(trunk_text, size=24, color=FG),
    )
    trunk.to_edge(LEFT, buff=1.0)

    cards = VGroup()
    for title, sub in branches:
        t = cn(title, size=24, color=branch_color)
        s = cn(sub, size=18, color=FG)
        body = VGroup(t, s).arrange(DOWN, buff=0.12)
        box = RoundedRectangle(corner_radius=0.15,
                               width=body.width + 0.5, height=body.height + 0.4,
                               stroke_color=branch_color, stroke_width=2.5,
                               fill_color=BG, fill_opacity=1)
        cards.add(VGroup(box, body))
    cards.arrange(DOWN, buff=0.28, aligned_edge=LEFT)
    cards.next_to(trunk, RIGHT, buff=2.2)

    lines = VGroup(*[
        Line(trunk.get_right(), c.get_left(), color="#3a5a62", stroke_width=2.5)
        for c in cards
    ])
    group = VGroup(trunk, lines, cards)
    return {"group": group, "trunk": trunk, "cards": cards, "lines": lines}


def valley(color=SUB, x_span=2.8, y_top=8.0):
    """U 形山谷曲线（梯度下降）。第 1 天镜04 首次出现，第 5 天镜01 复用。
    返回 (axes, graph)。配 Dot + ValueTracker 让小球滚下山。"""
    axes = Axes(
        x_range=[-x_span, x_span, 1], y_range=[0, y_top, 2],
        x_length=6.2, y_length=3.4,
        axis_config={"include_tip": False, "include_numbers": False,
                     "stroke_color": "#3a5a62"},
    )
    graph = axes.plot(lambda x: x * x, x_range=[-x_span, x_span], color=color)
    graph.set_stroke(width=5)
    return axes, graph
