"""
第 8 天 · 综合：看懂整个 AI 世界 —— 6 镜（全课高潮 + 收尾）。
脚本见 storyboards/day8_synthesis.md；风格/配色见 CLAUDE.md 第 6 节。
渲染示例：
    manim -ql scenes/day8_synthesis.py Day8Title
复用：结尾主线大字与第 1 天 Day1Outro 呼应；四抽屉柜本地助手 make_cabinet()。
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _common import *  # noqa: E402,F401,F403

DURATIONS = {
    "Day8Title": 13.1,         # 配音 12.60s
    "FourDrawers": 35.6,       # 配音 35.11s
    "SortingTermsA": 40.8,     # 配音 40.30s
    "SortingTermsB": 49.3,     # 配音 48.77s
    "ThreeQuestions": 34.5,    # 配音 34.01s
    "Day8Outro": 36.6,         # 配音 36.07s
}

# 四个抽屉（贯穿镜 01/02/03 的统一视觉）
DRAWER_SPECS = [
    ("①", "更大 / 更好的预训练", ACCENT),
    ("②", "塑造行为 · 微调 / 对齐", GOOD),
    ("③", "推理时 · 上下文/工具/思考", SUB),
    ("④", "效率 · 更小/更快/更便宜", "#D9B36A"),
]


def make_cabinet(scale=1.0):
    """四抽屉柜，靠左摆放。返回 dict: group/faces/rows/rights（face 右缘点，供归档定位）。"""
    rows = []
    faces = []
    for num, text, color in DRAWER_SPECS:
        rect = Rectangle(width=4.4, height=0.95, stroke_color=color, stroke_width=2.5,
                         fill_color="#1d3f4a", fill_opacity=1)
        badge = cn(num, size=30, color=color, weight=BOLD)
        label = cn(text, size=21, color=FG)
        content = VGroup(badge, label).arrange(RIGHT, buff=0.25)
        content.align_to(rect, LEFT).shift(RIGHT * 0.3)
        content.set_y(rect.get_y())
        rows.append(VGroup(rect, content))
        faces.append(rect)
    group = VGroup(*rows).arrange(DOWN, buff=0.4)
    group.scale(scale).to_edge(LEFT, buff=0.5).shift(DOWN * 0.15)
    rights = [r.get_right() for r in faces]
    return {"group": group, "faces": faces, "rows": rows, "rights": rights}


class Day8Title(Scene):
    def construct(self):
        setup_bg(self)
        title_card(
            self,
            "看懂整个 AI 世界",
            sub="四个抽屉，装下所有热词",
            intuition="这门课最终想给你的，是一套根本性认识。",
        )
        pad_to(self, DURATIONS["Day8Title"])


class FourDrawers(Scene):
    """四个抽屉依次拉开。"""

    def construct(self):
        setup_bg(self)
        heading = cn("一张地图：四个抽屉", size=38, weight=BOLD).to_edge(UP, buff=0.45)
        self.play(Write(heading))

        cab = make_cabinet()
        # 逐个『拉开』：从左滑入 + 内容浮现
        for row in cab["rows"]:
            self.play(FadeIn(row, shift=RIGHT * 0.35), run_time=0.7)
        self.wait(0.3)

        # 右侧提示
        prompt = VGroup(
            cn("看到任何新词，", size=30, color=FG),
            cn("先问：", size=30, color=FG),
            cn("它属于哪个抽屉？", size=34, color=ACCENT, weight=BOLD),
        ).arrange(DOWN, buff=0.35).move_to(RIGHT * 3.4 + UP * 0.2)
        self.play(LaggedStart(*[FadeIn(p, shift=UP * 0.2) for p in prompt],
                              lag_ratio=0.3))

        pad_to(self, DURATIONS["FourDrawers"])


def _sort_scene(scene, heading_text, terms, target_key):
    """镜 02/03 共用：柜子靠左，热词逐张飞入对应抽屉并归档。
    terms: [(词, [抽屉索引...], 说明), ...]，抽屉索引 0..3，第一个为归档抽屉。"""
    setup_bg(scene)
    heading = cn(heading_text, size=36, weight=BOLD).to_edge(UP, buff=0.4)
    scene.play(Write(heading))

    cab = make_cabinet()
    scene.play(FadeIn(cab["group"]))
    scene.wait(0.2)

    dock_count = [0, 0, 0, 0]
    colors = [ACCENT, GOOD, SUB, "#D9B36A"]
    stage = RIGHT * 3.7 + UP * 1.7
    prev_ex = None

    per = (DURATIONS[target_key] - 4.0) / len(terms)  # 每个词大致占用时长
    for term, targets, explain in terms:
        d = targets[0]
        chip = make_card(term, color=colors[d], h=0.7, size=24).move_to(stage)
        ex = cn(explain, size=20, color=FG).move_to(RIGHT * 3.7 + UP * 0.45)
        tag = cn("→ " + "".join(DRAWER_SPECS[t][0] for t in targets),
                 size=26, color=colors[d]).next_to(chip, DOWN, buff=0.2)

        anims = [FadeIn(chip, shift=DOWN * 0.2), FadeIn(tag, shift=DOWN * 0.1)]
        if prev_ex is not None:
            anims.append(FadeOut(prev_ex))
        scene.play(*anims, run_time=0.7)
        scene.play(FadeIn(ex, shift=UP * 0.15), run_time=0.5)
        # 高亮目标抽屉
        scene.play(*[Indicate(cab["faces"][t], color=colors[t], scale_factor=1.05)
                     for t in targets], run_time=0.6)
        # 归档：缩小飞入主抽屉右侧停靠
        slot = cab["rights"][d] + RIGHT * (0.7 + 1.25 * dock_count[d])
        scene.play(
            chip.animate.scale(0.42).move_to(slot),
            FadeOut(tag),
            run_time=0.7,
        )
        dock_count[d] += 1
        prev_ex = ex
        scene.wait(max(0.3, per - 2.5))

    pad_to(scene, DURATIONS[target_key])


class SortingTermsA(Scene):
    """热词归位（上）：规模法则 / MoE / RLHF / 推理模型。"""

    def construct(self):
        terms = [
            ("规模法则", [0], "堆更多·喂更多·烧更多\n损失可预测地下降"),
            ("MoE 专家混合", [0, 3], "前馈换成并排小专家\n每词只走几个"),
            ("RLHF / 对齐", [1], "再用人类偏好继续调\n机制还是梯度更新"),
            ("推理模型 o / R1", [1, 2], "回答前先写思考草稿\n本质仍是预测下一个词"),
        ]
        _sort_scene(self, "热词归位（上）", terms, "SortingTermsA")


class SortingTermsB(Scene):
    """热词归位（下）：RAG / Agent / 多模态 / 长上下文 / 量化蒸馏。"""

    def construct(self):
        terms = [
            ("RAG 检索增强", [2], "临时检索文本塞进上下文\n让注意力去读它"),
            ("智能体 Agent", [2], "放进循环：动作→结果→再决定\n这次 token 是动作"),
            ("多模态", [0, 2], "图/声也切成 token 变向量\n一切皆向量"),
            ("长上下文 128K/1M", [3], "绕开注意力 n² 成本\n一次读更多 token"),
            ("量化 / 蒸馏", [3], "量化=更少位数存参数\n蒸馏=小模型学大模型"),
        ]
        _sort_scene(self, "热词归位（下）", terms, "SortingTermsB")


class ThreeQuestions(Scene):
    """读 AI 新闻的三个追问。"""

    def construct(self):
        setup_bg(self)
        heading = cn("读 AI 新闻的三个追问", size=38, weight=BOLD).to_edge(UP, buff=0.5)
        self.play(Write(heading))

        # 一张『AI 新闻』卡片
        news = make_card("某 AI 新模型，号称颠覆一切！", color=SUB, h=1.0, size=26)
        news.move_to(LEFT * 3.4 + UP * 0.6)
        self.play(FadeIn(news, scale=0.9))

        qs = [
            ("1", "它属于哪个抽屉？", ACCENT),
            ("2", "动了哪个零件？\n嵌入/注意力/前馈/训练/微调/解码", GOOD),
            ("3", "真·新原理，还是主干上加的枝？\n（99% 是后者）", SUB),
        ]
        rows = VGroup()
        for num, text, color in qs:
            badge = cn(num, size=34, color=color, weight=BOLD)
            t = cn(text, size=22, color=FG)
            rows.add(VGroup(badge, t).arrange(RIGHT, buff=0.4, aligned_edge=UP))
        rows.arrange(DOWN, buff=0.55, aligned_edge=LEFT).move_to(RIGHT * 2.2 + DOWN * 0.1)
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in rows],
                              lag_ratio=0.35))

        arrow = Arrow(news.get_right(), rows.get_left(), color="#5f7a80",
                      stroke_width=4, buff=0.3)
        self.play(GrowArrow(arrow))

        caption = cn("能稳定回答这三问，你的认识就是根本性的了。",
                     size=26, color=ACCENT).to_edge(DOWN, buff=0.4)
        self.play(Write(caption))

        pad_to(self, DURATIONS["ThreeQuestions"])


class Day8Outro(Scene):
    """免疫力三行 + 全课主线收束（与第 1 天呼应）。"""

    def construct(self):
        setup_bg(self)
        heading = cn("你已经有了免疫力", size=36, weight=BOLD).to_edge(UP, buff=0.5)
        self.play(Write(heading))

        points = VGroup(
            cn("· 底层原理变化极慢 —— 你学的是骨架，不是新闻", size=24, color=FG),
            cn("· 新东西几乎都是已知零件的重新组合", size=24, color=FG),
            cn("· 判断进展看它在哪个抽屉、解决了什么真问题", size=24, color=FG),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT).move_to(UP * 1.4)
        self.play(LaggedStart(*[FadeIn(p, shift=RIGHT * 0.2) for p in points],
                              lag_ratio=0.3))
        self.wait(0.8)

        # 收束到全课主线大字（与 Day1Outro 同一版式）
        self.play(FadeOut(points), FadeOut(heading))
        main = cn("所有 AI，归根结底都在预测下一个词。",
                  size=44, color=ACCENT, weight=BOLD).move_to(UP * 0.3)
        sub = cn("这门课，到此结业。", size=28, color=SUB).next_to(main, DOWN, buff=0.6)
        self.play(Write(main))
        self.play(FadeIn(sub, shift=UP * 0.2))

        pad_to(self, DURATIONS["Day8Outro"])
