"""
第 7 天 · 工程现实与缺失的拼图 —— 5 镜。
脚本见 storyboards/day7_engineering.md；风格/配色见 CLAUDE.md 第 6 节。
渲染示例：
    manim -ql scenes/day7_engineering.py Day7Title
「主干+五枝」树用 _common.py 的 trunk_and_branches()，第 8 天复用。
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _common import *  # noqa: E402,F401,F403

DURATIONS = {
    "Day7Title": 15.6,        # 配音 15.14s
    "LoRA": 31.3,             # 配音 30.79s
    "MissingPieces": 50.5,    # 配音 50.02s
    "Day7RealWorld": 34.4,    # 配音 33.91s
    "Day7Outro": 10.4,        # 配音 9.89s
}


class Day7Title(Scene):
    def construct(self):
        setup_bg(self)
        title_card(
            self,
            "工程现实，和书没讲的拼图",
            sub="主干已成，其余皆是枝",
            intuition="真实世界的热词，几乎都长在这条主干上。",
        )
        pad_to(self, DURATIONS["Day7Title"])


class LoRA(Scene):
    """LoRA：冻结大模型，只挂一小撮新参数训练。"""

    def construct(self):
        setup_bg(self)
        heading = cn("LoRA：又快又省的微调", size=38, weight=BOLD).to_edge(UP, buff=0.5)
        self.play(Write(heading))

        # 大模型：所有参数冻结、变灰、上锁
        big = RoundedRectangle(corner_radius=0.3, width=4.2, height=3.0,
                               stroke_color=SUB, stroke_width=3,
                               fill_color="#1d3f4a", fill_opacity=1)
        big.move_to(LEFT * 3.0 + UP * 0.2)
        big_lbl = cn("超大模型\n几十亿参数", size=26, color=FG).move_to(big.get_center())
        self.play(FadeIn(big, scale=0.9), Write(big_lbl))
        self.wait(0.3)

        # 冻结：变灰（不用 emoji 锁，避免字形缺失）
        frozen_lbl = cn("冻结 · 不动", size=24, color="#5f7a80").next_to(big, DOWN, buff=0.3)
        self.play(
            big.animate.set_stroke("#5f7a80").set_fill("#16323b", opacity=1),
            big_lbl.animate.set_color("#7f9499"),
            Write(frozen_lbl),
        )
        self.wait(0.4)

        # 旁挂一小撮新参数（小外挂芯片）
        chip = RoundedRectangle(corner_radius=0.12, width=1.4, height=1.0,
                                stroke_color=ACCENT, stroke_width=3,
                                fill_color="#3a2418", fill_opacity=1)
        chip.move_to(RIGHT * 0.3 + UP * 0.6)
        chip_lbl = cn("小外挂\n只训练它", size=22, color=ACCENT).move_to(chip.get_center())
        conn = Line(big.get_right(), chip.get_left(), color=ACCENT, stroke_width=3)
        self.play(Create(conn), FadeIn(chip, scale=0.7), Write(chip_lbl))
        self.wait(0.3)

        # 对比条：完整微调（贵）vs LoRA（便宜）
        full_track = Rectangle(width=3.2, height=0.4, stroke_color="#3a5a62",
                               stroke_width=1.5, fill_opacity=0)
        full_fill = Rectangle(width=3.2, height=0.4, stroke_width=0,
                              fill_color="#5f7a80", fill_opacity=0.9).align_to(full_track, LEFT)
        full = VGroup(full_track, full_fill)
        full_lbl = cn("完整微调：动全部参数", size=20, color="#7f9499")
        full_row = VGroup(full_lbl, full).arrange(RIGHT, buff=0.3)

        lora_track = Rectangle(width=3.2, height=0.4, stroke_color="#3a5a62",
                               stroke_width=1.5, fill_opacity=0)
        lora_fill = Rectangle(width=0.4, height=0.4, stroke_width=0,
                              fill_color=ACCENT, fill_opacity=0.9).align_to(lora_track, LEFT)
        lora = VGroup(lora_track, lora_fill)
        lora_lbl = cn("LoRA：只动一小撮", size=20, color=ACCENT)
        lora_row = VGroup(lora_lbl, lora).arrange(RIGHT, buff=0.3)

        compare = VGroup(full_row, lora_row).arrange(DOWN, buff=0.25, aligned_edge=RIGHT)
        compare.move_to(RIGHT * 1.3 + DOWN * 2.55)
        self.play(FadeIn(compare, shift=UP * 0.2))

        caption = cn("一张消费级显卡，就能定制大模型。",
                     size=26, color=GOOD).to_edge(DOWN, buff=0.25)
        self.play(Write(caption))

        pad_to(self, DURATIONS["LoRA"])


class MissingPieces(Scene):
    """主干 + 五枝：RLHF / 推理 / RAG / Agent / 多模态。"""

    def construct(self):
        setup_bg(self)
        heading = cn("书故意没讲的五块拼图", size=38, weight=BOLD).to_edge(UP, buff=0.45)
        self.play(Write(heading))

        tree = trunk_and_branches()
        tree["group"].scale(0.92).move_to(DOWN * 0.3)
        trunk, cards, lines = tree["trunk"], tree["cards"], tree["lines"]

        self.play(FadeIn(trunk, scale=0.9))
        self.wait(0.2)
        # 逐根枝长出来
        for ln, card in zip(lines, cards):
            self.play(Create(ln), FadeIn(card, shift=RIGHT * 0.2), run_time=0.7)
        self.wait(0.4)

        caption = cn("不是另起炉灶的新原理，全建立在你这六天学的基础上。",
                     size=24, color=ACCENT).to_edge(DOWN, buff=0.35)
        self.play(Write(caption))

        pad_to(self, DURATIONS["MissingPieces"])


class Day7RealWorld(Scene):
    """开源追平闭源 + 拆穿『低成本微调专业模型』。"""

    def construct(self):
        setup_bg(self)
        heading = cn("连接现实", size=40, weight=BOLD).to_edge(UP, buff=0.5)
        self.play(Write(heading))

        # 开源底座们
        names = ["DeepSeek", "Qwen", "Llama"]
        chips = VGroup(*[make_card(n, color=GOOD, h=0.8, size=24) for n in names])
        chips.arrange(RIGHT, buff=0.5).move_to(UP * 1.9)
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in chips], lag_ratio=0.2))

        # 三条原因 → 护城河没那么深
        reasons = VGroup(
            cn("· 预训练配方公开", size=24, color=FG),
            cn("· LoRA 把微调成本打下来", size=24, color=FG),
            cn("· 算法上的各种巧思", size=24, color=FG),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT).move_to(LEFT * 2.5 + DOWN * 0.2)
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in reasons],
                              lag_ratio=0.25))
        moat = cn("→ 护城河没那么深", size=28, color=ACCENT).next_to(
            reasons, RIGHT, buff=0.7)
        self.play(Write(moat))
        self.wait(0.4)

        # 拆穿新闻气泡
        bubble = make_card("「我们用很少成本，微调出了专业领域模型」",
                           color=SUB, h=0.9, size=22).move_to(DOWN * 2.2 + UP * 0.1)
        self.play(FadeIn(bubble, shift=UP * 0.2))
        truth = cn("大概率：在开源底座上做了 LoRA，而非从零预训练。",
                   size=22, color=GOOD).to_edge(DOWN, buff=0.35)
        self.play(Write(truth))

        pad_to(self, DURATIONS["Day7RealWorld"])


class Day7Outro(Scene):
    """收束：主干+五枝树定格（与第 8 天呼应）。"""

    def construct(self):
        setup_bg(self)
        tree = trunk_and_branches()
        tree["group"].scale(0.85).move_to(DOWN * 0.2)
        self.play(FadeIn(tree["group"]))

        main = cn("主干，你已经亲手造完了。", size=40, weight=BOLD).to_edge(UP, buff=0.7)
        sub = cn("明天，把所有枝归位成一张地图。", size=26, color=SUB).to_edge(DOWN, buff=0.5)
        self.play(Write(main))
        self.play(FadeIn(sub, shift=UP * 0.2))

        pad_to(self, DURATIONS["Day7Outro"])
