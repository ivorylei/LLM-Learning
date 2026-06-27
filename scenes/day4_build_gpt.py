"""
第 4 天 · 把 GPT 拼起来 —— 9 镜。
脚本见 storyboards/day4_build_gpt.md；风格/配色见 CLAUDE.md 第 6 节。
渲染示例：
    manim -ql scenes/day4_build_gpt.py Day4Title
时长先用估值占位，配音(edge-tts)后再回填对齐 DURATIONS。
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _common import *  # noqa: E402,F401,F403

DURATIONS = {
    "Day4Title": 11.9,           # 配音 11.38s
    "BlockOverview": 13.8,       # 配音 13.30s
    "FeedForward": 19.0,         # 配音 18.48s
    "Residual": 19.7,            # 配音 19.15s
    "LayerNorm": 11.6,           # 配音 11.14s
    "StackToGPT": 22.9,          # 配音 22.37s
    "WhatAreParameters": 17.6,   # 配音 17.09s
    "Day4RealWorld": 29.4,       # 配音 28.90s
    "Day4Outro": 14.7,           # 配音 14.21s
}

BLOCK_ROWS = [("注意力层", ACCENT), ("前馈网络", GOOD),
              ("残差连接", SUB), ("层归一化", FG)]


def block_full():
    """积木块全貌：四层带标签，外面套一个圆角外框。"""
    rows = VGroup()
    for name, col in BLOCK_ROWS:
        bar = RoundedRectangle(corner_radius=0.08, width=4.2, height=0.65,
                               stroke_color=col, stroke_width=2,
                               fill_color=col, fill_opacity=0.14)
        lab = cn(name, size=26, color=col).move_to(bar)
        rows.add(VGroup(bar, lab))
    rows.arrange(DOWN, buff=0.22)
    outer = RoundedRectangle(corner_radius=0.2, width=rows.width + 0.7,
                             height=rows.height + 0.7, stroke_color=FG,
                             stroke_width=3, fill_opacity=0).move_to(rows)
    return VGroup(outer, rows)


def block_thin(w=3.0):
    """堆叠用的薄积木块：外框 + 三道彩色条。"""
    box = RoundedRectangle(corner_radius=0.06, width=w, height=0.5,
                           stroke_color=SUB, stroke_width=1.5,
                           fill_color=BG, fill_opacity=1)
    stripes = VGroup(*[
        Rectangle(width=w - 0.5, height=0.06, stroke_width=0,
                  fill_color=c, fill_opacity=0.75)
        for c in (ACCENT, GOOD, SUB)
    ]).arrange(DOWN, buff=0.04).move_to(box)
    return VGroup(box, stripes)


# ----------------------------------------------------------------------
# 镜 00 · 标题卡
# ----------------------------------------------------------------------
class Day4Title(Scene):
    def construct(self):
        setup_bg(self)
        title = cn("把 GPT 拼起来", size=68, weight=BOLD)
        sub = cn("一块积木，重复堆 N 层", size=36, color=ACCENT)
        intuition = cn("参数，就是积木里那一大堆可调的数字。", size=26, color=SUB)
        group = VGroup(title, sub, intuition).arrange(DOWN, buff=0.5)

        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(sub, shift=UP * 0.2), run_time=0.7)
        self.play(FadeIn(intuition, shift=UP * 0.2), run_time=0.7)
        pad_to(self, DURATIONS["Day4Title"] - 0.6)
        self.play(FadeOut(group), run_time=0.6)


# ----------------------------------------------------------------------
# 镜 01 · 一块积木的全貌
# ----------------------------------------------------------------------
class BlockOverview(Scene):
    def construct(self):
        setup_bg(self)
        heading = cn("一块积木里有什么", size=32, color=FG).to_edge(UP, buff=0.6)
        block = block_full().move_to(DOWN * 0.3)
        title_tag = cn("Transformer 积木块", size=24, color=SUB)
        title_tag.next_to(block, DOWN, buff=0.35)

        self.play(FadeIn(heading, shift=DOWN * 0.2))
        self.play(Create(block[0]))
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in block[1]],
                              lag_ratio=0.25))
        self.play(FadeIn(title_tag))
        # 注意力层已学过——描一圈带过
        self.play(Indicate(block[1][0], color=ACCENT, scale_factor=1.05))
        note = cn("注意力昨天讲过，今天看其余三样", size=24, color=SUB)
        note.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(note, shift=UP * 0.1))
        pad_to(self, DURATIONS["BlockOverview"])


# ----------------------------------------------------------------------
# 镜 02 · 前馈网络 + GELU
# ----------------------------------------------------------------------
class FeedForward(Scene):
    def construct(self):
        setup_bg(self)
        heading = cn("① 前馈网络：每个词自己关门消化", size=30, color=GOOD)
        heading.to_edge(UP, buff=0.6)

        vec_in = Rectangle(width=0.4, height=1.3, stroke_width=0,
                           fill_color=SUB, fill_opacity=0.7)
        ffn = make_card("前馈网络", color=GOOD, size=26, w=2.4, h=1.5)
        vec_out = Rectangle(width=0.4, height=1.3, stroke_width=0,
                            fill_color=GOOD, fill_opacity=0.85)
        flow = VGroup(vec_in, ffn, vec_out).arrange(RIGHT, buff=1.0)
        flow.move_to(UP * 1.9 + LEFT * 0.5)
        a1 = Arrow(vec_in.get_right(), ffn.get_left(), buff=0.2, color="#3a5a62")
        a2 = Arrow(ffn.get_right(), vec_out.get_left(), buff=0.2, color="#3a5a62")
        cap = cn("注意力 = 词之间交流；前馈 = 每个词自己消化", size=24, color=FG)
        cap.next_to(flow, DOWN, buff=0.45)

        # GELU 曲线（放在左下，标签在右侧，避免和上方文字打架）
        axes = Axes(x_range=[-3, 3, 1], y_range=[-1, 3, 1], x_length=3.8, y_length=2.3,
                    axis_config={"include_tip": False, "stroke_color": "#3a5a62"})
        axes.move_to(DOWN * 1.7 + LEFT * 2.2)
        gelu = axes.plot(lambda x: x / (1 + np.exp(-1.702 * x)),
                         x_range=[-3, 3], color=ACCENT)
        gelu.set_stroke(width=5)
        glabel = cn("GELU\n非线性函数", size=26, color=ACCENT).next_to(axes, RIGHT, buff=0.6)
        gnote = cn("没有非线性，堆再多层也等于一层", size=24, color=SUB)
        gnote.to_edge(DOWN, buff=0.4)

        self.play(FadeIn(heading, shift=DOWN * 0.2))
        self.play(FadeIn(vec_in, shift=RIGHT * 0.2))
        self.play(GrowArrow(a1), FadeIn(ffn, scale=0.9))
        self.play(GrowArrow(a2), FadeIn(vec_out, shift=RIGHT * 0.2))
        self.play(FadeIn(cap, shift=UP * 0.1))
        self.wait(0.4)
        self.play(Create(axes), FadeIn(glabel))
        self.play(Create(gelu), run_time=1.5)
        self.play(FadeIn(gnote, shift=UP * 0.1))
        pad_to(self, DURATIONS["FeedForward"])


# ----------------------------------------------------------------------
# 镜 03 · 残差连接（高速公路）
# ----------------------------------------------------------------------
class Residual(Scene):
    def construct(self):
        setup_bg(self)
        heading = cn("② 残差连接：给信息修一条高速公路", size=30, color=SUB)
        heading.to_edge(UP, buff=0.6)

        x_in = make_card("输入", color=FG, size=24, w=1.4, h=0.9)
        layer = make_card("某一层", color=GOOD, size=24, w=2.0, h=0.9)
        plus = VGroup(Circle(radius=0.32, stroke_color=ACCENT, stroke_width=3,
                             fill_color=BG, fill_opacity=1),
                      cn("+", size=34, color=ACCENT))
        x_out = make_card("输出", color=FG, size=24, w=1.4, h=0.9)
        row = VGroup(x_in, layer, plus, x_out).arrange(RIGHT, buff=1.1)
        row.move_to(UP * 0.9)
        a1 = Arrow(x_in.get_right(), layer.get_left(), buff=0.15, color="#3a5a62")
        a2 = Arrow(layer.get_right(), plus.get_left(), buff=0.15, color="#3a5a62")
        a3 = Arrow(plus.get_right(), x_out.get_left(), buff=0.15, color="#3a5a62")

        # 高速公路：从输入绕过层，直接到 +（弧度收一点，别顶到标题）
        bypass = CurvedArrow(x_in.get_top(), plus.get_top(),
                             angle=-PI / 3, color=ACCENT)
        bypass_lab = cn("跳过这层，直接相加", size=22, color=ACCENT)
        bypass_lab.next_to(bypass, UP, buff=0.12)

        # 对比：无残差(变暗坍塌) vs 有残差(稳传)
        def stack(fade):
            sq = VGroup()
            for i in range(5):
                op = (1.0 - 0.22 * i) if fade else 0.8
                sq.add(Rectangle(width=1.4, height=0.4, stroke_width=0,
                                 fill_color=GOOD, fill_opacity=max(0.05, op)))
            return sq.arrange(DOWN, buff=0.12)

        left = stack(True)
        right = stack(False)
        hw = Line(right.get_top(), right.get_bottom(),
                  stroke_color=ACCENT, stroke_width=4).next_to(right, RIGHT, buff=0.1)
        lg = VGroup(left, cn("无残差：越深越糊", size=22, color=FG)).arrange(DOWN, buff=0.3)
        rg = VGroup(VGroup(right, hw),
                    cn("有残差：稳稳传到底", size=22, color=GOOD)).arrange(DOWN, buff=0.3)
        compare = VGroup(lg, rg).arrange(RIGHT, buff=2.5).to_edge(DOWN, buff=0.5)

        self.play(FadeIn(heading, shift=DOWN * 0.2))
        self.play(FadeIn(x_in), GrowArrow(a1), FadeIn(layer),
                  GrowArrow(a2), FadeIn(plus), GrowArrow(a3), FadeIn(x_out))
        self.play(Create(bypass), FadeIn(bypass_lab, shift=DOWN * 0.1))
        self.wait(0.4)
        self.play(FadeIn(lg, shift=UP * 0.1), FadeIn(rg, shift=UP * 0.1))
        pad_to(self, DURATIONS["Residual"])


# ----------------------------------------------------------------------
# 镜 04 · 层归一化
# ----------------------------------------------------------------------
class LayerNorm(Scene):
    def construct(self):
        setup_bg(self)
        heading = cn("③ 层归一化：每层把数值重新标准化", size=30, color=FG)
        heading.to_edge(UP, buff=0.6)

        wild_h = [2.6, 0.3, 1.9, 0.15, 2.2, 0.4]
        wild = VGroup(*[
            Rectangle(width=0.5, height=h, stroke_width=0,
                      fill_color=ACCENT, fill_opacity=0.8)
            for h in wild_h
        ]).arrange(RIGHT, buff=0.25, aligned_edge=DOWN)
        wild.move_to(LEFT * 4 + DOWN * 0.2)
        wlab = cn("忽大忽小，要爆炸或消失", size=22, color=ACCENT)
        wlab.next_to(wild, DOWN, buff=0.4)

        gate = make_card("层归一化", color=SUB, size=24, w=2.2, h=1.2).move_to(ORIGIN)

        calm_h = [1.3, 1.0, 1.25, 0.95, 1.3, 1.05]
        calm = VGroup(*[
            Rectangle(width=0.5, height=h, stroke_width=0,
                      fill_color=GOOD, fill_opacity=0.85)
            for h in calm_h
        ]).arrange(RIGHT, buff=0.25, aligned_edge=DOWN)
        calm.move_to(RIGHT * 4 + DOWN * 0.2)
        clab = cn("平稳规整", size=22, color=GOOD).next_to(calm, DOWN, buff=0.4)

        a1 = Arrow(wild.get_right(), gate.get_left(), buff=0.3, color="#3a5a62")
        a2 = Arrow(gate.get_right(), calm.get_left(), buff=0.3, color="#3a5a62")

        self.play(FadeIn(heading, shift=DOWN * 0.2))
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in wild], lag_ratio=0.1),
                  FadeIn(wlab))
        self.play(GrowArrow(a1), FadeIn(gate, scale=0.9))
        self.play(GrowArrow(a2),
                  LaggedStart(*[GrowFromEdge(b, DOWN) for b in calm], lag_ratio=0.1),
                  FadeIn(clab))
        pad_to(self, DURATIONS["LayerNorm"])


# ----------------------------------------------------------------------
# 镜 05 · 堆叠成 GPT
# ----------------------------------------------------------------------
class StackToGPT(Scene):
    def construct(self):
        setup_bg(self)
        heading = cn("把积木重复堆 N 层，就是一个 GPT", size=30, color=FG)
        heading.to_edge(UP, buff=0.5)

        stack = VGroup(*[block_thin(w=2.6) for _ in range(12)])
        stack.arrange(UP, buff=0.06)
        stack.scale_to_fit_height(4.6).move_to(LEFT * 3.5 + DOWN * 0.3)
        brace = Brace(stack, LEFT, color=SUB)
        brace_lab = cn("12 块\nGPT-2 small", size=22, color=SUB)
        brace_lab.next_to(brace, LEFT, buff=0.2)
        big_hint = cn("大模型堆几十、上百块", size=22, color=ACCENT)
        big_hint.next_to(stack, UP, buff=0.2)

        out_layer = make_card("输出层", color=GOOD, size=24, w=2.0, h=0.9)
        out_layer.next_to(stack, RIGHT, buff=1.6)
        a_out = Arrow(stack.get_right(), out_layer.get_left(), buff=0.2,
                      color="#3a5a62")

        # 下一个词概率（未训练→近似乱码）
        bars = VGroup(
            prob_bar("猫", 0.22, bar_max=2.0, color=SUB, size=24),
            prob_bar("桌子", 0.19, bar_max=2.0, color=SUB, size=24),
            prob_bar("的", 0.21, bar_max=2.0, color=SUB, size=24),
            prob_bar("紫色", 0.18, bar_max=2.0, color=SUB, size=24),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        bars.next_to(out_layer, RIGHT, buff=0.9)
        a_bars = Arrow(out_layer.get_right(), bars.get_left(), buff=0.2,
                       color="#3a5a62")
        prob_lab = cn("下一个词的概率", size=22, color=GOOD).next_to(bars, UP, buff=0.3)
        untrained = cn("未训练 → 乱码，但架构是真的、完整的", size=24, color=ACCENT)
        untrained.to_edge(DOWN, buff=0.45)

        self.play(FadeIn(heading, shift=DOWN * 0.2))
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.15) for b in stack],
                              lag_ratio=0.12), run_time=2.2)
        self.play(GrowFromCenter(brace), FadeIn(brace_lab))
        self.play(FadeIn(big_hint, shift=DOWN * 0.1))
        self.play(GrowArrow(a_out), FadeIn(out_layer, scale=0.9))
        self.play(GrowArrow(a_bars), FadeIn(prob_lab),
                  LaggedStart(*[FadeIn(b) for b in bars], lag_ratio=0.15))
        self.play(FadeIn(untrained, shift=UP * 0.1))
        pad_to(self, DURATIONS["StackToGPT"])


# ----------------------------------------------------------------------
# 镜 06 · 参数是什么
# ----------------------------------------------------------------------
class WhatAreParameters(Scene):
    def construct(self):
        setup_bg(self)
        heading = cn("「参数量」到底是什么？", size=32, color=FG).to_edge(UP, buff=0.6)

        # 一格放大的积木内部：满是可调的小数字
        vals = ["0.21", "-1.34", "0.07", "2.10", "-0.88", "0.45",
                "1.62", "-0.30", "0.99", "-2.01", "0.13", "0.76"]
        grid = VGroup(*[cn(v, size=24, color=SUB) for v in vals])
        grid.arrange_in_grid(rows=3, cols=4, buff=0.6).move_to(UP * 0.6)
        frame = RoundedRectangle(corner_radius=0.2, width=grid.width + 0.8,
                                 height=grid.height + 0.8, stroke_color=FG,
                                 stroke_width=2, fill_opacity=0).move_to(grid)

        counter = ValueTracker(1)
        num = always_redraw(lambda: VGroup(
            DecimalNumber(counter.get_value(), num_decimal_places=0,
                          color=ACCENT, font_size=44),
            cn("亿个可调数字", size=28, color=ACCENT),
        ).arrange(RIGHT, buff=0.2).to_edge(DOWN, buff=1.1))
        caption = cn("参数 = 模型里所有这些可调数字的总数", size=26, color=GOOD)
        caption.to_edge(DOWN, buff=0.5)

        self.play(FadeIn(heading, shift=DOWN * 0.2))
        self.play(Create(frame),
                  LaggedStart(*[FadeIn(v, scale=0.6) for v in grid], lag_ratio=0.06))
        self.play(LaggedStart(*[Indicate(v, color=ACCENT, scale_factor=1.3)
                                for v in grid], lag_ratio=0.05, run_time=1.5))
        self.add(num)
        self.play(counter.animate.set_value(3000), run_time=2.5,
                  rate_func=rate_functions.ease_in_out_sine)
        self.play(FadeIn(caption, shift=UP * 0.1))
        pad_to(self, DURATIONS["WhatAreParameters"])


# ----------------------------------------------------------------------
# 镜 07 · 连接现实（规模法则雏形 + MoE 钩子）
# ----------------------------------------------------------------------
class Day4RealWorld(Scene):
    def construct(self):
        setup_bg(self)
        heading = cn("GPT-2、3、4、5 的差别：堆得更多", size=30, color=FG)
        heading.to_edge(UP, buff=0.6)
        self.play(FadeIn(heading, shift=DOWN * 0.2))

        names = ["GPT-2", "GPT-3", "GPT-4", "GPT-5"]
        heights = [1.0, 1.8, 2.6, 3.3]
        towers = VGroup()
        for nm, h in zip(names, heights):
            n_blocks = max(2, int(h / 0.4))
            col = VGroup(*[block_thin(w=1.5) for _ in range(n_blocks)])
            col.arrange(UP, buff=0.05).scale_to_fit_height(h)
            lab = cn(nm, size=22, color=SUB).next_to(col, DOWN, buff=0.2)
            towers.add(VGroup(col, lab))
        towers.arrange(RIGHT, buff=1.0, aligned_edge=DOWN).move_to(UP * 0.2 + LEFT * 0.5)
        scale_lab = cn("规模法则的雏形：原理没变，是堆得更多", size=24, color=ACCENT)
        scale_lab.to_edge(DOWN, buff=0.5)

        self.play(LaggedStart(*[FadeIn(t, shift=UP * 0.2) for t in towers],
                              lag_ratio=0.25), run_time=2.2)
        self.play(FadeIn(scale_lab, shift=UP * 0.1))
        self.wait(0.8)
        self.play(FadeOut(towers), FadeOut(scale_lab))

        # MoE 钩子
        moe_head = cn("钩子：MoE = 把前馈换成多个并排小专家", size=28, color=GOOD)
        moe_head.move_to(UP * 2.0)
        word = make_card("一个词", color=FG, size=24, w=1.8, h=0.8).move_to(
            LEFT * 4.5 + DOWN * 0.3)
        experts = VGroup(*[make_card(f"专家{i + 1}", color=SUB, size=22, w=1.6, h=0.7)
                           for i in range(4)])
        experts.arrange(DOWN, buff=0.35).move_to(RIGHT * 1.0 + DOWN * 0.3)
        # 只走其中两个（高亮）
        chosen = (0, 2)
        routes = VGroup()
        for i, ex in enumerate(experts):
            on = i in chosen
            r = Arrow(word.get_right(), ex.get_left(), buff=0.2,
                      color=GOOD if on else "#3a5a62",
                      stroke_width=5 if on else 2)
            if on:
                ex[0].set_stroke(GOOD, width=3)
                ex[1].set_color(GOOD)
            routes.add(r)
        moe_note = cn("每个词只走其中几个 → 省算力", size=24, color=ACCENT)
        moe_note.to_edge(DOWN, buff=0.5)

        self.play(FadeIn(moe_head, shift=DOWN * 0.1))
        self.play(FadeIn(word, shift=RIGHT * 0.2),
                  LaggedStart(*[FadeIn(e) for e in experts], lag_ratio=0.15))
        self.play(LaggedStart(*[GrowArrow(r) for r in routes], lag_ratio=0.15))
        self.play(FadeIn(moe_note, shift=UP * 0.1))
        pad_to(self, DURATIONS["Day4RealWorld"])


# ----------------------------------------------------------------------
# 镜 08 · 收束卡
# ----------------------------------------------------------------------
class Day4Outro(Scene):
    def construct(self):
        setup_bg(self)
        chain = cn("一块积木 × N 层 → 一个 GPT", size=42, weight=BOLD)
        sub = cn("顶上那排概率，就是在预测下一个词。", size=26, color=SUB)
        group = VGroup(chain, sub).arrange(DOWN, buff=0.6)

        self.play(Write(chain))
        self.play(FadeIn(sub, shift=UP * 0.2))
        pad_to(self, DURATIONS["Day4Outro"])
