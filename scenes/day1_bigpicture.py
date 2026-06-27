"""
第 1 天 · 大地图 —— 7 镜。
脚本见 storyboards/day1_bigpicture.md；风格/配色见 CLAUDE.md 第 6 节。
渲染示例：
    manim -ql scenes/day1_bigpicture.py Day1Title
    manim -ql scenes/day1_bigpicture.py NextTokenCore
时长先用估值占位，配音(edge-tts)后再回填对齐 DURATIONS。
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _common import *  # noqa: E402,F401,F403

# 每镜目标时长 = edge-tts 实测配音时长 + 0.5s 收尾（见 audio/day1_*.mp3）
DURATIONS = {
    "Day1Title": 10.4,        # 配音 9.86s
    "NextTokenCore": 25.0,    # 配音 24.46s
    "TwoStages": 33.4,        # 配音 32.86s
    "OneFamily": 27.0,        # 配音 26.45s
    "MathIntuitions": 36.6,   # 配音 36.10s
    "Day1RealWorld": 23.4,    # 配音 22.85s
    "Day1Outro": 11.3,        # 配音 10.78s
}


def make_card(text, color=FG, w=None, h=1.0, fill=BG, size=30):
    """一个圆角卡片（VGroup[box, label]），宽度按文字自适应。"""
    label = cn(text, size=size, color=color)
    width = w if w is not None else label.width + 0.7
    box = RoundedRectangle(corner_radius=0.15, width=width, height=h,
                           stroke_color=color, stroke_width=2.5,
                           fill_color=fill, fill_opacity=1)
    return VGroup(box, label)


# ----------------------------------------------------------------------
# 镜 00 · 标题卡
# ----------------------------------------------------------------------
class Day1Title(Scene):
    def construct(self):
        setup_bg(self)
        title = cn("大地图", size=78, weight=BOLD)
        sub = cn("一句话看懂整个 LLM", size=36, color=ACCENT)
        intuition = cn("一句话：大模型就是一个超级「下一个词预测器」。",
                       size=28, color=SUB)
        group = VGroup(title, sub, intuition).arrange(DOWN, buff=0.5)

        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(sub, shift=UP * 0.2), run_time=0.7)
        self.play(FadeIn(intuition, shift=UP * 0.2), run_time=0.7)
        pad_to(self, DURATIONS["Day1Title"] - 0.6)
        self.play(FadeOut(group), run_time=0.6)


# ----------------------------------------------------------------------
# 镜 01 · 预测下一个词（主线）
# ----------------------------------------------------------------------
class NextTokenCore(Scene):
    def construct(self):
        setup_bg(self)
        heading = cn("大模型只做一件事：预测下一个词", size=30, color=SUB)
        heading.to_edge(UP, buff=0.6)

        anchor = LEFT * 5.2 + UP * 1.1

        def sent(txt):
            return cn(txt, size=46).move_to(anchor, aligned_edge=LEFT)

        s = sent("今天天气真不＿")
        s[-1].set_color(ACCENT)  # 闪烁的空格

        self.play(FadeIn(heading, shift=DOWN * 0.2))
        self.play(Write(s))
        # 空格闪烁两下
        for _ in range(2):
            self.play(s[-1].animate.set_opacity(0.2), run_time=0.35)
            self.play(s[-1].animate.set_opacity(1.0), run_time=0.35)

        # 两个候选 + 概率条
        bars = VGroup(
            prob_bar("好", 0.70, color=GOOD),
            prob_bar("错", 0.20, color=SUB),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5).move_to(DOWN * 1.4)
        cap = cn("脑子里立刻冒出候选，挑概率最高的", size=24, color=FG)
        cap.next_to(bars, DOWN, buff=0.5)

        self.play(LaggedStart(*[FadeIn(b, shift=RIGHT * 0.2) for b in bars],
                              lag_ratio=0.3))
        self.play(FadeIn(cap, shift=UP * 0.2))
        self.wait(0.8)
        self.play(Indicate(bars[0], color=GOOD, scale_factor=1.1))

        # 选中“好”，补全句子
        # 中文整句变化一律用淡出+淡入（在原位），不要用 Transform/TransformMatchingShapes：
        # 逐字形匹配会让 CJK 字形飞行中互相穿插、糊成一团（见 CLAUDE.md 第 6 节规则）。
        s2 = sent("今天天气真不好")
        self.play(FadeOut(s), FadeIn(s2), run_time=0.6)
        s = s2
        self.play(FadeOut(bars), FadeOut(cap))

        # 循环：预测→填上→再预测
        loop_tag = cn("预测 → 填上 → 再预测", size=26, color=ACCENT)
        loop_tag.move_to(DOWN * 1.6)
        self.play(FadeIn(loop_tag))

        for txt in ["今天天气真不好，出门＿", "今天天气真不好，出门带伞＿"]:
            cur = sent(txt)
            cur[-1].set_color(ACCENT)
            self.play(FadeOut(s), FadeIn(cur), run_time=0.6)
            s = cur
            self.wait(0.25)
            done = sent(txt[:-1])
            self.play(FadeOut(s), FadeIn(done), run_time=0.4)
            s = done

        magic = cn("没有别的魔法。这条主线，贯穿整门课。", size=26, color=FG)
        magic.next_to(loop_tag, DOWN, buff=0.5)
        self.play(FadeIn(magic, shift=UP * 0.2))
        pad_to(self, DURATIONS["NextTokenCore"])


# ----------------------------------------------------------------------
# 镜 02 · 两大阶段：预训练 → 微调
# ----------------------------------------------------------------------
class TwoStages(Scene):
    def construct(self):
        setup_bg(self)
        heading = cn("造一个大模型，分两步", size=32, color=FG).to_edge(UP, buff=0.6)

        a = make_card("海量文本", color=SUB, size=28)
        b = make_card("基础模型\n（会语言但很野）", color=FG, size=24, h=1.3)
        c = make_card("听话的助手", color=GOOD, size=28)
        nodes = VGroup(a, b, c).arrange(RIGHT, buff=2.2).move_to(UP * 0.3)

        arr1 = Arrow(a.get_right(), b.get_left(), buff=0.15, color=ACCENT,
                     stroke_width=6)
        arr2 = Arrow(b.get_right(), c.get_left(), buff=0.15, color=GOOD,
                     stroke_width=6)

        # 预训练标签 + “烧钱/贵”
        lab1 = cn("预训练 · 烧钱", size=24, color=ACCENT).next_to(arr1, UP, buff=0.25)
        money = cn("$$$", size=30, color=ACCENT).next_to(arr1, DOWN, buff=0.3)
        cost1 = cn("极贵", size=22, color=ACCENT).next_to(money, DOWN, buff=0.15)

        # 微调标签 + 调音旋钮
        lab2 = cn("微调 · 调教", size=24, color=GOOD).next_to(arr2, UP, buff=0.25)
        knob = VGroup(
            Circle(radius=0.28, stroke_color=GOOD, stroke_width=3, fill_opacity=0),
            Line(ORIGIN, UP * 0.24, stroke_color=GOOD, stroke_width=3).rotate(
                -PI / 5, about_point=ORIGIN),
        ).next_to(arr2, DOWN, buff=0.3)
        cost2 = cn("便宜，却很关键", size=22, color=GOOD).next_to(knob, DOWN, buff=0.15)

        self.play(FadeIn(heading, shift=DOWN * 0.2))
        self.play(FadeIn(a, shift=RIGHT * 0.2))
        self.play(GrowArrow(arr1), FadeIn(lab1, shift=DOWN * 0.1))
        self.play(FadeIn(money, scale=0.6), FadeIn(cost1))
        self.play(FadeIn(b, shift=RIGHT * 0.2))
        self.wait(0.5)
        self.play(GrowArrow(arr2), FadeIn(lab2, shift=DOWN * 0.1))
        self.play(FadeIn(knob), FadeIn(cost2))
        self.play(FadeIn(c, shift=RIGHT * 0.2))
        self.wait(0.4)
        tail = cn("后面六天，就是拆开这两个箭头", size=26, color=SUB).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(tail, shift=UP * 0.2))
        pad_to(self, DURATIONS["TwoStages"])


# ----------------------------------------------------------------------
# 镜 03 · 同一个家族 Transformer
# ----------------------------------------------------------------------
class OneFamily(Scene):
    def construct(self):
        setup_bg(self)
        center = make_card("Transformer（2017）", color=ACCENT, size=30, h=1.1)
        center.move_to(UP * 0.3)

        names = ["GPT", "Claude", "Gemini", "DeepSeek", "Qwen", "Llama"]
        radius = 2.8
        chips = VGroup()
        lines = VGroup()
        for i, name in enumerate(names):
            ang = PI / 2 - i * TAU / len(names)
            pos = center.get_center() + radius * np.array(
                [np.cos(ang) * 1.5, np.sin(ang), 0])
            chip = make_card(name, color=SUB, size=26, h=0.7)
            chip.move_to(pos)
            line = Line(center.get_center(), chip.get_center(),
                        stroke_color="#3a5a62", stroke_width=2)
            chips.add(chip)
            lines.add(line)

        self.play(FadeIn(center, scale=0.8))
        self.play(
            LaggedStart(*[GrowFromPoint(c, center.get_center()) for c in chips],
                        lag_ratio=0.15),
            LaggedStart(*[Create(l) for l in lines], lag_ratio=0.15),
            run_time=2.5,
        )
        caption = cn("名字五花八门，骨架是同一个。", size=30, color=FG)
        caption.to_edge(DOWN, buff=0.6)
        self.play(Write(caption))
        self.wait(0.5)
        self.play(Indicate(center, color=ACCENT, scale_factor=1.08))
        pad_to(self, DURATIONS["OneFamily"])


# ----------------------------------------------------------------------
# 镜 04 · 三个数学直觉（向量 / 矩阵乘法 / 梯度）
# ----------------------------------------------------------------------
class MathIntuitions(Scene):
    def construct(self):
        setup_bg(self)
        heading = cn("后面反复用到的三个直觉", size=30, color=SUB).to_edge(UP, buff=0.5)
        self.play(FadeIn(heading, shift=DOWN * 0.2))

        # ---- ① 向量 ----
        tag1 = cn("① 向量 = 一串数 = 空间里的点", size=28, color=ACCENT)
        tag1.next_to(heading, DOWN, buff=0.4)
        axes = Axes(x_range=[0, 6, 1], y_range=[0, 6, 1], x_length=4.2, y_length=4.2,
                    axis_config={"include_tip": True, "stroke_color": "#3a5a62"})
        axes.move_to(LEFT * 3 + DOWN * 0.6)
        pairs = [("国王", (4.4, 4.7), ACCENT, UP), ("女王", (3.9, 3.9), ACCENT, DOWN),
                 ("猫", (1.3, 1.7), GOOD, UP), ("狗", (1.9, 1.1), GOOD, DOWN)]
        dots, labels = VGroup(), VGroup()
        for name, (x, y), col, d_dir in pairs:
            d = Dot(axes.c2p(x, y), color=col, radius=0.08)
            lab = cn(name, size=22, color=col).next_to(d, d_dir, buff=0.12)
            dots.add(d)
            labels.add(lab)
        note1 = VGroup(
            cn("2 个数 → 3 个数 → … 768 个数", size=24, color=FG),
            cn("意思近，点也近。", size=26, color=SUB),
        ).arrange(DOWN, buff=0.4).move_to(RIGHT * 3.2 + DOWN * 0.4)

        self.play(FadeIn(tag1, shift=DOWN * 0.1))
        self.play(Create(axes))
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.2),
                  LaggedStart(*[FadeIn(l) for l in labels], lag_ratio=0.2))
        self.play(FadeIn(note1, shift=UP * 0.2))
        self.wait(1.2)
        part1 = VGroup(tag1, axes, dots, labels, note1)
        self.play(FadeOut(part1))

        # ---- ② 矩阵乘法 ----
        tag2 = cn("② 矩阵乘法 = 各乘权重再相加", size=28, color=ACCENT)
        tag2.next_to(heading, DOWN, buff=0.4)
        ins = VGroup(*[cn(t, size=30) for t in ("2", "5", "1")])
        ins.arrange(DOWN, buff=0.7).move_to(LEFT * 4 + DOWN * 0.4)
        ws = ["×0.3", "×0.1", "×0.6"]
        summ = Circle(radius=0.5, stroke_color=GOOD, fill_color=BG, fill_opacity=1)
        summ.move_to(RIGHT * 0.5 + DOWN * 0.4)
        sigma = cn("Σ", size=36, color=GOOD).move_to(summ)
        arrows = VGroup()
        wlabels = VGroup()
        for x, w in zip(ins, ws):
            ar = Arrow(x.get_right(), summ.get_left(), buff=0.2,
                       color=SUB, stroke_width=3)
            wl = cn(w, size=22, color=ACCENT).next_to(ar.get_center(), UP, buff=0.08)
            arrows.add(ar)
            wlabels.add(wl)
        out_arrow = Arrow(summ.get_right(), summ.get_right() + RIGHT * 1.3,
                          buff=0.1, color=GOOD)
        out = cn("新的数", size=26, color=GOOD).next_to(out_arrow, RIGHT, buff=0.2)
        hl = cn("权重 = 参数 = 模型要学的东西", size=26, color=ACCENT)
        hl.to_edge(DOWN, buff=0.6)

        self.play(FadeIn(tag2, shift=DOWN * 0.1))
        self.play(LaggedStart(*[FadeIn(x) for x in ins], lag_ratio=0.2))
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.2),
                  LaggedStart(*[FadeIn(w) for w in wlabels], lag_ratio=0.2))
        self.play(FadeIn(summ), FadeIn(sigma))
        self.play(GrowArrow(out_arrow), FadeIn(out, shift=RIGHT * 0.2))
        self.play(Write(hl))
        self.wait(1.2)
        part2 = VGroup(tag2, ins, arrows, wlabels, summ, sigma, out_arrow, out, hl)
        self.play(FadeOut(part2))

        # ---- ③ 梯度（滚下山）----
        tag3 = cn("③ 梯度：训练 = 滚下山找谷底", size=28, color=ACCENT)
        tag3.next_to(heading, DOWN, buff=0.4)
        axes2, graph = valley()
        vgroup = VGroup(axes2, graph).move_to(DOWN * 0.5)
        x = ValueTracker(2.6)
        ball = always_redraw(
            lambda: Dot(axes2.c2p(x.get_value(), x.get_value() ** 2),
                        color=ACCENT, radius=0.12))
        bottom_lab = cn("谷底 = 错得最少", size=24, color=GOOD)
        bottom_lab.next_to(axes2.c2p(0, 0), DOWN, buff=0.3)

        self.play(FadeIn(tag3, shift=DOWN * 0.1))
        self.play(Create(axes2), Create(graph))
        self.add(ball)
        self.play(FadeIn(ball, scale=0.5))
        self.play(x.animate.set_value(0.0), run_time=2.6, rate_func=rate_functions.ease_in_out_sine)
        self.play(FadeIn(bottom_lab, shift=UP * 0.2))
        pad_to(self, DURATIONS["MathIntuitions"])


# ----------------------------------------------------------------------
# 镜 05 · 连接现实
# ----------------------------------------------------------------------
class Day1RealWorld(Scene):
    def construct(self):
        setup_bg(self)
        heading = cn("为什么各家模型扎堆发布？", size=32, color=FG).to_edge(UP, buff=0.6)

        releases = ["GPT-5.5", "Claude", "Gemini", "DeepSeek", "Qwen"]
        cards = VGroup(*[make_card(r, color=SUB, size=24, h=0.8) for r in releases])
        # 错落排布，像信息流里扎堆冒出来
        cards.arrange_in_grid(rows=2, cols=3, buff=0.5).move_to(UP * 0.6)
        cards[3].shift(RIGHT * 1.4)  # 第二排只有两个，往中间挪
        cards[4].shift(RIGHT * 1.4)

        self.play(FadeIn(heading, shift=DOWN * 0.2))
        self.play(LaggedStart(*[FadeIn(c, scale=0.6) for c in cards],
                              lag_ratio=0.25), run_time=2.0)

        line1 = cn("同一套配方：Transformer + 预训练 + 微调，且配方公开。",
                   size=26, color=GOOD)
        line2 = cn("算力 · 数据 · 调教 = 谁能往前冲一步", size=26, color=ACCENT)
        lines = VGroup(line1, line2).arrange(DOWN, buff=0.4).to_edge(DOWN, buff=0.7)

        self.play(Write(line1))
        self.play(FadeIn(line2, shift=UP * 0.2))
        pad_to(self, DURATIONS["Day1RealWorld"])


# ----------------------------------------------------------------------
# 镜 06 · 收束卡（主线收束大字，与第 8 天首尾呼应）
# ----------------------------------------------------------------------
class Day1Outro(Scene):
    def construct(self):
        setup_bg(self)
        outro_card(
            self,
            "所有 AI，归根结底都在预测下一个词。",
            keywords="预测下一个词 · 预训练 → 微调 · 同一个 Transformer 家族",
            hold=1.2,
        )
        pad_to(self, DURATIONS["Day1Outro"])
