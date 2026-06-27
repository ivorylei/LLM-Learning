"""
第 5 天 · 预训练 —— 8 镜。
脚本见 storyboards/day5_pretraining.md；风格/配色见 CLAUDE.md 第 6 节。
渲染示例：
    manim -ql scenes/day5_pretraining.py Day5Title
时长先用估值占位，配音(edge-tts)后再回填对齐 DURATIONS。
复用 _common.py 的 valley()（梯度山谷，承接第 1 天）与 prob_bar()（概率条）。
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _common import *  # noqa: E402,F401,F403

DURATIONS = {
    "Day5Title": 12.8,        # 配音 12.29s
    "TrainingLoop": 33.5,     # 配音 32.98s
    "WhyItLearns": 22.2,      # 配音 21.72s
    "SelfSupervised": 24.5,   # 配音 24.02s
    "Decoding": 27.3,         # 配音 26.81s
    "LoadWeights": 19.8,      # 配音 19.32s
    "Day5RealWorld": 34.7,    # 配音 34.18s
    "Day5Outro": 11.8,        # 配音 11.26s
}


class Day5Title(Scene):
    def construct(self):
        setup_bg(self)
        title_card(
            self,
            "预训练",
            sub="模型怎么学会整个世界",
            intuition="把半个互联网，压进几十亿个参数里。",
        )
        pad_to(self, DURATIONS["Day5Title"])


class TrainingLoop(Scene):
    """四步训练循环：前向 → 算损失 → 反向更新 → 重复。"""

    def construct(self):
        setup_bg(self)
        heading = cn("训练循环", size=40, weight=BOLD).to_edge(UP, buff=0.4)
        self.play(Write(heading))

        # 左半屏：四个节点排成一个环（上、右、下、左），环心在 LEFT*3.4
        ring_c = LEFT * 3.4 + DOWN * 0.3
        R = 1.5
        steps = [
            ("① 前向", ACCENT, ring_c + UP * R),
            ("② 算损失", GOOD, ring_c + RIGHT * R),
            ("③ 反向更新", SUB, ring_c + DOWN * R),
            ("④ 重复", FG, ring_c + LEFT * R),
        ]
        nodes = VGroup()
        for text, color, pos in steps:
            card = make_card(text, color=color, h=0.7, size=22)
            card.move_to(pos)
            nodes.add(card)

        # 环形箭头连接
        arrows = VGroup()
        for i in range(4):
            arr = CurvedArrow(
                nodes[i].get_center(), nodes[(i + 1) % 4].get_center(),
                angle=-TAU / 8, color="#3a5a62", stroke_width=3, tip_length=0.18,
            )
            arrows.add(arr)

        self.play(LaggedStart(*[FadeIn(n, scale=0.8) for n in nodes], lag_ratio=0.3))
        self.play(LaggedStart(*[Create(a) for a in arrows], lag_ratio=0.2))
        self.wait(0.4)

        # 右半屏：当前步骤的细节面板（每步淡入淡出，环始终保留）
        panel_c = RIGHT * 3.0 + DOWN * 0.2

        def focus(idx):
            return Indicate(nodes[idx], color=nodes[idx][1].get_color(),
                            scale_factor=1.15)

        # ① 前向：一段文本逐位置预测
        sent = sentence_blocks(["天", "气", "真", "好"], width=1.0, height=0.9, size=30)
        sent.scale(0.85)
        cap1 = cn("预测每个位置的下一个词", size=24, color=ACCENT)
        det1 = VGroup(sent, cap1).arrange(DOWN, buff=0.4).move_to(panel_c)
        self.play(focus(0))
        self.play(FadeIn(det1, shift=UP * 0.2))
        self.wait(0.6)
        self.play(FadeOut(det1))

        # ② 算损失：交叉熵数值条
        loss_lbl = cn("交叉熵损失", size=26, color=GOOD)
        loss_track = Rectangle(width=2.8, height=0.45, stroke_color="#3a5a62",
                               stroke_width=1.5, fill_opacity=0)
        loss_fill = Rectangle(width=2.6, height=0.45, stroke_width=0,
                              fill_color=ACCENT, fill_opacity=0.9)
        loss_fill.align_to(loss_track, LEFT)
        loss_bar = VGroup(loss_track, loss_fill)
        cap2 = cn("预测和真实差多少", size=22, color=SUB)
        det2 = VGroup(loss_lbl, loss_bar, cap2).arrange(DOWN, buff=0.3).move_to(panel_c)
        self.play(focus(1))
        self.play(FadeIn(det2, shift=UP * 0.2))
        self.wait(0.6)

        # ③ 反向更新：山谷曲线 + 小球下移（损失同步变短）
        axes, graph = valley(color=SUB, x_span=2.4, y_top=6.0)
        det3 = VGroup(axes, graph).scale(0.65).move_to(panel_c + DOWN * 0.1)
        t = ValueTracker(-2.0)
        ball = always_redraw(
            lambda: Dot(axes.c2p(t.get_value(), t.get_value() ** 2),
                        color=ACCENT, radius=0.1)
        )
        cap3 = cn("挪一小步，让损失变小", size=22, color=SUB).next_to(det3, DOWN, buff=0.25)
        self.play(focus(2))
        self.play(FadeOut(VGroup(loss_lbl, cap2)), FadeIn(det3), FadeIn(ball),
                  FadeIn(cap3))
        # loss_bar 移到山谷上方并随小球下降变短
        self.play(loss_bar.animate.next_to(det3, UP, buff=0.3))
        self.play(
            t.animate.set_value(-0.4),
            loss_fill.animate.stretch_to_fit_width(0.6).align_to(loss_track, LEFT),
            run_time=1.6,
        )
        self.wait(0.4)
        ball.clear_updaters()
        self.play(FadeOut(VGroup(det3, ball, cap3, loss_bar)))

        # ④ 重复：计数器（左固定增长，预留最大宽度避免压住「次」）
        rep_lbl = cn("重复", size=28, color=FG)
        counter = DecimalNumber(0, num_decimal_places=0, color=ACCENT, font_size=40,
                                edge_to_fix=LEFT)
        max_w = DecimalNumber(3_000_000_000, num_decimal_places=0,
                              font_size=40).width
        rep_unit = cn("次", size=28, color=FG)
        rep_lbl.move_to(panel_c + LEFT * (max_w / 2 + 0.9))
        counter.next_to(rep_lbl, RIGHT, buff=0.25)
        rep_unit.move_to(counter.get_left() + RIGHT * max_w + RIGHT * 0.3)
        det4 = VGroup(rep_lbl, counter, rep_unit)
        cnt = ValueTracker(0)
        counter.add_updater(lambda m: m.set_value(cnt.get_value()))
        self.play(focus(3))
        self.play(FadeIn(det4))
        # 环整体转一圈示意 + 计数飙升
        self.play(
            cnt.animate.set_value(3_000_000_000),
            Indicate(arrows, color=ACCENT, scale_factor=1.0),
            run_time=2.0,
        )
        counter.clear_updaters()
        note = cn("参数被一点点雕刻成形", size=26, color=SUB).to_edge(DOWN, buff=0.4)
        self.play(Write(note))

        pad_to(self, DURATIONS["TrainingLoop"])


class WhyItLearns(Scene):
    """猜词被逼出世界知识。"""

    def construct(self):
        setup_bg(self)
        heading = cn("为什么猜词能懂世界？", size=38, weight=BOLD).to_edge(UP, buff=0.55)
        self.play(Write(heading))

        # 残缺句子
        prompt = cn("水的沸点是 100 ___", size=46).move_to(UP * 1.4)
        self.play(Write(prompt))
        self.wait(0.4)

        answer = cn("摄氏度", size=46, color=GOOD).next_to(prompt, RIGHT, buff=0.2)
        # 让 ___ 处接上答案：直接放在句末右侧
        self.play(FadeIn(answer, shift=LEFT * 0.2))
        forced = cn("要接对，就得「知道」水和温度的关系", size=26, color=ACCENT)
        forced.next_to(prompt, DOWN, buff=0.5)
        self.play(Write(forced))
        self.wait(0.6)

        # 被逼学会的能力清单
        skills = ["语法", "事实", "推理", "常识"]
        chips = VGroup(*[make_card(s, color=SUB, h=0.8, size=28) for s in skills])
        chips.arrange(RIGHT, buff=0.5).next_to(forced, DOWN, buff=0.7)
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in chips], lag_ratio=0.25))

        caption = cn("没人专门教，是为了猜词被逼出来的。", size=28, color=FG)
        caption.to_edge(DOWN, buff=0.5)
        self.play(Write(caption))

        pad_to(self, DURATIONS["WhyItLearns"])


class SelfSupervised(Scene):
    """自监督：遮住后半句，文本自带答案。"""

    def construct(self):
        setup_bg(self)
        heading = cn("自监督", size=40, weight=BOLD).to_edge(UP, buff=0.55)
        self.play(Write(heading))

        full = ["猫", "坐", "在", "垫", "子", "上"]
        blocks = sentence_blocks(full, width=1.1, height=1.0, size=32)
        blocks.scale(0.95).move_to(UP * 0.7)
        self.play(FadeIn(blocks, shift=DOWN * 0.2))
        self.wait(0.3)

        # 前半句=题目，后半句=答案
        q_brace = Brace(VGroup(*blocks[:3]), DOWN, color=ACCENT)
        a_brace = Brace(VGroup(*blocks[3:]), DOWN, color=GOOD)
        q_lbl = cn("题目（给模型看）", size=24, color=ACCENT).next_to(q_brace, DOWN, buff=0.2)
        a_lbl = cn("答案（遮住，让它猜）", size=24, color=GOOD).next_to(a_brace, DOWN, buff=0.2)

        # 遮住后半句
        cover = Rectangle(
            width=VGroup(*blocks[3:]).width + 0.1,
            height=VGroup(*blocks[3:]).height + 0.1,
            fill_color="#1d3f4a", fill_opacity=0.92, stroke_color=GOOD, stroke_width=2,
        ).move_to(VGroup(*blocks[3:]).get_center())

        self.play(GrowFromCenter(q_brace), Write(q_lbl))
        self.play(GrowFromCenter(a_brace), Write(a_lbl), FadeIn(cover))
        self.wait(0.6)

        caption = cn("海量免费文本，瞬间都成了练习题——无需人工标注。",
                     size=28, color=FG).to_edge(DOWN, buff=0.45)
        self.play(Write(caption))

        pad_to(self, DURATIONS["SelfSupervised"])


class Decoding(Scene):
    """解码与温度：调温度让概率分布变尖/变平。"""

    def construct(self):
        setup_bg(self)
        heading = cn("解码：温度", size=40, weight=BOLD).to_edge(UP, buff=0.55)
        self.play(Write(heading))

        # 概率条组（低温：尖锐）
        words = ["好", "热", "棒", "怪"]
        low = [0.7, 0.15, 0.1, 0.05]
        high = [0.34, 0.26, 0.22, 0.18]

        bars = VGroup(*[prob_bar(w, p, bar_max=3.2) for w, p in zip(words, low)])
        bars.arrange(DOWN, buff=0.3, aligned_edge=LEFT).move_to(LEFT * 2.2 + DOWN * 0.3)
        self.play(LaggedStart(*[FadeIn(b, shift=RIGHT * 0.2) for b in bars], lag_ratio=0.2))

        # 温度滑块
        line = Line(LEFT * 1.2, RIGHT * 1.2, color="#3a5a62", stroke_width=4)
        slider = VGroup(line).move_to(RIGHT * 3.6 + UP * 1.2)
        knob = Dot(line.get_left(), color=ACCENT, radius=0.13)
        lo_lbl = cn("低温", size=22, color=SUB).next_to(line, LEFT, buff=0.2)
        hi_lbl = cn("高温", size=22, color=ACCENT).next_to(line, RIGHT, buff=0.2)
        temp_title = cn("temperature", size=24, color=FG).next_to(slider, UP, buff=0.3)
        self.play(FadeIn(slider), FadeIn(knob), Write(lo_lbl), Write(hi_lbl),
                  Write(temp_title))

        low_note = cn("保守、确定", size=24, color=SUB).next_to(slider, DOWN, buff=0.5)
        self.play(Write(low_note))
        self.wait(0.6)

        # 调高温度 → 分布变平
        new_bars = VGroup(*[prob_bar(w, p, bar_max=3.2) for w, p in zip(words, high)])
        for nb, ob in zip(new_bars, bars):
            nb.move_to(ob.get_center())
        high_note = cn("随机、有创意（也可能跑偏）", size=24, color=ACCENT)
        high_note.move_to(low_note.get_center())
        self.play(
            knob.animate.move_to(line.get_right()),
            Transform(bars, new_bars),
            FadeTransform(low_note, high_note),
            run_time=1.8,
        )
        self.wait(0.4)

        caption = cn("Top-k / Top-p：只在概率最高的几个候选里挑。",
                     size=26, color=FG).to_edge(DOWN, buff=0.4)
        self.play(Write(caption))

        pad_to(self, DURATIONS["Decoding"])


class LoadWeights(Scene):
    """下载现成 GPT-2 权重点亮本地模型。"""

    def construct(self):
        setup_bg(self)
        heading = cn("站在巨人肩上", size=40, weight=BOLD).to_edge(UP, buff=0.55)
        self.play(Write(heading))

        # 云端权重包
        cloud = VGroup(
            RoundedRectangle(corner_radius=0.3, width=3.4, height=1.6,
                             stroke_color=SUB, stroke_width=2.5,
                             fill_color="#1d3f4a", fill_opacity=1),
            cn("OpenAI\nGPT-2 权重", size=26, color=SUB),
        ).move_to(UP * 1.6)
        self.play(FadeIn(cloud, shift=DOWN * 0.2))

        # 本地小模型（未点亮）
        local = VGroup(
            RoundedRectangle(corner_radius=0.2, width=2.4, height=1.4,
                             stroke_color="#3a5a62", stroke_width=2.5,
                             fill_color=BG, fill_opacity=1),
            cn("你的本地模型", size=24, color="#5f7a80"),
        ).move_to(DOWN * 1.7)
        self.play(FadeIn(local))

        # 下载箭头
        arrow = Arrow(cloud.get_bottom(), local.get_top(), color=ACCENT,
                      stroke_width=5, buff=0.15)
        dl = cn("下载", size=24, color=ACCENT).next_to(arrow, RIGHT, buff=0.2)
        self.play(GrowArrow(arrow), Write(dl))

        # 点亮本地模型
        self.play(
            local[0].animate.set_stroke(GOOD, width=3).set_fill("#1d3f4a", opacity=1),
            local[1].animate.set_color(FG),
            Flash(local.get_center(), color=GOOD, line_length=0.3, num_lines=16),
        )
        caption = cn("别人花大钱训好的大脑，直接下载来用。", size=28, color=FG)
        caption.to_edge(DOWN, buff=0.45)
        self.play(Write(caption))

        pad_to(self, DURATIONS["LoadWeights"])


class Day5RealWorld(Scene):
    """规模法则 + base model 很怪。"""

    def construct(self):
        setup_bg(self)
        heading = cn("连接现实", size=40, weight=BOLD).to_edge(UP, buff=0.5)
        self.play(Write(heading))

        # —— 第一部分：规模法则 ——
        knobs_lbl = VGroup(*[cn(s, size=24, color=SUB) for s in ["参数", "数据", "算力"]])
        knobs = VGroup()
        for lbl in knobs_lbl:
            ring = Circle(radius=0.35, stroke_color=ACCENT, stroke_width=4, fill_opacity=0)
            pointer = Line(ring.get_center(), ring.get_center() + UP * 0.3,
                           color=ACCENT, stroke_width=4)
            knob = VGroup(ring, pointer, lbl)
            lbl.next_to(ring, DOWN, buff=0.15)
            knobs.add(knob)
        knobs.arrange(RIGHT, buff=0.7).move_to(LEFT * 3.6 + UP * 0.6)

        # 下降的损失曲线
        ax = Axes(x_range=[0, 5, 1], y_range=[0, 5, 1], x_length=4.2, y_length=2.8,
                  axis_config={"include_tip": False, "include_numbers": False,
                               "stroke_color": "#3a5a62"})
        curve = ax.plot(lambda x: 4.2 * np.exp(-0.7 * x) + 0.4, x_range=[0.05, 5],
                        color=GOOD)
        curve.set_stroke(width=5)
        loss_grp = VGroup(ax, curve).move_to(RIGHT * 3.0 + UP * 0.6)
        loss_cap = cn("损失可预测地下降", size=22, color=GOOD).next_to(loss_grp, DOWN, buff=0.15)

        self.play(LaggedStart(*[FadeIn(k) for k in knobs], lag_ratio=0.2))
        self.play(Create(ax), Create(curve), Write(loss_cap))
        # 旋钮拧大
        self.play(*[Rotate(k[1], angle=-PI * 0.7, about_point=k[0].get_center())
                    for k in knobs], run_time=1.2)
        law = cn("规模法则：越大越强，甚至「涌现」新能力", size=24, color=ACCENT)
        law.next_to(knobs, DOWN, buff=0.7)
        self.play(Write(law))
        self.wait(0.6)

        self.play(FadeOut(VGroup(knobs, loss_grp, loss_cap, law)))

        # —— 第二部分：base model 很怪 ——
        weird_title = cn("但「基础模型」很怪：", size=30, color=FG).move_to(UP * 1.5)
        self.play(Write(weird_title))

        you = make_card("你：今天天气怎么样？", color=SUB, h=0.9, size=26)
        you.move_to(UP * 0.4)
        it = make_card("它：……明天呢？后天呢？", color=ACCENT, h=0.9, size=26)
        it.move_to(DOWN * 0.7)
        self.play(FadeIn(you, shift=RIGHT * 0.2))
        self.play(FadeIn(it, shift=LEFT * 0.2))
        weird_note = cn("不回答，反而续写出更多问题——它还没「懂事」",
                        size=24, color=SUB).next_to(it, DOWN, buff=0.5)
        self.play(Write(weird_note))

        tail = cn("规模红利递减 → 前沿转向别处（第 8 天接上）",
                  size=22, color="#7FC6C0").to_edge(DOWN, buff=0.35)
        self.play(FadeIn(tail))

        pad_to(self, DURATIONS["Day5RealWorld"])


class Day5Outro(Scene):
    def construct(self):
        setup_bg(self)
        outro_card(
            self,
            "半个互联网 → 压进参数",
            keywords="压缩的方式，还是——预测下一个词。",
        )
        pad_to(self, DURATIONS["Day5Outro"])
