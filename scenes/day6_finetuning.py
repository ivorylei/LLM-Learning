"""
第 6 天 · 微调 —— 7 镜。
脚本见 storyboards/day6_finetuning.md；风格/配色见 CLAUDE.md 第 6 节。
渲染示例：
    manim -ql scenes/day6_finetuning.py Day6Title
时长先用估值占位，配音(edge-tts)后再回填对齐 DURATIONS。
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _common import *  # noqa: E402,F401,F403

DURATIONS = {
    "Day6Title": 16.0,                 # 配音 15.48s
    "WhatIsFinetuning": 16.3,          # 配音 15.74s
    "ClassificationFinetune": 26.5,    # 配音 25.99s
    "InstructionTuning": 21.5,         # 配音 20.95s
    "BirthOfChatGPT": 29.1,            # 配音 28.58s
    "Day6RealWorld": 41.9,             # 配音 41.42s
    "Day6Outro": 14.5,                 # 配音 14.02s
}


def brain(size=1.4, color=SUB, fill="#1d3f4a", label="预训练大脑", lsize=22):
    """一个圆角『大脑』方块 + 标签。返回 VGroup[box, label]。"""
    box = RoundedRectangle(corner_radius=0.35, width=size * 1.5, height=size,
                           stroke_color=color, stroke_width=3,
                           fill_color=fill, fill_opacity=1)
    lab = cn(label, size=lsize, color=FG).move_to(box.get_center())
    return VGroup(box, lab)


class Day6Title(Scene):
    def construct(self):
        setup_bg(self)
        title_card(
            self,
            "微调",
            sub="从『会接话』到『听话』",
            intuition="指令微调，就是 ChatGPT 诞生的那一步。",
        )
        pad_to(self, DURATIONS["Day6Title"])


class WhatIsFinetuning(Scene):
    """微调的本质：大脑底子已成，少量数据轻轻掰方向。"""

    def construct(self):
        setup_bg(self)
        heading = cn("微调的本质", size=40, weight=BOLD).to_edge(UP, buff=0.5)
        self.play(Write(heading))

        # 预训练大脑（丰满）
        b = brain(size=1.8, color=SUB, label="已懂语言的大脑", lsize=26)
        b.move_to(LEFT * 3.2 + UP * 0.3)
        self.play(FadeIn(b, scale=0.85))

        # 对比数据量：预训练一大摞 vs 微调一小撮（标签放在上方，避免压住底部字幕）
        big = VGroup(*[Rectangle(width=1.5, height=0.15, fill_color=GOOD,
                                 fill_opacity=0.8, stroke_width=0) for _ in range(8)])
        big.arrange(UP, buff=0.05)
        big_lbl = cn("预训练：海量数据", size=20, color=GOOD).next_to(big, UP, buff=0.2)
        big_grp = VGroup(big, big_lbl).move_to(LEFT * 4.0 + DOWN * 1.9)

        small = VGroup(*[Rectangle(width=1.5, height=0.15, fill_color=ACCENT,
                                   fill_opacity=0.9, stroke_width=0) for _ in range(2)])
        small.arrange(UP, buff=0.05)
        small_lbl = cn("微调：一小撮数据", size=20, color=ACCENT).next_to(small, UP, buff=0.2)
        small_grp = VGroup(small, small_lbl).move_to(LEFT * 1.3 + DOWN * 2.2)

        self.play(FadeIn(big_grp, shift=UP * 0.2))
        self.play(FadeIn(small_grp, shift=UP * 0.2))
        self.wait(0.3)

        # 行为方向箭头被轻轻掰偏
        origin = RIGHT * 2.8 + UP * 0.3
        base_arrow = Arrow(origin, origin + RIGHT * 2.2 + UP * 0.0,
                           color="#5f7a80", stroke_width=5, buff=0)
        self.play(GrowArrow(base_arrow))
        bent = Arrow(origin, origin + RIGHT * 2.0 + UP * 1.0,
                     color=ACCENT, stroke_width=6, buff=0)
        bend_lbl = cn("行为被轻轻掰一掰", size=24, color=ACCENT).next_to(
            origin, UP, buff=1.4).shift(RIGHT * 1.0)
        self.play(Transform(base_arrow, bent), FadeIn(bend_lbl, shift=UP * 0.2))

        caption = cn("底子已经很好，只需很少数据和算力就见效。",
                     size=26, color=FG).to_edge(DOWN, buff=0.4)
        self.play(Write(caption))

        pad_to(self, DURATIONS["WhatIsFinetuning"])


class ClassificationFinetune(Scene):
    """分类微调：把『下一个词』输出头换成『是/否』两类。"""

    def construct(self):
        setup_bg(self)
        heading = cn("分类微调：做判断题", size=38, weight=BOLD).to_edge(UP, buff=0.5)
        self.play(Write(heading))

        # 一封邮件进入大脑
        mail = make_card("一封邮件：恭喜中奖，点此领取！", color=SUB, h=0.9, size=24)
        mail.move_to(UP * 1.7)
        b = brain(size=1.3, color=SUB, label="同一个大脑", lsize=24).move_to(DOWN * 0.1)
        arrow_in = Arrow(mail.get_bottom(), b.get_top(), color="#5f7a80",
                         stroke_width=4, buff=0.15)
        self.play(FadeIn(mail, shift=DOWN * 0.2))
        self.play(GrowArrow(arrow_in), FadeIn(b, scale=0.85))
        self.wait(0.3)

        # 原输出头：下一个词概率条
        head_old = VGroup(*[prob_bar(w, p, bar_max=1.8, size=22)
                            for w, p in [("领取", 0.4), ("中奖", 0.3), ("点击", 0.3)]])
        head_old.arrange(DOWN, buff=0.18).scale(0.8)
        head_old.next_to(b, DOWN, buff=0.6)
        old_lbl = cn("原：预测下一个词", size=22, color="#7FC6C0").next_to(head_old, LEFT, buff=0.4)
        self.play(FadeIn(head_old), FadeIn(old_lbl))
        self.wait(0.6)

        # 换成 是/否 两类的小头部
        head_new = VGroup(
            make_card("垃圾邮件 ✓", color=ACCENT, h=0.8, size=24),
            make_card("正常邮件", color="#5f7a80", h=0.8, size=24),
        ).arrange(DOWN, buff=0.3)
        head_new.move_to(head_old.get_center())
        new_lbl = cn("换成：是 / 否 两类", size=22, color=ACCENT)
        new_lbl.move_to(old_lbl.get_center())
        self.play(
            FadeTransform(head_old, head_new),
            FadeTransform(old_lbl, new_lbl),
        )
        self.wait(0.3)

        caption = cn("理解语言的能力直接复用，产出从续写变成分类。",
                     size=26, color=FG).to_edge(DOWN, buff=0.4)
        self.play(Write(caption))

        pad_to(self, DURATIONS["ClassificationFinetune"])


class InstructionTuning(Scene):
    """指令微调（重点）：成千上万条『指令→理想回答』样例。"""

    def construct(self):
        setup_bg(self)
        heading = cn("指令微调（今天的重点）", size=38, weight=BOLD,
                     color=ACCENT).to_edge(UP, buff=0.5)
        box = SurroundingRectangle(heading, color=ACCENT, buff=0.2,
                                   corner_radius=0.1)
        self.play(Write(heading), Create(box))

        # 成千上万条 指令—理想回答 样例
        samples = [
            ("把这句翻成英文：你好", "Hello"),
            ("总结这段话", "（给出要点…）"),
            ("写一句祝福", "祝你顺利！"),
        ]
        rows = VGroup()
        for instr, ans in samples:
            ins = make_card(instr, color=SUB, h=0.8, size=22)
            arr = cn("→", size=30, color=ACCENT)
            res = make_card(ans, color=GOOD, h=0.8, size=22)
            rows.add(VGroup(ins, arr, res).arrange(RIGHT, buff=0.3))
        rows.arrange(DOWN, buff=0.32, aligned_edge=LEFT).move_to(UP * 0.3)
        sub = cn("成千上万条「指令 → 理想回答」", size=24, color=SUB).next_to(
            rows, UP, buff=0.45)
        self.play(Write(sub))
        self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.2) for r in rows],
                              lag_ratio=0.3))
        self.wait(0.5)

        # 训练后：从乱续写 → 接到指令就好好回答
        before = make_card("续写机：问问题→乱接更多字", color="#5f7a80", h=0.9, size=22)
        after = make_card("助手：接到指令→给出合适回答", color=ACCENT, h=0.9, size=22)
        trans = VGroup(before, after).arrange(DOWN, buff=0.5).next_to(rows, DOWN, buff=0.6)
        self.play(FadeIn(before, shift=RIGHT * 0.2))
        big_arrow = Arrow(before.get_bottom(), after.get_top(), color=GOOD,
                          stroke_width=5, buff=0.1)
        self.play(GrowArrow(big_arrow), FadeIn(after, shift=LEFT * 0.2))

        pad_to(self, DURATIONS["InstructionTuning"])


class BirthOfChatGPT(Scene):
    """时间轴：GPT-3 续写机 → 指令微调+RLHF → ChatGPT 引爆。"""

    def construct(self):
        setup_bg(self)
        heading = cn("ChatGPT 诞生的那一步", size=38, weight=BOLD).to_edge(UP, buff=0.5)
        self.play(Write(heading))

        # 横轴时间线
        line = Line(LEFT * 5.5, RIGHT * 5.5, color="#3a5a62", stroke_width=3)
        line.move_to(UP * 0.3)
        self.play(Create(line))

        nodes = [
            (LEFT * 4.5, "GPT-3\n2020", "很强，但只是续写机", "#5f7a80"),
            (ORIGIN + UP * 0.3, "指令微调\n+ RLHF", "调教成会对话、肯帮忙", ACCENT),
            (RIGHT * 4.5, "ChatGPT\n2022 底", "引爆全球", GOOD),
        ]
        for i, (pos, title, desc, color) in enumerate(nodes):
            x = pos[0]
            p = np.array([x, 0.3, 0.0])
            dot = Dot(p, color=color, radius=0.12)
            tlab = cn(title, size=24, color=color).next_to(dot, UP, buff=0.3)
            dlab = cn(desc, size=20, color=FG).next_to(dot, DOWN, buff=0.4)
            self.play(FadeIn(dot, scale=0.6), Write(tlab), FadeIn(dlab, shift=UP * 0.15),
                      run_time=0.9)
            if i < 2:
                nxt = nodes[i + 1][0][0]
                arr = Arrow(np.array([x + 0.4, 0.3, 0]), np.array([nxt - 0.4, 0.3, 0]),
                            color="#5f7a80", stroke_width=4, buff=0)
                self.play(GrowArrow(arr), run_time=0.6)

        caption = cn("底层模型没变多少，变的是这一层调教。",
                     size=28, color=ACCENT).to_edge(DOWN, buff=0.5)
        self.play(Write(caption))

        pad_to(self, DURATIONS["BirthOfChatGPT"])


class Day6RealWorld(Scene):
    """对齐祛魅 + 开源生态树 + SFT vs RLHF 钩子。"""

    def construct(self):
        setup_bg(self)
        heading = cn("连接现实", size=40, weight=BOLD).to_edge(UP, buff=0.4)
        self.play(Write(heading))

        # —— 对齐祛魅 ——
        align = cn("「对齐」= 预训练后调教行为，让它有用、诚实、无害",
                   size=26, color=FG).move_to(UP * 2.0)
        self.play(Write(align))
        self.wait(0.5)

        # —— 开源生态树 ——
        root = make_card("开源底座 (Llama/Qwen)", color=GOOD, h=0.8, size=22)
        root.move_to(LEFT * 3.2 + DOWN * 0.2)
        leaves = VGroup()
        names = ["客服", "写代码", "医疗", "角色扮演", "翻译"]
        for i, nm in enumerate(names):
            leaf = make_card(nm, color=SUB, h=0.6, size=18)
            leaves.add(leaf)
        leaves.arrange(DOWN, buff=0.22).move_to(LEFT * 0.3 + DOWN * 0.2)
        self.play(FadeIn(root, scale=0.85))
        lines = VGroup(*[Line(root.get_right(), lf.get_left(), color="#3a5a62",
                              stroke_width=2) for lf in leaves])
        self.play(LaggedStart(*[Create(ln) for ln in lines], lag_ratio=0.15),
                  LaggedStart(*[FadeIn(lf, shift=RIGHT * 0.2) for lf in leaves],
                              lag_ratio=0.15))
        eco = cn("预训练贵→少数巨头\n微调便宜→人人能调", size=20, color=ACCENT)
        eco.next_to(leaves, RIGHT, buff=0.6)
        self.play(Write(eco))
        self.wait(0.5)

        # —— SFT vs RLHF 钩子 ——
        sft = cn("SFT：给标准答案让它模仿（今天）", size=22, color=SUB)
        rlhf = cn("RLHF：生成几个答案，评好坏再调（明天）", size=22, color="#7FC6C0")
        hooks = VGroup(sft, rlhf).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        hooks.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(hooks, shift=UP * 0.2))

        pad_to(self, DURATIONS["Day6RealWorld"])


class Day6Outro(Scene):
    def construct(self):
        setup_bg(self)
        outro_card(
            self,
            "同一个大脑，换一层调教，换一种行为。",
            keywords="底下那台机器，仍在预测下一个词。",
        )
        pad_to(self, DURATIONS["Day6Outro"])
