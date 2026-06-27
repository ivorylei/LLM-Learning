"""
第 3 天 · 注意力机制（试点）—— 7 镜。
脚本见 storyboards/day3_attention.md；风格/配色见 CLAUDE.md 第 6 节。
渲染示例：
    manim -ql scenes/day3_attention.py Day3Title
    manim -ql scenes/day3_attention.py BankAmbiguity
等时长先用估值占位，配音(edge-tts)后再回填对齐。
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _common import *  # noqa: E402,F401,F403

# 每镜目标时长 = edge-tts 实测配音时长 + 0.5s 收尾（见 audio/day3_*.mp3）
DURATIONS = {
    "Day3Title": 12.5,
    "BankAmbiguity": 17.1,
    "QKVRoles": 23.3,
    "AttentionWeights": 21.3,
    "CausalMask": 20.3,
    "MultiHead": 15.8,
    "Day3Outro": 13.3,
}


# ----------------------------------------------------------------------
# 镜 00 · 标题卡
# ----------------------------------------------------------------------
class Day3Title(Scene):
    def construct(self):
        setup_bg(self)
        title = cn("注意力机制", size=72, weight=BOLD)
        sub = cn("整本书的心脏", size=36, color=ACCENT)
        intuition = cn("一句话：每个词环顾全场，把相关信息拉过来。", size=28, color=SUB)
        group = VGroup(title, sub, intuition).arrange(DOWN, buff=0.5)

        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(sub, shift=UP * 0.2), run_time=0.7)
        self.play(FadeIn(intuition, shift=UP * 0.2), run_time=0.7)
        pad_to(self, DURATIONS["Day3Title"] - 0.6)
        self.play(FadeOut(group), run_time=0.6)


# ----------------------------------------------------------------------
# 镜 01 · 一词多义
# ----------------------------------------------------------------------
class BankAmbiguity(Scene):
    def construct(self):
        setup_bg(self)
        heading = cn("同一个词，意思天差地别", size=30, color=SUB).to_edge(UP, buff=0.6)

        s1 = cn("我去银行取钱", size=46)
        s1[2:4].set_color(ACCENT)          # “银行”
        s2 = cn("我坐在河的银行边", size=46)
        s2[5:7].set_color(ACCENT)          # “银行”
        sentences = VGroup(s1, s2).arrange(DOWN, buff=1.6).move_to(UP * 0.6)

        lbl1 = cn("金融机构", size=28, color=GOOD).next_to(s1[2:4], UP, buff=0.35)
        lbl2 = cn("河岸", size=28, color=SUB).next_to(s2[5:7], DOWN, buff=0.35)

        qmark = cn("？", size=110, color=ACCENT)
        caption = cn("模型怎么知道该取哪个意思？", size=28, color=FG)
        qgroup = VGroup(qmark, caption).arrange(DOWN, buff=0.25).to_edge(DOWN, buff=0.5)

        self.play(FadeIn(heading, shift=DOWN * 0.2))
        self.play(Write(s1))
        self.play(Write(s2))
        self.wait(0.5)
        self.play(FadeIn(lbl1, shift=DOWN * 0.2), FadeIn(lbl2, shift=UP * 0.2))
        self.wait(0.8)
        self.play(FadeIn(qmark, scale=0.6), Write(caption))
        pad_to(self, DURATIONS["BankAmbiguity"])


# ----------------------------------------------------------------------
# 镜 02 · 三个角色 Q/K/V
# ----------------------------------------------------------------------
class QKVRoles(Scene):
    def construct(self):
        setup_bg(self)
        heading = cn("记住三个角色", size=32, color=FG).to_edge(UP, buff=0.6)

        blocks = sentence_blocks(["我", "去", "银行", "取钱"]).move_to(UP * 1.6)
        bank = blocks[2]
        bank[0].set_stroke(ACCENT, width=4)
        bank[1].set_color(ACCENT)

        # 银行 头上的 Query 标签
        q_tag = VGroup(
            RoundedRectangle(corner_radius=0.1, width=1.0, height=0.55,
                             stroke_color=ACCENT, fill_color=ACCENT, fill_opacity=0.2),
            cn("Q", size=28, color=ACCENT, weight=BOLD),
        )
        q_tag.next_to(bank, UP, buff=0.3)

        # 每个词下面挂 K / V 小芯片
        kv_rows = VGroup()
        for b in blocks:
            k = cn("K", size=22, color=SUB)
            v = cn("V", size=22, color=GOOD)
            row = VGroup(k, v).arrange(RIGHT, buff=0.3).next_to(b, DOWN, buff=0.25)
            kv_rows.add(row)

        # 底部图例
        legend = VGroup(
            cn("Query（查询）= 我在找什么", size=26, color=ACCENT),
            cn("Key（键）= 我这儿有什么", size=26, color=SUB),
            cn("Value（值）= 选中我，就把这些给你", size=26, color=GOOD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).to_edge(DOWN, buff=0.7)

        # 节奏跟着旁白走：逐个角色揭示，每段后停一拍
        self.play(FadeIn(heading, shift=DOWN * 0.2))
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.2) for b in blocks], lag_ratio=0.2))
        self.wait(1.5)  # “记住三个角色”
        self.play(FadeIn(q_tag, shift=DOWN * 0.2), Write(legend[0]))
        self.play(Indicate(q_tag, color=ACCENT, scale_factor=1.2))
        self.wait(2.0)  # Query
        self.play(LaggedStart(*[FadeIn(r[0]) for r in kv_rows], lag_ratio=0.15), Write(legend[1]))
        self.wait(2.5)  # Key
        self.play(LaggedStart(*[FadeIn(r[1]) for r in kv_rows], lag_ratio=0.15), Write(legend[2]))
        self.wait(2.0)  # Value
        # “全书最该记住的一组比喻”——给图例描一圈
        self.play(Circumscribe(legend, color=FG, run_time=2.0))
        pad_to(self, DURATIONS["QKVRoles"])


# ----------------------------------------------------------------------
# 镜 03 · 匹配与加权
# ----------------------------------------------------------------------
class AttentionWeights(Scene):
    def construct(self):
        setup_bg(self)
        heading = cn("谁匹配得好，就分到更多注意力", size=30, color=SUB).to_edge(UP, buff=0.5)

        words = ["我", "去", "银行", "取钱"]
        weights = [0.10, 0.05, 0.10, 0.75]
        blocks = sentence_blocks(words).move_to(UP * 1.3)
        bank = blocks[2]
        bank[0].set_stroke(ACCENT, width=4)
        bank[1].set_color(ACCENT)

        # 从“银行”向每个词画弧线，强度=粗细/颜色
        arcs = VGroup()
        for i, b in enumerate(blocks):
            if i == 2:
                continue
            strong = i == 3
            arc = ArcBetweenPoints(
                bank.get_top(), b.get_top(),
                angle=-PI / 2 if i < 2 else -PI / 2,
            )
            arc.set_stroke(
                color=GOOD if strong else SUB,
                width=8 if strong else 2,
                opacity=1.0 if strong else 0.4,
            )
            arcs.add(arc)

        # 热力图：每个词一格，颜色深浅=权重；格子下标注对应的词
        cells = VGroup()
        cell_labels = VGroup()
        for b, w, word in zip(blocks, weights, words):
            cell = Square(side_length=0.8, stroke_color=SUB, stroke_width=1.5)
            cell.set_fill(GOOD, opacity=max(0.12, w))  # 弱项也留一点可见度
            cell.next_to(b, DOWN, buff=1.4)
            cell.match_x(b)
            lab = cn(word, size=22, color=FG).next_to(cell, DOWN, buff=0.18)
            cells.add(cell)
            cell_labels.add(lab)
        hm_label = cn("注意力权重", size=24, color=SUB).next_to(cell_labels, DOWN, buff=0.3)

        result = cn("金融机构", size=30, color=GOOD).next_to(bank, UP, buff=0.4)

        # 节奏跟着旁白走
        self.play(FadeIn(heading, shift=DOWN * 0.2))
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.2) for b in blocks], lag_ratio=0.15))
        self.wait(1.0)  # “银行拿自己的 Query，去和每个词的 Key 比对”
        self.play(LaggedStart(*[Create(a) for a in arcs], lag_ratio=0.25), run_time=2.5)
        self.wait(1.5)  # “谁匹配得好，就分到更多注意力”
        self.play(LaggedStart(*[FadeIn(c) for c in cells], lag_ratio=0.15),
                  LaggedStart(*[FadeIn(l) for l in cell_labels], lag_ratio=0.15),
                  FadeIn(hm_label))
        self.wait(1.0)
        self.play(Indicate(cells[3], color=GOOD, scale_factor=1.25))  # “取钱”最强
        self.wait(1.5)  # “大量吸收取钱的值”
        self.play(FadeIn(result, shift=UP * 0.2))
        self.play(Indicate(result, color=GOOD))
        pad_to(self, DURATIONS["AttentionWeights"])


# ----------------------------------------------------------------------
# 镜 04 · 因果掩码
# ----------------------------------------------------------------------
class CausalMask(Scene):
    def construct(self):
        setup_bg(self)
        heading = cn("预测下一个词时，不能偷看未来", size=30, color=ACCENT).to_edge(UP, buff=0.5)

        nums = [str(i) for i in range(1, 8)]
        blocks = sentence_blocks(nums, width=1.0, height=1.0, size=30).move_to(UP * 1.3)
        # 当前处理第 5 个
        cur = blocks[4]
        cur[0].set_stroke(GOOD, width=4)
        cur[1].set_color(GOOD)
        cur_tag = cn("正在处理", size=22, color=GOOD).next_to(cur, UP, buff=0.25)

        # 遮住第 6、7（未来）
        masks = VGroup()
        for j in (5, 6):
            cover = Rectangle(width=1.0, height=1.0,
                              stroke_width=0, fill_color="#000000", fill_opacity=0.6)
            cover.move_to(blocks[j])
            x = Cross(stroke_color="#888888").scale(0.35).move_to(blocks[j])
            masks.add(VGroup(cover, x))

        # 简化注意力网格：上三角（未来）打叉
        n = 5
        grid = VGroup()
        for r in range(n):
            for c in range(n):
                cell = Square(side_length=0.5, stroke_color=SUB, stroke_width=1)
                if c <= r:
                    cell.set_fill(GOOD, opacity=0.55)   # 允许看（含自己）
                else:
                    cell.set_fill(BG, opacity=1)         # 未来：屏蔽
                grid.add(cell)
        grid.arrange_in_grid(n, n, buff=0).to_edge(DOWN, buff=0.7)
        # 给被屏蔽的格子打叉
        crosses = VGroup()
        for r in range(n):
            for c in range(n):
                if c > r:
                    crosses.add(Cross(stroke_color="#5b7b80", stroke_width=2)
                                .scale(0.16).move_to(grid[r * n + c]))
        grid_label = cn("因果掩码", size=24, color=FG).next_to(grid, UP, buff=0.25)

        self.play(FadeIn(heading, shift=DOWN * 0.2))
        self.play(LaggedStart(*[FadeIn(b) for b in blocks], lag_ratio=0.1))
        self.play(FadeIn(cur_tag, shift=DOWN * 0.2))
        self.wait(0.3)
        self.play(LaggedStart(*[FadeIn(m) for m in masks], lag_ratio=0.2))
        self.wait(0.5)
        self.play(FadeIn(grid), FadeIn(grid_label))
        self.play(LaggedStart(*[Create(x) for x in crosses], lag_ratio=0.05))
        pad_to(self, DURATIONS["CausalMask"])


# ----------------------------------------------------------------------
# 镜 05 · 多头注意力
# ----------------------------------------------------------------------
class MultiHead(Scene):
    def construct(self):
        setup_bg(self)
        heading = cn("同时用很多套注意力，各看各的关系", size=30, color=SUB).to_edge(UP, buff=0.5)

        words = ["我", "去", "银行", "取钱"]
        head_specs = [("语法", ACCENT, (0, 1)), ("指代", SUB, (2, 0)), ("主题", GOOD, (2, 3))]

        heads = VGroup()
        for name, color, (a, b) in head_specs:
            row = sentence_blocks(words, width=1.0, height=0.7, size=24, stroke="#3a5a62")
            arc = ArcBetweenPoints(row[a].get_top(), row[b].get_top(), angle=-PI / 2)
            arc.set_stroke(color=color, width=5)
            tag = cn(name, size=24, color=color)
            unit = VGroup(tag, VGroup(row, arc)).arrange(RIGHT, buff=0.5)
            heads.add(unit)
        heads.arrange(DOWN, buff=0.55, aligned_edge=LEFT).move_to(UP * 0.6)

        out = VGroup(
            RoundedRectangle(corner_radius=0.15, width=3.2, height=0.9,
                             stroke_color=FG, fill_color=BG, fill_opacity=1),
            cn("合并后的理解", size=26, color=FG),
        ).to_edge(DOWN, buff=0.7)
        arrow = Arrow(heads.get_bottom(), out.get_top(), buff=0.2, color=SUB)

        self.play(FadeIn(heading, shift=DOWN * 0.2))
        for unit in heads:
            self.play(FadeIn(unit[0]), Create(unit[1][1]),
                      FadeIn(unit[1][0], shift=UP * 0.1), run_time=0.9)
        self.wait(0.4)
        self.play(GrowArrow(arrow), FadeIn(out, shift=UP * 0.2))
        pad_to(self, DURATIONS["MultiHead"])


# ----------------------------------------------------------------------
# 镜 06 · 收束卡
# ----------------------------------------------------------------------
class Day3Outro(Scene):
    def construct(self):
        setup_bg(self)
        outro_card(
            self,
            "归根结底，还是为了——预测下一个词。",
            keywords="环顾全场 · Query / Key / Value · 因果掩码 · 多头",
            hold=1.0,
        )
        pad_to(self, DURATIONS["Day3Outro"])
