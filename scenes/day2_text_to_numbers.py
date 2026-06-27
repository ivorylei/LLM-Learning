"""
第 2 天 · 文字怎么变成数字 —— 8 镜。
脚本见 storyboards/day2_text_to_numbers.md；风格/配色见 CLAUDE.md 第 6 节。
渲染示例：
    manim -ql scenes/day2_text_to_numbers.py Day2Title
时长先用估值占位，配音(edge-tts)后再回填对齐 DURATIONS。
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _common import *  # noqa: E402,F401,F403

# 每镜目标时长 = edge-tts 实测配音时长 + 0.5s 收尾（见 audio/day2_*.mp3）
DURATIONS = {
    "Day2Title": 14.9,            # 配音 14.40s
    "Pipeline": 11.1,            # 配音 10.61s
    "Tokenization": 19.9,        # 配音 19.37s
    "TokenToID": 11.0,           # 配音 10.49s
    "Embedding": 26.9,           # 配音 26.40s
    "PositionalEncoding": 19.7,  # 配音 19.20s
    "Day2RealWorld": 37.0,       # 配音 36.46s
    "Day2Outro": 13.8,           # 配音 13.34s
}

STEPS = ["分词", "转 ID", "词嵌入", "位置编码"]


# ----------------------------------------------------------------------
# 镜 00 · 标题卡
# ----------------------------------------------------------------------
class Day2Title(Scene):
    def construct(self):
        setup_bg(self)
        title = cn("文字怎么变成数字", size=66, weight=BOLD)
        sub = cn("意义即位置", size=38, color=ACCENT)
        intuition = cn("计算机只会算数，第一步永远是把字变成空间里的点。",
                       size=26, color=SUB)
        group = VGroup(title, sub, intuition).arrange(DOWN, buff=0.5)

        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(sub, shift=UP * 0.2), run_time=0.7)
        self.play(FadeIn(intuition, shift=UP * 0.2), run_time=0.7)
        pad_to(self, DURATIONS["Day2Title"] - 0.6)
        self.play(FadeOut(group), run_time=0.6)


# ----------------------------------------------------------------------
# 镜 01 · 一条四步流水线
# ----------------------------------------------------------------------
class Pipeline(Scene):
    def construct(self):
        setup_bg(self)
        heading = cn("文字喂给模型前，走一条四步流水线", size=30, color=SUB)
        heading.to_edge(UP, buff=0.6)

        cards = VGroup(*[make_card(s, color=SUB, size=26, h=1.0) for s in STEPS])
        cards.arrange(RIGHT, buff=1.0).move_to(ORIGIN)
        arrows = VGroup(*[
            Arrow(cards[i].get_right(), cards[i + 1].get_left(),
                  buff=0.1, color="#3a5a62", stroke_width=4)
            for i in range(len(cards) - 1)
        ])

        text_in = cn("一段文字", size=26, color=FG).next_to(cards, LEFT, buff=0.9)
        vecs = VGroup(*[
            Rectangle(width=0.22, height=0.9, stroke_width=0,
                      fill_color=GOOD, fill_opacity=0.6 + 0.1 * i)
            for i in range(4)
        ]).arrange(RIGHT, buff=0.12).next_to(cards, RIGHT, buff=0.9)
        vec_lab = cn("一摞向量", size=22, color=GOOD).next_to(vecs, DOWN, buff=0.25)
        in_arrow = Arrow(text_in.get_right(), cards[0].get_left(),
                         buff=0.15, color="#3a5a62", stroke_width=4)
        out_arrow = Arrow(cards[-1].get_right(), vecs.get_left(),
                          buff=0.15, color="#3a5a62", stroke_width=4)

        # 整条流水线偏宽，统一缩放居中，避免两端出界
        body = VGroup(text_in, in_arrow, cards, arrows,
                      out_arrow, vecs, vec_lab)
        body.scale_to_fit_width(12.6).move_to(DOWN * 0.2)

        self.play(FadeIn(heading, shift=DOWN * 0.2))
        self.play(FadeIn(text_in), GrowArrow(in_arrow))
        for i, c in enumerate(cards):
            self.play(FadeIn(c, shift=RIGHT * 0.2), run_time=0.5)
            if i < len(arrows):
                self.play(GrowArrow(arrows[i]), run_time=0.35)
        self.play(GrowArrow(out_arrow),
                  LaggedStart(*[FadeIn(v, shift=UP * 0.1) for v in vecs],
                              lag_ratio=0.15),
                  FadeIn(vec_lab))
        # 逐个点亮，呼应“后面四镜一步步看”
        for c in cards:
            self.play(c[0].animate.set_stroke(ACCENT, width=4),
                      c[1].animate.set_color(ACCENT), run_time=0.35)
            self.play(c[0].animate.set_stroke(SUB, width=2.5),
                      c[1].animate.set_color(SUB), run_time=0.2)
        pad_to(self, DURATIONS["Pipeline"])


# ----------------------------------------------------------------------
# 镜 02 · 分词 Tokenization（BPE）
# ----------------------------------------------------------------------
class Tokenization(Scene):
    def construct(self):
        setup_bg(self)
        heading = cn("第一步：把文字切成小块（token）", size=30, color=SUB)
        heading.to_edge(UP, buff=0.6)

        word = cn("unbelievable", size=52).move_to(UP * 1.7)
        pieces = sentence_blocks(["un", "believ", "able"], width=2.0, height=0.9,
                                 size=30, stroke=ACCENT)
        pieces.move_to(UP * 0.4)

        note_token = cn("token ≠ 字，也 ≠ 词", size=26, color=FG).move_to(DOWN * 0.9)

        zh = sentence_blocks(["注意", "力", "机制"], width=1.5, height=0.8,
                             size=26, stroke=SUB).move_to(DOWN * 2.0)
        zh_lab = cn("中文同理，切成若干 token", size=22, color=SUB)
        zh_lab.next_to(zh, DOWN, buff=0.25)

        bpe = cn("BPE：一套有限小块，拼出几乎任何词（含没见过的新词）",
                 size=24, color=ACCENT).to_edge(DOWN, buff=0.4)

        self.play(FadeIn(heading, shift=DOWN * 0.2))
        self.play(Write(word))
        self.wait(0.5)
        # 切割：单词分裂成三块下移
        self.play(TransformFromCopy(word, pieces), run_time=1.2)
        self.play(FadeIn(note_token, shift=UP * 0.2))
        self.wait(0.5)
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.1) for b in zh],
                              lag_ratio=0.2), FadeIn(zh_lab))
        self.wait(0.4)
        self.play(Write(bpe))
        pad_to(self, DURATIONS["Tokenization"])


# ----------------------------------------------------------------------
# 镜 03 · 转成 ID
# ----------------------------------------------------------------------
class TokenToID(Scene):
    def construct(self):
        setup_bg(self)
        heading = cn("第二步：每个 token 查字典，得到一个编号", size=30, color=SUB)
        heading.to_edge(UP, buff=0.6)

        toks = ["un", "believ", "able"]
        ids = ["465", "1827", "390"]
        tok_blocks = sentence_blocks(toks, width=2.0, height=0.9, size=30,
                                     stroke=ACCENT).move_to(UP * 1.3)

        dict_box = make_card("模型字典（词表）", color=SUB, size=24, w=4.0, h=0.9)
        dict_box.move_to(ORIGIN)

        id_blocks = VGroup()
        for t, i in zip(tok_blocks, ids):
            b = make_card(i, color=GOOD, size=30, w=2.0, h=0.9)
            b.match_x(t)
            id_blocks.add(b)
        id_blocks.move_to(DOWN * 1.4)

        arrows_down = VGroup(*[
            Arrow(tok_blocks[k].get_bottom(), id_blocks[k].get_top(),
                  buff=0.15, color="#3a5a62", stroke_width=3)
            for k in range(len(toks))
        ])

        result = cn("一句话 → 一串数字  [465, 1827, 390]", size=28, color=FG)
        result.to_edge(DOWN, buff=0.6)

        self.play(FadeIn(heading, shift=DOWN * 0.2))
        self.play(LaggedStart(*[FadeIn(b) for b in tok_blocks], lag_ratio=0.15))
        self.play(FadeIn(dict_box, scale=0.9))
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows_down], lag_ratio=0.2),
                  LaggedStart(*[FadeIn(b, shift=DOWN * 0.1) for b in id_blocks],
                              lag_ratio=0.2))
        self.play(Write(result))
        pad_to(self, DURATIONS["TokenToID"])


# ----------------------------------------------------------------------
# 镜 04 · 词嵌入：意义即位置
# ----------------------------------------------------------------------
class Embedding(Scene):
    def construct(self):
        setup_bg(self)
        heading = cn("第三步：把编号变成一串小数（向量）", size=30, color=SUB)
        heading.to_edge(UP, buff=0.55)

        # 误解：编号 5 / 6 意思相近——划掉
        myth = cn("编号 5 和 6 ≠ 意思相近", size=30, color=FG).move_to(UP * 1.9)
        cross = Cross(myth, stroke_color=ACCENT, stroke_width=6)

        # 编号 → 向量
        id_b = make_card("465", color=GOOD, size=28, w=1.4, h=0.8).move_to(
            LEFT * 4 + UP * 0.4)
        vec = cn("[0.12, -0.83, 0.05, …]", size=26, color=SUB).next_to(
            id_b, RIGHT, buff=0.6)
        vec_arrow = Arrow(id_b.get_right(), vec.get_left(), buff=0.15,
                          color="#3a5a62", stroke_width=3)

        self.play(FadeIn(heading, shift=DOWN * 0.2))
        self.play(Write(myth))
        self.play(Create(cross))
        self.wait(0.4)
        self.play(FadeOut(myth), FadeOut(cross))
        self.play(FadeIn(id_b), GrowArrow(vec_arrow), Write(vec))
        self.wait(0.6)
        self.play(FadeOut(VGroup(id_b, vec_arrow, vec)))

        # 语义地图
        axes = Axes(x_range=[0, 6, 1], y_range=[0, 6, 1], x_length=5, y_length=4,
                    axis_config={"include_tip": True, "stroke_color": "#3a5a62"})
        axes.move_to(LEFT * 2.5 + DOWN * 0.4)
        pairs = [("国王", (4.5, 4.7), ACCENT, UP), ("女王", (4.0, 4.0), ACCENT, DOWN),
                 ("猫", (1.2, 1.8), GOOD, UP), ("狗", (1.8, 1.2), GOOD, DOWN),
                 ("巴黎", (4.7, 1.6), SUB, UP), ("伦敦", (4.2, 1.0), SUB, DOWN)]
        dots, labels, links = VGroup(), VGroup(), VGroup()
        for k in range(0, len(pairs), 2):
            (n1, p1, c1, d1) = pairs[k]
            (n2, p2, c2, d2) = pairs[k + 1]
            a = Dot(axes.c2p(*p1), color=c1, radius=0.08)
            b = Dot(axes.c2p(*p2), color=c2, radius=0.08)
            la = cn(n1, size=20, color=c1).next_to(a, d1, buff=0.1)
            lb = cn(n2, size=20, color=c2).next_to(b, d2, buff=0.1)
            link = Line(a.get_center(), b.get_center(),
                        stroke_color=c1, stroke_width=1.5)
            dots.add(a, b)
            labels.add(la, lb)
            links.add(link)

        big = cn("意义，被编码成了空间里的位置。", size=30, color=ACCENT)
        big.move_to(RIGHT * 3.2 + DOWN * 0.3)

        self.play(Create(axes))
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.1),
                  LaggedStart(*[FadeIn(l) for l in labels], lag_ratio=0.1))
        self.play(LaggedStart(*[Create(l) for l in links], lag_ratio=0.2))
        self.play(Write(big))
        pad_to(self, DURATIONS["Embedding"])


# ----------------------------------------------------------------------
# 镜 05 · 位置编码
# ----------------------------------------------------------------------
class PositionalEncoding(Scene):
    def construct(self):
        setup_bg(self)
        heading = cn("第四步：再加上「它排在第几位」", size=30, color=SUB)
        heading.to_edge(UP, buff=0.6)

        # 两句上下堆叠，标注放右侧，避免和词块重叠
        s1 = sentence_blocks(["狗", "咬", "人"], width=1.2, height=0.9, size=32)
        s2 = sentence_blocks(["人", "咬", "狗"], width=1.2, height=0.9, size=32)
        s1.move_to(UP * 1.9 + LEFT * 1.6)
        s2.move_to(DOWN * 0.3 + LEFT * 1.6)

        note = VGroup(
            cn("用的字完全一样", size=24, color=FG),
            cn("意思却相反", size=30, color=ACCENT),
        ).arrange(DOWN, buff=0.35).move_to(RIGHT * 3.7 + UP * 0.8)

        # 给两句都标序号 ①②③（位置信息）
        order = ["①", "②", "③"]
        tags = VGroup(*[cn(o, size=26, color=GOOD).next_to(blk, DOWN, buff=0.2)
                        for blk, o in zip(s1, order)])
        tags2 = VGroup(*[cn(o, size=26, color=GOOD).next_to(blk, DOWN, buff=0.2)
                         for blk, o in zip(s2, order)])

        concl = cn("加了位置信息，模型才分得清谁在前、谁在后", size=26, color=GOOD)
        concl.to_edge(DOWN, buff=0.7)

        self.play(FadeIn(heading, shift=DOWN * 0.2))
        self.play(FadeIn(s1, shift=UP * 0.1))
        self.play(FadeIn(s2, shift=UP * 0.1))
        self.play(FadeIn(note, shift=LEFT * 0.2))
        self.wait(0.6)
        self.play(LaggedStart(*[FadeIn(t, shift=UP * 0.1) for t in tags],
                              lag_ratio=0.2),
                  LaggedStart(*[FadeIn(t, shift=UP * 0.1) for t in tags2],
                              lag_ratio=0.2))
        self.play(Write(concl))
        pad_to(self, DURATIONS["PositionalEncoding"])


# ----------------------------------------------------------------------
# 镜 06 · 连接现实
# ----------------------------------------------------------------------
class Day2RealWorld(Scene):
    def construct(self):
        setup_bg(self)
        heading = cn("懂了原理，这些怪现象就不神秘了", size=30, color=FG)
        heading.to_edge(UP, buff=0.6)
        self.play(FadeIn(heading, shift=DOWN * 0.2))

        # ① 上下文窗口
        t1 = cn("① 上下文窗口 128K", size=30, color=ACCENT)
        d1 = cn("= 一次最多读多少个 token（token 数 ≠ 字数）", size=24, color=FG)
        g1 = VGroup(t1, d1).arrange(DOWN, buff=0.35).move_to(UP * 0.4)
        self.play(FadeIn(t1, shift=RIGHT * 0.2))
        self.play(FadeIn(d1, shift=UP * 0.1))
        self.wait(0.8)
        self.play(FadeOut(g1))

        # ② 容器装满，挤出最早
        t2 = cn("② 按 token 收费 · 长对话会忘事", size=30, color=ACCENT).move_to(UP * 1.2)
        box = Rectangle(width=4.2, height=0.9, stroke_color=SUB, stroke_width=2)
        toks = VGroup(*[
            Square(side_length=0.55, stroke_width=0, fill_color=GOOD,
                   fill_opacity=0.6).set_z_index(-1) for _ in range(6)
        ]).arrange(RIGHT, buff=0.1)
        toks.move_to(box.get_center())
        cont = VGroup(box, toks).next_to(t2, DOWN, buff=0.35)
        self.play(FadeIn(t2, shift=RIGHT * 0.2))
        self.play(Create(box), LaggedStart(*[FadeIn(s) for s in toks], lag_ratio=0.1))
        # 装满后，新 token 进来，最早的被挤出
        new_tok = Square(side_length=0.55, stroke_width=0, fill_color=ACCENT,
                         fill_opacity=0.8).next_to(box, RIGHT, buff=0.3)
        self.play(FadeIn(new_tok))
        self.play(toks[0].animate.shift(LEFT * 0.8).set_opacity(0.0),
                  toks[1:].animate.shift(LEFT * 0.65),
                  new_tok.animate.move_to(toks[-1].get_center() + RIGHT * 0.0))
        self.wait(0.5)
        self.play(FadeOut(VGroup(t2, cont, new_tok)))

        # ③ strawberry 数 r
        t3 = cn("③ strawberry 里有几个 r？", size=30, color=ACCENT).move_to(UP * 1.0)
        straw = sentence_blocks(["st", "raw", "berry"], width=2.0, height=0.8,
                                size=28, stroke=SUB).next_to(t3, DOWN, buff=0.4)
        d3 = cn("模型看到的是切好的 token 块，不是一个个字母 → 会数错",
                size=24, color=FG).to_edge(DOWN, buff=0.7)
        self.play(FadeIn(t3, shift=RIGHT * 0.2))
        self.play(LaggedStart(*[FadeIn(b) for b in straw], lag_ratio=0.2))
        self.play(FadeIn(d3, shift=UP * 0.1))
        pad_to(self, DURATIONS["Day2RealWorld"])


# ----------------------------------------------------------------------
# 镜 07 · 收束卡
# ----------------------------------------------------------------------
class Day2Outro(Scene):
    def construct(self):
        setup_bg(self)
        chain = cn("文字 → token → ID → 向量（空间里的点）", size=38, weight=BOLD)
        sub = cn("这摞带位置的向量，就是去预测下一个词的原料。", size=26, color=SUB)
        group = VGroup(chain, sub).arrange(DOWN, buff=0.6)

        self.play(Write(chain))
        self.play(FadeIn(sub, shift=UP * 0.2))
        pad_to(self, DURATIONS["Day2Outro"])
