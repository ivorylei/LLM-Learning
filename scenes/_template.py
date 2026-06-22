"""
模板 scene —— 统一项目的配色、中文字体与标题/收束卡的起手式。
也用来自检中文渲染是否正常：
    manim -ql scenes/_template.py Template
其他每一镜都可参考这里的常量与排版方式（详见 CLAUDE.md 第 6 节）。
注意：本项目用 Manim 社区版（manim-ce），`from manim import *`，命令是 `manim`（不是 manimgl）。
"""

from manim import *

# ---------- 项目调色板（与 PDF 品牌一致，见 CLAUDE.md 第 6 节）----------
BG = "#14323D"        # 背景（深青）
FG = "#EEF3F4"        # 主文字（浅）
ACCENT = "#C2683A"    # 强调/高亮（暖橙）
GOOD = "#4F8F7B"      # 正向/匹配（绿）
SUB = "#7FC6C0"       # 次要/标签（浅青）

CJK = "Noto Sans CJK SC"   # 中文字体（需系统已安装 fonts-noto-cjk）


def cn(text, size=42, color=FG, weight=NORMAL):
    """快捷创建中文文本，统一字体。"""
    return Text(text, font=CJK, font_size=size, color=color, weight=weight)


class Template(Scene):
    """标题卡 → 收束卡的最小示例，兼中文渲染自检。"""

    def construct(self):
        self.camera.background_color = BG

        # —— 标题卡 ——
        title = cn("注意力机制", size=64, weight=BOLD)
        subtitle = cn("整本书的心脏", size=34, color=ACCENT)
        intuition = cn("一句话：每个词环顾全场，把相关信息拉过来。", size=26, color=SUB)
        group = VGroup(title, subtitle, intuition).arrange(DOWN, buff=0.45)

        self.play(Write(title))
        self.play(FadeIn(subtitle, shift=UP * 0.2))
        self.play(FadeIn(intuition, shift=UP * 0.2))
        self.wait(1.5)
        self.play(FadeOut(group))

        # —— 词块排布示例（VGroup.arrange，勿用魔法坐标）——
        words = ["我", "去", "银行", "取钱"]
        blocks = VGroup()
        for w in words:
            box = RoundedRectangle(corner_radius=0.15, width=1.6, height=1.0,
                                   stroke_color=SUB, fill_color=BG, fill_opacity=1)
            label = cn(w, size=34)
            blocks.add(VGroup(box, label))
        blocks.arrange(RIGHT, buff=0.35)
        # 高亮"银行"
        blocks[2][0].set_stroke(ACCENT, width=4)
        blocks[2][1].set_color(ACCENT)

        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.2) for b in blocks], lag_ratio=0.2))
        self.wait(1)

        # —— 收束卡（呼应主线）——
        self.play(FadeOut(blocks))
        outro = cn("归根结底，还是为了——预测下一个词。", size=40, weight=BOLD)
        self.play(Write(outro))
        self.wait(2)
