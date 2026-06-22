#!/usr/bin/env bash
# 一次性环境安装：Manim 社区版 + 中文字体 + ffmpeg + 配音
# 用法： bash setup.sh
set -e

echo "==> [1/4] 系统依赖（ffmpeg / 中文字体 / Manim 渲染所需库）"
# Debian/Ubuntu。其他发行版请用对应包管理器，或参考 docs.manim.community 安装指南。
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y \
    ffmpeg \
    fonts-noto-cjk fonts-noto-cjk-extra \
    build-essential pkg-config python3-dev python3-pip python3-venv \
    libcairo2-dev libpango1.0-dev
  # LaTeX 仅供 MathTex 数学公式使用（中文不需要它）。如暂不画公式可跳过这步以省时间/空间。
  sudo apt-get install -y texlive texlive-latex-extra texlive-fonts-extra dvisvgm || \
    echo "（LaTeX 安装跳过/失败：不影响中文 Text 渲染，仅影响数学公式 MathTex）"
elif command -v brew >/dev/null 2>&1; then
  brew install ffmpeg pango cairo pkg-config
  brew install --cask font-noto-sans-cjk-sc || true
  echo "macOS 如需公式，请另装 MacTeX（较大）：brew install --cask mactex-no-gui"
else
  echo "未检测到 apt 或 brew。请手动安装：ffmpeg、Noto CJK 字体、cairo/pango 开发库；"
  echo "Manim 安装指南见 https://docs.manim.community/en/stable/installation.html"
fi

echo "==> [2/4] 创建并启用 Python 虚拟环境 .venv"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

echo "==> [3/4] 安装 Manim 社区版 + 配音（edge-tts，免费无需 key）+ 动画-配音对齐插件"
pip install manim
pip install edge-tts
pip install "manim-voiceover[edge]" || pip install manim-voiceover || true

echo "==> [4/4] 自检"
echo "----- manim 版本 -----"; manim --version || echo "manim 未就绪，请检查上面安装日志"
echo "----- ffmpeg -----"; ffmpeg -version | head -1 || echo "ffmpeg 未就绪"
echo "----- 中文字体 -----"; fc-list 2>/dev/null | grep -i "Noto Sans CJK SC" | head -1 || echo "未找到 Noto Sans CJK SC，请确认字体已安装"
echo "----- edge-tts -----"; python -c "import edge_tts; print('edge-tts OK')" || echo "edge-tts 未就绪"

cat <<'EOF'

✅ 安装流程结束。

下次进入项目记得先启用虚拟环境：
    source .venv/bin/activate

然后启动 Claude Code：
    claude

快速验证中文渲染（可选）：
    manim -ql scenes/_template.py Template
EOF
