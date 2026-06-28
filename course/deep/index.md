# 从零理解大语言模型 · 代码精讲册

> 跟着 Sebastian Raschka《从零构建大语言模型（LLM）》，**逐章读懂每一段代码、并能自己改着玩**。
> 这是 [8 天概念课](/?v=course) 的"第二层"——概念课讲清八个大想法，本册把它们**坐实成能跑的代码**。

---

## 这册和你想的不一样

它**不要求你能从零默写代码**。它要你做到两件事：

1. **读懂**——书里每一段代码，知道它在干什么、为什么这么写、数据的形状怎么变。
2. **能改**——能自己改参数、改一两行逻辑，预测结果会怎样，跑一下验证。

为达到这两点，**在卡住的地方就地补 Python / PyTorch 基础**（不另开一门枯燥语法课）。写不出整个类没关系，但**每一行你都该能说出"它在算什么"**——这就是及格线。

!!! tip "怎么用这一册"
    每章都配 GitHub 上**能在 Colab 一键运行的 notebook**。**读到本册某段代码，就去 notebook 里跑同一段、试试每节的「🔧 亲手改一下」。** 本册是"导游词"，notebook 是"现场"，两个一起用。

---

## 章节地图

| 本册章 | 对应书 | 你将亲手得到 | 在 Colab 跑 |
|---|---|---|---|
| [第 0 章 · Python/PyTorch 加油站](ch00_python_pytorch.md) | 附录 A | 一个完整跑通的训练循环（全书母版）| [Colab ▶](https://colab.research.google.com/github/rasbt/LLMs-from-scratch/blob/main/appendix-A/01_main-chapter-code/code-part1.ipynb) |
| [第 1 章 · 大地图（代码层）](ch01_overview.md) | 第 1 章 | 一张"造什么、分几步造"的施工图 | —（概念导览） |
| [第 2 章 · 文字变数字](ch02_text_data.md) | 第 2 章 | 文本 → token → 嵌入向量的输入流水线 | [Colab ▶](https://colab.research.google.com/github/rasbt/LLMs-from-scratch/blob/main/ch02/01_main-chapter-code/ch02.ipynb) |
| [第 3 章 · 注意力机制 ★](ch03_attention.md) | 第 3 章 | 从零搭出 `MultiHeadAttention`，逐行能讲 | [Colab ▶](https://colab.research.google.com/github/rasbt/LLMs-from-scratch/blob/main/ch03/01_main-chapter-code/ch03.ipynb) |
| [第 4 章 · 拼装 GPT](ch04_gpt.md) | 第 4 章 | 一个完整、能生成文本的 GPT 模型 | [Colab ▶](https://colab.research.google.com/github/rasbt/LLMs-from-scratch/blob/main/ch04/01_main-chapter-code/ch04.ipynb) |
| [第 5 章 · 预训练](ch05_pretrain.md) | 第 5 章 | 训练循环 + 解码策略 + 加载 GPT-2 权重 | [Colab ▶](https://colab.research.google.com/github/rasbt/LLMs-from-scratch/blob/main/ch05/01_main-chapter-code/ch05.ipynb) |
| [第 6 章 · 分类微调](ch06_classify.md) | 第 6 章 | 把 GPT 改成垃圾邮件分类器 | [Colab ▶](https://colab.research.google.com/github/rasbt/LLMs-from-scratch/blob/main/ch06/01_main-chapter-code/ch06.ipynb) |
| [第 7 章 · 指令微调](ch07_instruct.md) | 第 7 章 | 把"续写机"调成"听话的问答助手" | [Colab ▶](https://colab.research.google.com/github/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/ch07.ipynb) |
| [附录 D · 训练技巧](appD_training_tricks.md) | 附录 D | 学习率热身 / 余弦衰减 / 梯度裁剪 | [Colab ▶](https://colab.research.google.com/github/rasbt/LLMs-from-scratch/blob/main/appendix-D/01_main-chapter-code/appendix-D.ipynb) |
| [附录 E · LoRA](appE_lora.md) | 附录 E | 用一张消费级显卡微调大模型 | [Colab ▶](https://colab.research.google.com/github/rasbt/LLMs-from-scratch/blob/main/appendix-E/01_main-chapter-code/appendix-E.ipynb) |
| [第 8 章 · 综合](ch08_synthesis.md) | 全书综合 | 四个抽屉装下整个 AI 世界（无新代码）| — |

> 从文字到微调，10 章把骨架逐行拆齐；第 8 章把它收成一套判断力，完整闭环。

---

## 一条主线，请带走

整本书、整个 AI 领域，归根结底都在围绕一件事：

> **预测下一个词。**

你现在用 ChatGPT，它回答你的本质就是——不停地预测下一个词。把这条线攥紧，本册里所有的张量、矩阵乘、注意力、训练循环，都会归位。

---

!!! note "新手只需要会一个操作"
    零基础不用装任何软件：打开 [Google Colab](https://colab.research.google.com)，点代码格子左边的**播放按钮**让它运行，看下面冒出结果。会这个，就够开始了。第 0 章会带你把工具一次性摸熟。
