# 深化册 · 第 4 章　拼装 GPT：把"积木"一块块堆成一个完整模型

> **对应**：概念课 **第 4 天** ｜ 书 **第 4 章（从头实现一个 GPT 模型来生成文本）**
> **一句话直觉**：一个 GPT 就是把"注意力 + 前馈"打包成一块**积木**，重复堆 N 层；所谓"参数"，就是这一大堆积木里那些**可调的数字**。
> **这一章你将亲手得到**：一个**完整、能跑出（乱码）文本**的 GPT 模型——结构和真正的 GPT-2 一模一样，只差还没训练。
>
> **前置**（卡住就翻回去）：
> - `nn.Module` 的"零件 + 数据流"套路、`numel()` 数参数量 → 深化册 [第 0 章](ch00_python_pytorch.md)
> - `MultiHeadAttention`（本章直接 `import` 它当心脏）、`nn.Linear` / `register_buffer` / `softmax` → 深化册 [第 3 章](ch03_attention.md)

---

[![在 Colab 中打开本章](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rasbt/LLMs-from-scratch/blob/main/ch04/01_main-chapter-code/ch04.ipynb)

> 配套可运行 notebook（rasbt/LLMs-from-scratch）。**边读边跑、跟着每节的「🔧 亲手改一下」改一改。**

## 0. 三分钟回顾（概念课直觉）

第 3 章你造出了**注意力**——句子里每个词环顾全场、把相关信息拉过来。但注意力只是**一个零件**，单独一个它还不是 GPT。

这一章要做的，是把零件**组装成整机**。整机长这样（从下往上数据流）：

```
输入：一串 token ID
  → token 嵌入 + 位置嵌入        （把词变成向量，第 2 章学过）
  → [ Transformer 积木 ] × N      （本章核心：堆 N 块同样的积木）
  → 最终层归一化
  → 输出层（线性）                 （把向量翻译回"下一个词的概率"）
输出：每个位置上、下一个词的打分（logits）
```

注意最后两步：**输出层把每个位置的向量，翻译成"词表里每个词当下一个词的可能性"**。这正是全册主线——**预测下一个词**——在代码里最终落地的地方。

本章的策略和第 3 章一样**层层加料**：先搭一个空壳（占位符）看清整体 → 把空壳里的零件一个个换成真货（层归一化 → GELU/前馈 → 残差 → Transformer 块）→ 最后拼成 `GPTModel`，再写一个循环让它**真的生成文本**。

---

## 1. 先定一张"配置表"：`GPT_CONFIG_124M`（书 4.1）

### 1.1 要解决什么

一个 GPT 有很多尺寸（词表多大、向量多少维、堆几层、几个注意力头……）。与其把这些数字散落在代码各处，不如**集中写进一个字典**，传给模型一份就行。书里给最小的 GPT-2（1.24 亿参数）定了这张表：

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # 词表大小
    "context_length": 1024, # 上下文长度（一次最多能看多少 token）
    "emb_dim": 768,         # 嵌入维度（每个 token 用 768 维向量表示）
    "n_heads": 12,          # 注意力头数
    "n_layers": 12,         # Transformer 块的层数
    "drop_rate": 0.1,       # dropout 比例（10% 随机丢弃）
    "qkv_bias": False       # Q/K/V 线性层是否带偏置
}
```

### 1.2 逐行拆

- **`vocab_size`：50257**——这是第 2 章 BPE 分词器的词表大小。它决定了**最后输出层有多宽**：模型每一步要对这 50257 个候选词各打一个分。
- **`context_length`：1024**——模型一次最多能"记住"多少个 token。超过就得截断（后面 `generate_text_simple` 里会看到 `idx[:, -context_size:]` 这一刀）。
- **`emb_dim`：768**——每个 token 被表示成 768 维向量。第 3 章为了印在纸上用了 3 维，这里换成真实尺寸 768，**算法一字不变，只是数大了**。
- **`n_heads`：12** / **`n_layers`：12**——12 个注意力头、12 块 Transformer 积木。"124M"这个参数量就是这套数字堆出来的。
- **`drop_rate`：0.1** / **`qkv_bias`：False**——dropout 防过拟合（第 3 章学过）；`qkv_bias=False` 是现代 LLM 的惯例，第 6 章加载 OpenAI 权重时才会改回 `True`。

> **为什么是"124M"不是"117M"？** 原始 GPT-2 论文先写成 1.17 亿，后来更正为 1.24 亿。GPT-3 架构和 GPT-2 **几乎一样**，只是参数从 15 亿涨到 1750 亿、数据更多。书选 GPT-2 是因为它的权重公开、能在笔记本上跑——学原理够用了。

🔧 **亲手改一下**：把 `emb_dim` 从 768 改成 1024、`n_layers` 改成 24、`n_heads` 改成 16——这就是练习 4.2 里的"GPT-2 中型"。光改这张表、一行模型代码都不用动，就能造出更大的 GPT。这正是"配置表"的威力。

---

## 2. 先看整体骨架：占位符 `DummyGPTModel`（代码清单 4.1）

### 2.1 要解决什么

直接上完整模型，零件太多容易迷路。书的办法很聪明：**先搭一个空壳**，把每个零件用"什么都不做的占位符"顶上，**先把数据流跑通、把形状看清**，再回头把占位符一个个换成真货。

```python
import torch
import torch.nn as nn

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

class DummyTransformerBlock(nn.Module):   # 占位符：什么都不做
    def __init__(self, cfg):
        super().__init__()
    def forward(self, x):
        return x

class DummyLayerNorm(nn.Module):          # 占位符：什么都不做
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
    def forward(self, x):
        return x
```

### 2.2 逐行拆

盯住 `forward` 里那条**数据流水线**（后面真模型 `GPTModel` 一字不改）：

- `self.tok_emb`：**token 嵌入层**，把每个 token ID 查成一个 768 维向量。输入 `[batch, seq_len]` → 输出 `[batch, seq_len, emb_dim]`。
- `self.pos_emb`：**位置嵌入层**。`torch.arange(seq_len)` 生成 `[0, 1, 2, ...]` 这串位置号，查成位置向量。这样模型才知道"谁在前谁在后"。
- `x = tok_embeds + pos_embeds`：**词义 + 位置，直接相加**——这就是喂进 Transformer 的初始向量。
- `self.drop_emb`：嵌入处的 dropout。
- `self.trf_blocks`：`n_layers` 块 Transformer 积木**串成一串**（这里还是占位符，原样吐回）。
- `self.final_norm`：最后的层归一化。
- `self.out_head`：**输出层**，`nn.Linear(emb_dim, vocab_size)`，把每个位置的 768 维向量映射成 50257 个分数。
- 返回 `logits`，形状 `[batch, seq_len, vocab_size]`。

> ### 🐍 加油站 — `nn.Sequential` 与那个 `*[...]`
> - **`nn.Embedding(行数, 列数)`**：一张**查找表**。给它一个整数 ID，它返回表里第 ID 行那个向量。token 嵌入表是 `[vocab_size, emb_dim]`，位置嵌入表是 `[context_length, emb_dim]`。
> - **`nn.Sequential(a, b, c)`**：把几个层**首尾串起来**，数据自动 `a → b → c` 顺着流。`x = self.trf_blocks(x)` 一行就把 N 块积木全过了一遍。
> - **`*[DummyTransformerBlock(cfg) for _ in range(n_layers)]`**：列表推导先造出 N 个积木，前面的 `*` 把这个**列表拆开**当成 N 个独立参数喂给 `nn.Sequential`（因为 `Sequential` 要的是 `Sequential(块1, 块2, ...)` 而不是 `Sequential([块1, 块2])`）。"堆 N 层"这件事，就浓缩在这一行。

跑一下空壳（用第 2 章的分词器造一个两句话的 batch）：

```python
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)        # [2, 4]

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape)     # torch.Size([2, 4, 50257])
```

**输出 `[2, 4, 50257]` 怎么读**：2 句话、每句 4 个 token、每个 token 配一个 50257 维向量——**这 50257 个数，就是"下一个词该是词表里哪个"的打分**。空壳跑通了，接下来把占位符换成真货。

---

## 3. 层归一化：把每层输出"拉回标准身材"（书 4.2，代码清单 4.2）

### 3.1 要解决什么

层数一深，每层输出的数值容易忽大忽小，导致训练不稳（梯度爆炸/消失）。**层归一化（layer normalization）** 的主意很简单：把每个样本的那一行激活，**调整成均值 0、方差 1**，让数值始终待在温和区间，训练更稳更快。

先手算一遍，看清它在干嘛。造一个"5 进 6 出"的小层：

```python
torch.manual_seed(123)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)               # [2, 6]

mean = out.mean(dim=-1, keepdim=True)    # 每行的均值 [2, 1]
var  = out.var(dim=-1, keepdim=True)     # 每行的方差 [2, 1]
out_norm = (out - mean) / torch.sqrt(var)
```

**逐行拆**：
- `out.mean(dim=-1, keepdim=True)`：沿**最后一维**（每一行的 6 个数）求均值。`keepdim=True` 保留维度，结果是 `[2, 1]` 而不是 `[2]`——这样才能和 `[2, 6]` 做减法广播。
- `(out - mean) / torch.sqrt(var)`：**减均值、除标准差**——这就是归一化的核心动作。做完每行均值≈0、方差≈1。

> ### 🐍 加油站 — `dim=-1` 和 `keepdim=True`
> - **`dim=-1`** 指"最后一维"。对 `[行, 列]` 就是**沿列方向**（把每一行那串数收拢成一个统计量）。后面 GPT 里张量是 `[batch, num_tokens, emb_dim]` 三维，`dim=-1` 仍然正好对准 `emb_dim`——**对每个 token 的 768 维向量各自归一化**，不用改代码。
> - **`keepdim=True`** 让求和/求均值后**保留那一维的位置**（变成长度 1），这样结果能和原张量"广播"对齐做加减。不加它形状会塌一维，后面减法就对不齐。

### 3.2 贴书真实代码：`LayerNorm` 类

把上面的手算封装成一个可复用模块：

```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
```

**逐行拆**：
- `self.eps = 1e-5`：一个**极小的正数 epsilon**，加在方差上防止"除以 0"。万一某行所有数都一样、方差为 0，没有它就崩了。
- `self.scale` / `self.shift`：两个**可训练参数**，初值分别是全 1 和全 0。归一化把数据压成"标准身材"后，模型可以通过这两个旋钮**再缩放、再平移**——如果它觉得某些维度该放大或偏移，训练中会自己调。
- `forward` 里：减均值、除 `sqrt(var + eps)`、最后 `scale * norm_x + shift`。**先标准化，再让模型有机会自定义。**
- `unbiased=False`：方差按 `/n`（而不是 `/(n-1)`）算，叫**有偏方差**。这么选是为了和 GPT-2 原始实现（TensorFlow 默认行为）对齐，**第 6 章加载预训练权重时才不会错位**。维度 n 很大时，n 和 n-1 的差别可忽略。

> ### 🐍 加油站 — `nn.Parameter`：为什么 scale/shift 要这么包
> `nn.Parameter(torch.ones(emb_dim))` 把一个张量**登记成"模型的可训练参数"**。一旦登记，PyTorch 在 `.backward()` 时会自动给它算梯度、在训练中帮你更新它。`scale`、`shift` 就是这么变成"模型能学的旋钮"的。对比一下：第 3 章的 `register_buffer('mask', ...)` 登记的是"**不**训练、但要跟模型搬 GPU"的常量。`Parameter` = 要学；`buffer` = 不学但要带着走。

> **层归一化 vs 批归一化**：批归一化是"**跨一个批次**的样本"归一化，层归一化是"**在每个样本内、跨特征维度**"归一化。后者**不依赖批大小**，所以批大小怎么变、分布式训练怎么切，它都稳——这对 LLM 特别重要。

🔧 **亲手改一下**：`ln = LayerNorm(emb_dim=5)`，把 `batch_example` 喂进去，再 `print(ln(batch_example).mean(dim=-1))` 和 `.var(dim=-1, unbiased=False)`。你会看到均值≈0（像 `-0.0000`）、方差≈1。再把 `self.eps` 改成 `1.0` 重试——方差会明显偏离 1，你就直观看到了 eps 该"小"的原因。

---

## 4. GELU 与前馈网络：积木里的"思考层"（书 4.3，代码清单 4.3 / 4.4）

### 4.1 要解决什么

Transformer 积木里除了注意力，还有一个**前馈网络（feed-forward network）**：注意力负责"词与词之间互相看"，前馈负责"在每个位置上，把这个词的信息再深加工一遍"。前馈里要用一个比 ReLU 更平滑的激活函数——**GELU（高斯误差线性单元）**。

```python
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
```

**逐行拆**：这是 GELU 的**近似公式**（原始 GPT-2 就用这个曲线拟合版，比精确版算得快）。你不用背这条式子，只需知道它的**形状**：
- ReLU：负数一律压成 0，0 点有个"尖角"。
- GELU：一条**平滑曲线**，几乎处处有非零梯度，连负数也允许漏出一点点（不是硬砍成 0）。

平滑 = 优化时更"顺滑"、调参更细腻；负区间不归零 = 即使收到负输入的神经元，也还能为学习出点力。深网络里这通常带来更好的训练表现。

### 4.2 贴书真实代码：`FeedForward` 类

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
```

**逐行拆**：三明治结构——**先撑大，再压回**：
- 第一层 `nn.Linear(768, 4*768)`：把 768 维**扩张到 3072 维**（4 倍）。
- 中间 `GELU()`：非线性激活，给模型"拐弯"的能力。
- 第二层 `nn.Linear(4*768, 768)`：再**缩回 768 维**。

进 768、出 768，**形状不变**——这点至关重要：正因为前馈进出同维，N 块积木才能**无缝堆叠**，不用在中间反复调维度。中间撑到 4 倍，是给模型一个**更宽阔的空间去探索更丰富的表示**，加工完再收回来。

验证形状不变：

```python
ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)            # [batch, num_tokens, emb_dim]
out = ffn(x)
print(out.shape)                     # torch.Size([2, 3, 768])
```

🔧 **亲手改一下**：把 `4 * cfg["emb_dim"]` 里的 `4` 改成 `8`，再 `print(ffn(x).shape)`。输出形状**还是 `[2,3,768]`**（外壳没变），但内部参数量大了一截——这就是 LLM 里调容量最常见的旋钮之一"扩张倍数"。

---

## 5. 残差连接：给梯度修一条"近路"（书 4.4，代码清单 4.5）

### 5.1 要解决什么

网络一深，梯度从最后一层往前传时会**越传越小**（梯度消失），靠前的层几乎学不动。**残差连接（residual / shortcut / skip connection）** 的解法朴素到不可思议：**把一层的输入，直接加到它的输出上**（`x = x + layer(x)`），等于给梯度修了一条**绕过该层的近路**，让它能畅通地一路传回前面。

书用一个 5 层小网络做对照实验，亲眼看"加不加近路"的差别：

```python
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output       # ← 近路：把输入加回输出
            else:
                x = layer_output
        return x
```

**逐行拆**：
- `nn.ModuleList([...])`：装一串子层的"正规列表"，里面的层会被 PyTorch 正确登记（能训练、能搬 GPU）。
- `forward` 里逐层过；**关键就在 `x = x + layer_output`**：只有当输入输出**形状相同**时才加得回去（不同维就没法逐元素相加），所以代码里有 `x.shape == layer_output.shape` 这道判断。
- `use_shortcut=False` 时退化成普通深网络（`x = layer_output`），用来做对照。

> ### 🐍 加油站 — `nn.ModuleList` vs 普通 `[]`
> 直接用 Python 列表 `self.layers = [层1, 层2]` 装子层，PyTorch **看不见**里面的参数——不会训练、不会搬 GPU。必须用 `nn.ModuleList(...)`（或上一节的 `nn.Sequential`）把它们**登记**进模型。区别：`ModuleList` 只是"正规列表"，你要自己写循环决定怎么流（像这里要插入残差判断）；`Sequential` 会自动顺序流过。需要在层间做手脚（如加近路）就用 `ModuleList`。

### 5.2 亲眼看梯度：近路救活了前面的层

```python
def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.]])
    loss = nn.MSELoss()
    loss = loss(output, target)
    loss.backward()
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., -1.]])

torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)
print_gradients(model_without_shortcut, sample_input)
```

**没有近路**，梯度从后往前一路缩水（注意数量级）：

```
layers.0.0.weight has gradient mean of 0.00020173...   ← 第一层，几乎没梯度
layers.1.0.weight has gradient mean of 0.00012011...
layers.2.0.weight has gradient mean of 0.00071520...
layers.3.0.weight has gradient mean of 0.00139887...
layers.4.0.weight has gradient mean of 0.00504964...   ← 最后一层
```

**加上近路**（`use_shortcut=True`）后：

```python
torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
print_gradients(model_with_shortcut, sample_input)
```

```
layers.0.0.weight has gradient mean of 0.22169...      ← 第一层梯度回来了！
layers.1.0.weight has gradient mean of 0.20694...
layers.2.0.weight has gradient mean of 0.32896...
layers.3.0.weight has gradient mean of 0.26657...
layers.4.0.weight has gradient mean of 1.32585...
```

**逐行拆**：
- `loss.backward()`：PyTorch 自动反向传播，给每个参数算好梯度，省你手推数学。
- `param.grad.abs().mean()`：取每层梯度的**平均绝对值**，一层一个数，方便横向比。
- 对比两组数：没近路时第一层梯度只有 `0.0002` 级别（基本学不动）；加了近路，第一层回到 `0.22`——**梯度不再缩水**。这就是残差连接的全部意义：**在很深的网络里，保证梯度能畅通流到每一层**。LLM 这种几十上百层的怪物，离了它没法训练。

🔧 **亲手改一下**：把 `layer_sizes` 加长成 `[3,3,3,3,3,3,3,1]`（更深），分别跑两种 `use_shortcut`。没近路时第一层梯度会更小（消失更严重），有近路时依旧稳——你就复现了"越深越离不开残差"这个结论。

---

## 6. Transformer 块：把零件拼成一块积木（书 4.5，代码清单 4.6）

### 6.1 要解决什么

到这里四样零件都齐了：**多头注意力（第 3 章）、前馈、层归一化、残差**。现在把它们**组装成一块标准积木**——这就是 GPT 里被重复堆叠十几次的那个 `TransformerBlock`。

```python
from chapter03 import MultiHeadAttention   # ← 第 3 章你亲手造的那个心脏

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # —— 第一段：注意力子层 ——
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut          # 残差：把输入加回来

        # —— 第二段：前馈子层 ——
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut          # 残差：把输入加回来
        return x
```

### 6.2 逐行拆

- `__init__` 里**备齐四样零件**：一个 `MultiHeadAttention`（注意这里 `d_in = d_out = emb_dim = 768`，进出同维才能堆叠）、一个 `FeedForward`、**两个** `LayerNorm`（注意力前一个、前馈前一个）、一个共用的 dropout。
- `forward` 里是**两段几乎一模一样的结构**，每段都是 **"归一化 → 子层 → dropout → 加残差"**：
  - 第一段的子层是**注意力** `self.att`；
  - 第二段的子层是**前馈** `self.ff`。
- 每段开头 `shortcut = x` 把入口存一份，结尾 `x = x + shortcut` 加回去——这正是第 5 节的残差近路，**在真模型里就这么用**。
- **`norm` 在子层之前**（先归一化再进注意力/前馈），这叫 **Pre-LayerNorm**（前置层归一化）。老式 Transformer 把 norm 放在子层之后（Post-LayerNorm），训练动态更差。现代 GPT 都用 Pre-LN。

> **一句话记住这块积木的节奏**：`x → [归一化 → 注意力 → +残差] → [归一化 → 前馈 → +残差] → x`。注意力让词互相看，前馈逐位置深加工，两次残差保梯度——**进 768 维、出 768 维，形状纹丝不动**。

验证进出同形：

```python
torch.manual_seed(123)
x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)
print("Input shape:", x.shape)       # torch.Size([2, 4, 768])
print("Output shape:", output.shape) # torch.Size([2, 4, 768])
```

**进出同形不是巧合，是设计**：正因为一块积木"吃 768 吐 768"，你才能把 N 块**首尾相接**堆成任意深度，而每个输出向量仍**一一对应**一个输入位置——只不过它的内容已经被**重新编码**、融进了整句话的上下文。

🔧 **亲手改一下**：把 `forward` 第一段的 `x = x + shortcut` 注释掉（去掉注意力子层的残差），再跑一遍——形状不会变，但你已经亲手"拆掉了一条梯度近路"。结合第 5 节想想：如果整个 12 层 GPT 都这么拆，训练会怎样？

---

## 7. 拼成完整的 `GPTModel`（书 4.6，代码清单 4.7）

### 7.1 要解决什么

现在把第 2 节那个空壳里的占位符，**换成刚造好的真货**：`DummyTransformerBlock → TransformerBlock`，`DummyLayerNorm → LayerNorm`。其余数据流**一行不改**。

```python
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
```

### 7.2 逐行拆

和第 2 节那张流水线**逐字对照，唯二变化**就是 `trf_blocks` 里换成了真 `TransformerBlock`、`final_norm` 换成了真 `LayerNorm`：

- `tok_emb` + `pos_emb`：token 嵌入 + 位置嵌入，相加得到初始向量 `[batch, seq_len, 768]`。
- `trf_blocks`：**12 块真 Transformer 积木**串行处理。这是模型的主体。
- `final_norm`：最后再归一化一次，稳住输出。
- `out_head`：`nn.Linear(768, 50257)`，把每个位置的向量**翻译成 50257 个分数**——这就是"下一个词的非归一化概率"（logits）。

> **这一行就是主线落地处**：`logits = self.out_head(x)`。前面所有零件忙活半天，最终目的就是让**这一步对"下一个词该是谁"打分打得越来越准**。整本书、整个 GPT，归根到底就在做这一件事。

跑一下完整模型：

```python
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)               # batch 是第 2 节那个 [2, 4]
print("Output shape:", out.shape)  # torch.Size([2, 4, 50257])
```

### 7.3 数一数参数：为什么是 1.63 亿，不是 1.24 亿？

```python
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")
# Total number of parameters: 163,009,536
```

**逐行拆**：`p.numel()` 数一个参数张量里有多少个数，`sum(...)` 把全模型加起来。算出来是 **1.63 亿**——可我们不是说好 124M 吗？

谜底是 **权重捆绑（weight tying）**。原始 GPT-2 让**输出层复用 token 嵌入层的权重**（两者形状都是 `[50257, 768]`，可以共用同一张表）：

```python
print("Token embedding layer shape:", model.tok_emb.weight.shape)  # [50257, 768]
print("Output layer shape:", model.out_head.weight.shape)          # [50257, 768]

total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")
# 124,412,160
```

把输出层那份重复的参数减掉，正好回到 **124M**——和原始 GPT-2 对上了。

> ### 🐍 加油站 — 权重捆绑（weight tying）
> token 嵌入做的是"**词 ID → 向量**"，输出层做的是"**向量 → 每个词的分数**"——一个是查表，一个是反查表，**方向正好相反**。GPT-2 干脆让它俩共用同一张 `[50257, 768]` 的表，省下 3800 多万参数和对应内存。**注意**：本书的 `GPTModel` 实现里**故意用两张独立的表**（作者经验是分开训练效果更好），所以才数出 1.63 亿；第 6 章加载 OpenAI 权重时会再把捆绑用上。

最后估一下"占多大内存"：

```python
total_size_mb = (total_params * 4) / (1024 * 1024)
print(f"Total size of the model: {total_size_mb:.2f} MB")  # 621.83 MB
```

每个参数是 32 位浮点（4 字节），1.63 亿 × 4 字节 ≈ **621.83 MB**。连"最小"的 GPT 都要占大半个 G——这还只是参数本身。

🔧 **亲手改一下**：照练习 4.2，把配置改成 GPT-2 中型（`emb_dim=1024, n_layers=24, n_heads=16`），重新建模型、重数 `total_params`。**只动配置字典、模型代码一行不改**，参数量就涨到约 3.45 亿。这就是"配置即规模"。

---

## 8. 让它真的生成文本：`generate_text_simple`（书 4.7，代码清单 4.8）

### 8.1 要解决什么

模型现在能吐 logits 了，但那还是一堆数。**怎么从数变回文字？** 核心循环只有一句话能概括——**"预测 → 接上 → 再预测"**：模型预测下一个词，把它拼回输入末尾，拿这条更长的序列再预测下一个……如此往复。这正是第 1 天主线"超级下一个词预测器"在代码里的样子。

```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

### 8.2 逐行拆

`idx` 是当前已有的 token 序列，形状 `[batch, num_tokens]`。循环 `max_new_tokens` 次，每次生成一个新 token：

- `idx_cond = idx[:, -context_size:]`：**只保留最后 `context_size` 个 token**。序列再长，模型一次也只能看 1024 个（配置里的 `context_length`），超了就从尾巴截。
- `with torch.no_grad():`：生成（推理）阶段**不需要算梯度**，关掉它省内存、提速。
- `logits = model(idx_cond)`：过一遍模型，得到 `[batch, num_tokens, vocab_size]`——每个位置都有一份"下一个词"的打分。
- `logits = logits[:, -1, :]`：**只取最后一个位置**的打分 `[batch, vocab_size]`。因为我们只关心"接在当前序列后面的那个词"，前面位置的预测这一步用不上。
- `probas = torch.softmax(logits, dim=-1)`：把分数变成**加起来为 1 的概率分布**。
- `idx_next = torch.argmax(probas, dim=-1, keepdim=True)`：**挑概率最大的那个词**的 ID。这种"总选最可能的"策略叫**贪心解码（greedy decoding）**。
- `idx = torch.cat((idx, idx_next), dim=1)`：把新词**拼到序列末尾**——下一轮循环就拿这条更长的序列继续预测。

> **一个小彩蛋**：`softmax` 是**单调**的，不改变大小顺序，所以"先 softmax 再 argmax"和"直接对 logits 取 argmax"**结果一样**——这一步的 softmax 严格说是多余的。书保留它，只是为了让你看清"logits → 概率 → 选词"的**完整直觉**。

> ### 🐍 加油站 — `argmax` + `torch.cat`：生成循环的两个支点
> - **`torch.argmax(x, dim=-1, keepdim=True)`**：返回最后一维里**最大值所在的下标**（不是最大值本身）。这里下标恰好就是 token ID。`keepdim=True` 让它保持 `[batch, 1]`，好和 `idx` 拼接。
> - **`torch.cat((a, b), dim=1)`**：沿第 1 维（token 维）把两个张量**接起来**。`[batch, n]` 接 `[batch, 1]` → `[batch, n+1]`。序列每轮**长一个 token**，全靠它。
> 一句话：argmax 负责"选出下一个词"，cat 负责"把它接上去"——**生成就是这两个动作转圈圈**。

### 8.3 跑一次（结果是乱码，这正常）

```python
start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)   # [1, 4]

model.eval()                                          # 关掉 dropout 等训练专用组件
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"])

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)
# Hello, I am Featureiman Byeswickattribute argue
```

**逐行拆**：
- `.unsqueeze(0)`：在最前面加一个 batch 维，把 `[4]` 变成 `[1, 4]`（模型要求带 batch）。
- `model.eval()`：切到**评估模式**，关掉 dropout 这类只在训练时生效的随机组件，让输出确定。
- 结果是 `Hello, I am Featureiman Byeswick...`——**一串乱码**。

为什么乱码？因为**模型还没训练**——参数全是随机初始值，它压根没学过语言。结构是对的、流程是通的、形状全对，**只差训练**。把"猜下一个词"练好，是下一章（第 5 天）的事。

🔧 **亲手改一下**：把 `max_new_tokens` 从 6 改成 20，看它继续往后"编"更多乱码 token；再把 `start_context` 换成别的句子。你会确认：**无论喂什么，没训练的模型都只会胡说**——这恰恰反证了"训练"才是让它说人话的关键。

---

## 9. 练习

> 📓 **对照官方答案**：卡住了别硬磕——[在 Colab 打开本章练习解答 notebook ↗](https://colab.research.google.com/github/rasbt/LLMs-from-scratch/blob/main/ch04/01_main-chapter-code/exercise-solutions.ipynb)

### 书上原题

**练习 4.1：前馈模块 vs 多头注意力模块，谁的参数多？**
分别数 `FeedForward` 和 `MultiHeadAttention` 里的参数量，比一比。

<details><summary>解题思路（点开）</summary>

直接用 `numel` 数：

```python
block = TransformerBlock(GPT_CONFIG_124M)
ff_params  = sum(p.numel() for p in block.ff.parameters())
att_params = sum(p.numel() for p in block.att.parameters())
print("FeedForward:", ff_params)        # ≈ 4,722,432
print("Attention:  ", att_params)       # ≈ 2,360,064
```

**前馈约是注意力的 2 倍**。原因：前馈有两个 `768↔3072` 的大线性层（`768*3072*2 ≈ 472 万`）；注意力主要是 Q/K/V 和输出投影四个 `768×768` 的矩阵（`768*768*4 ≈ 236 万`）。**收获**：很多人以为注意力最"重"，其实单看参数量，前馈才是积木里的"大胃王"。
</details>

**练习 4.2：用同一个 `GPTModel` 类造出更大的 GPT-2。**
不改任何模型代码，只改配置，实现 GPT-2 中型（`emb_dim=1024, n_layers=24, n_heads=16`）、大型（`1280, 36, 20`）、XL（`1600, 48, 25`），并算各自参数量。

<details><summary>解题思路（点开）</summary>

复制 `GPT_CONFIG_124M`，只改那三个数即可：

```python
config = GPT_CONFIG_124M.copy()
config.update({"emb_dim": 1024, "n_layers": 24, "n_heads": 16})  # GPT-2 中型
model = GPTModel(config)
print(sum(p.numel() for p in model.parameters()))   # ≈ 3.45 亿
```

**收获**：彻底体会"配置即规模"——同一套代码，靠一张表就能从 1.24 亿伸缩到 15 亿。这就是为什么第 1 节要把所有尺寸抽进字典。
</details>

**练习 4.3：给不同位置用各自独立的 dropout。**
现在全模型共用一个 `drop_rate`。改成三处（嵌入层、残差/shortcut、注意力模块）各用各的 dropout 值。

<details><summary>解题思路（点开）</summary>

在配置里拆成三个键，比如 `"drop_rate_emb"`、`"drop_rate_shortcut"`、`"drop_rate_attn"`，然后在 `GPTModel`、`TransformerBlock`、`MultiHeadAttention` 里分别读对应的键。**收获**：理解 dropout 在 GPT 里**不止一处**，以及"超参数"为什么常常要按位置分别调。
</details>

### 本册自测题（改了会怎样）

1. **拆掉残差会怎样**：把 `TransformerBlock.forward` 里两处 `x = x + shortcut` 都删掉，模型还能跑（形状不变），但回想第 5 节——12 层叠起来后，梯度会怎样？为什么说残差是深模型的"命根子"？
2. **Pre-LN 改 Post-LN**：把 `norm1`/`norm2` 从子层**之前**挪到子层**之后**（即先 `att` 再 `norm`）。代码能跑，但这就退回了老式 Post-LayerNorm。结合 6.2 节想：训练动态会更好还是更差？
3. **贪心 vs 随机**：`generate_text_simple` 里把 `torch.argmax(...)` 换成"按 `probas` 概率随机采样一个"（`torch.multinomial(probas, num_samples=1)`）。同一个输入多跑几次，输出还一样吗？这就是下一章"采样让生成有创造性"的前奏。

---

## 10. 本章小结

1. **一个 GPT = 嵌入 → 堆 N 块 Transformer 积木 → 最终归一化 → 输出层**。整章就是把第 2 节那张空壳流水线里的占位符，一个个换成真零件。
2. **一块 Transformer 积木 = 两段"归一化 → 子层 → dropout → 加残差"**：第一段子层是多头注意力，第二段是前馈网络。进 768 出 768、形状不变，所以能无限堆叠。
3. 四样关键零件各司其职：**层归一化**稳住数值、**GELU/前馈**逐位置深加工、**残差**给梯度修近路、**注意力**让词互相看。
4. **配置即规模**：所有尺寸抽进一张字典，改几个数就能从 124M 伸到 XL，模型代码一行不动。
5. **参数量 1.63 亿 vs 1.24 亿** 的差别来自**权重捆绑**——本书故意用独立的嵌入/输出层，所以数得多；减去重复的输出层就回到 124M。
6. **生成 = "预测 → 接上 → 再预测"的循环**（`generate_text_simple`）；末尾**输出层把向量翻译回"下一个词的概率"**——这就是全册主线在代码里落地的那一行。没训练的模型只会吐乱码，但**结构已经和真 GPT-2 一模一样**。

---

## 11. 能改自检清单（全勾＝过关）

- [ ] 我能不看书，画出从"一串 token ID"到"logits"的完整数据流（嵌入 → N 块积木 → 归一化 → 输出层）。
- [ ] 我能说出 Transformer 块里那两段结构，并指出哪段是注意力、哪段是前馈、残差加在哪。
- [ ] 我能解释层归一化在归一化什么、为什么要 `eps`、`scale`/`shift` 是干嘛的。
- [ ] 我能讲清残差连接为什么能救梯度消失，并能用 `print_gradients` 的两组数佐证。
- [ ] 我能解释为什么参数量数出来是 1.63 亿而不是 1.24 亿（权重捆绑）。
- [ ] 我能把 `generate_text_simple` 的循环用"预测→接上→再预测"一句话复述，并指出 `argmax` 和 `cat` 各管什么。
- [ ] 我能独立完成练习 4.2（改配置造更大模型），并预测自测题里每个改动的后果再验证。

---

## 12. 通往下一章

你已经有了一个**完整但还不会说话**的 GPT——结构齐全、流程跑通，就差**训练**。**第 5 章（第 5 天）** 会把"猜下一个词"变成一个能优化的**损失函数**，让模型在半个互联网上反复练习、看着损失一路滚下山，直到它从吐乱码变成吐人话。

> 带走主线：这一章你把注意力这块"心脏"装进了 Transformer 积木、又堆成了整机；而整机忙活的唯一目的，仍然是那一行 `out_head`——**把下一个词预测得更准**。
