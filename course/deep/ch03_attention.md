# 深化册 · 第 3 章　注意力机制：一行一行把"心脏"造出来

> **对应**：概念课 **第 3 天** ｜ 书 **第 3 章（编码注意力机制）**
> **一句话直觉**：句子里每个词都"环顾全场"，从别的词那里把和自己相关的信息**拉过来**——这个"环顾并拉取"的动作就是注意力。
> **这一章你将亲手得到**：一个能跑的 `MultiHeadAttention` 类（就是真实 GPT 里那一块），而且**每一行你都能说出它在算什么**。
>
> **前置**（卡住就翻到对应「加油站」）：
> - 张量是什么、形状（shape）怎么读 → 本章 [加油站 ①](#加油站1)
> - 矩阵乘法 `@` 与"加权求和" → [加油站 ②](#加油站2)
> - `torch.softmax` 和 `dim` 参数 → [加油站 ③](#加油站3)
> - `nn.Module` / `nn.Linear` / `nn.Parameter` → [加油站 ④](#加油站4)
> - `.view` / `.transpose` 改形状 → [加油站 ⑤](#加油站5)

---

[![在 Colab 中打开本章](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rasbt/LLMs-from-scratch/blob/main/ch03/01_main-chapter-code/ch03.ipynb)

> 配套可运行 notebook（rasbt/LLMs-from-scratch）。**边读边跑、跟着每节的「🔧 亲手改一下」改一改。**

## 0. 三分钟回顾（概念课直觉）

为什么需要注意力？因为**词的意思依赖语境**。

> "我去**银行**取钱" vs "我坐在河的**银行**边"——同一个"银行"，意思天差地别。

模型怎么知道该取哪个意思？让"银行"去看周围的词（"取钱" / "河边"），据此调整自己的含义。这个机制用三个角色实现（概念课里背过的那组比喻）：

- **Query（查询）**＝"我在找什么"
- **Key（键）**＝"我能提供什么"
- **Value（值）**＝"选中我，我就给你这些"

本章就是把这套比喻，**一步一步变成能跑的代码**。书里的策略很聪明：先做一个**没有可训练权重**的"玩具版"（3.3 节），把流程跑通；再加上可训练权重（3.4 节）；再加因果掩码（3.5 节）；最后扩成多头（3.6 节）。我们就按这条阶梯爬。

> 全章只用一个袖珍例子：把句子 `"Your journey starts with one step"` 的 6 个词，各自用一个 **3 维**向量表示。维度取 3 只是为了能印在纸上看清楚——真实 GPT 里是几百上千维，但**算法一模一样**。

---

## 1. 玩具版自注意力：先不带任何"可学的东西"（书 3.3）

### 1.1 输入长什么样

```python
import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89],  # Your    (x^1)
   [0.55, 0.87, 0.66],  # journey (x^2)
   [0.57, 0.85, 0.64],  # starts  (x^3)
   [0.22, 0.58, 0.33],  # with    (x^4)
   [0.77, 0.25, 0.10],  # one     (x^5)
   [0.05, 0.80, 0.55]]  # step    (x^6)
)
```

**逐行拆**：`inputs` 是一个形状为 `[6, 3]` 的张量——6 行（6 个词），每行 3 个数（这个词的 3 维嵌入向量）。这一摞向量就是第 2 章那条"文字→数字"流水线的产物，这里直接给定，省去前面的步骤。

<a name="加油站1"></a>
> ### 🐍 加油站 ① — 张量与"形状"
> **张量（tensor）** = 多维数字表格。一个数是 0 维（标量），一串数是 1 维（向量），一张表是 2 维（矩阵），再叠起来就是 3 维、4 维……
> **形状（shape）** 告诉你每一维有多大。`inputs.shape` 是 `torch.Size([6, 3])`，读作"6 行 3 列"。
> 记一个习惯动作：**任何时候看不懂一段张量代码，先 `print(x.shape)`**——形状对了，逻辑基本就对了。本章后面会反复用这招。

### 1.2 第一步：算"注意力分数"——谁和谁相关

先只算第 2 个词 `journey`（下标 1）对所有词的关注。做法：拿它的向量，和每个词的向量做**点积**。

```python
query = inputs[1]                       # 第 2 个词 journey 当"查询"
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)
# tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])
```

**逐行拆**：
- `query = inputs[1]`：取出第 2 行（`journey`）当查询。
- 循环里 `torch.dot(x_i, query)`：把 `journey` 和每个词 `x_i` 做点积，得到 6 个分数。
- **为什么用点积？** 点积是"两个向量有多对齐"的度量：方向越一致，点积越大。所以分数越高＝两个词越相关。注意第 2 个分数 `1.4950` 最大——`journey` 和它自己当然最像。

> **点积就是"对应位相乘再求和"**：`0.55*0.43 + 0.87*0.15 + 0.66*0.89 = 0.9544`，和 `torch.dot` 结果一致。

<a name="加油站2"></a>
> ### 🐍 加油站 ② — 点积、`@` 矩阵乘、和"加权求和"
> 整个神经网络，翻来覆去就一个动作：**一串数各乘一些权重再加起来**（加权求和）。点积是它最小的样子。
> - **点积**（两个向量）：`a·b = a[0]*b[0]+a[1]*b[1]+...`，结果是一个数。
> - **`@` 矩阵乘法**：一次做很多个点积。`A @ B`，其中 `A` 是 `[m, k]`、`B` 是 `[k, n]`，结果是 `[m, n]`——`结果[i,j] = A的第i行 · B的第j列`。
> - 关键规则：**左边的列数，必须等于右边的行数**（那个 `k` 要对上）。形状对不上就会报错，这是 PyTorch 最常见的错误，养成 `print(shape)` 的习惯能救你很多次。
> 后面你会看到，所有 `for` 循环都能被一句 `@` 替换掉——又快又短。

### 1.3 第二步：归一化成"注意力权重"——总和为 1

分数大小不便直接用，先把它变成**加起来等于 1** 的比例（像把"票数"变成"得票率"）。标准做法是 **softmax**：

```python
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print(attn_weights_2)
# tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print(attn_weights_2.sum())   # tensor(1.)
```

**逐行拆**：`softmax` 把任意一串数压成"全是正数、加起来为 1"的分布。于是这 6 个数就成了"`journey` 该给每个词分多少注意力"的比例。`dim=0` 指明沿着这一维（这里只有一维，共 6 个元素）做归一化。

<a name="加油站3"></a>
> ### 🐍 加油站 ③ — `softmax` 与那个总让人懵的 `dim`
> **softmax** 做两件事：① 把所有数变正（用指数 `e^x`）；② 除以总和，让它们加起来为 1。结果可以读成"概率/重要性占比"。
> **`dim=?` 是新手最大的坑**。它指"沿哪个方向求和归一化"。对一个 `[行, 列]` 的二维张量：
> - `dim=-1`（最后一维＝列方向）：**每一行**各自加起来为 1。← 注意力里几乎总是用这个。
> - `dim=0`（行方向）：每一列各自加起来为 1。
> 记法："`dim` 指的是**被压扁求和的那个方向**。" 拿不准就 `print(x.sum(dim=...))` 看看哪一边变成了 1。

### 1.4 第三步：算"上下文向量"——按比例把信息混起来

最后，按上面这 6 个比例，把 6 个词的向量加权混合，得到 `journey` 的**升级版向量**（上下文向量）：

```python
query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
print(context_vec_2)
# tensor([0.4419, 0.6515, 0.5683])
```

**逐行拆**：`attn_weights_2[i] * x_i` 是"第 i 个词的向量，乘以它分到的注意力比例"；全部加起来，就是"`journey` 环顾全场后，融合了相关信息的新表示"。这就是注意力的**最终产物**：上下文向量。

🔧 **亲手改一下**：把 `query = inputs[1]` 改成 `inputs[4]`（换成第 5 个词 `one`），重跑这三步。预测：分数最大的那一项会变成第几个？（提示：一个词和自己点积通常最大。）跑一下验证。

### 1.5 一次算全部 6 个词：用矩阵乘把循环消掉

上面只算了 `journey` 一个词。要对**所有 6 个词**同时算，不用写两层 `for`，一句矩阵乘搞定：

```python
attn_scores = inputs @ inputs.T          # [6,3] @ [3,6] -> [6,6]
attn_weights = torch.softmax(attn_scores, dim=-1)   # 每行归一化
all_context_vecs = attn_weights @ inputs            # [6,6] @ [6,3] -> [6,3]
print(all_context_vecs[1])
# tensor([0.4419, 0.6515, 0.5683])  # 和上面单算的 journey 完全一致
```

**逐行拆**（盯住形状）：
- `inputs @ inputs.T`：`[6,3]` 乘 `[3,6]` 得 `[6,6]`——一张"谁对谁"的关系表，第 `i` 行第 `j` 列＝词 `i` 对词 `j` 的分数。`.T` 是转置（行列对调），把 `[6,3]` 变 `[3,6]` 好让形状对上。
- `softmax(..., dim=-1)`：**每一行**归一化（第 `i` 行是"词 `i` 该怎么分配注意力"）。
- `attn_weights @ inputs`：`[6,6]` 乘 `[6,3]` 得 `[6,3]`——6 个词各自的上下文向量。
- 最后第 2 行（下标 1）和 1.4 节单算的结果一模一样，**说明矩阵版和循环版等价**——只是快得多。

> 到这里，注意力的"骨架流程"已经完整：**算分数 → softmax 归一化 → 加权求和**。后面所有复杂版本，都只是在这三步上加料。

---

## 2. 真实版自注意力：加上"可学习的权重"（书 3.4）

玩具版有个根本缺陷：它**没有任何可调的东西**——给定输入，输出就定死了，模型学不到任何"该怎么看"。真实的注意力引入**三个可训练的权重矩阵** `W_query`、`W_key`、`W_value`，让模型在训练中学会"如何提问、如何应答、如何给值"。这也叫**缩放点积注意力（scaled dot-product attention）**，是 GPT 等所有主流 LLM 用的版本。

### 2.1 把每个词投影成 Q、K、V 三种身份

```python
x_2 = inputs[1]          # 还是先盯住 journey
d_in = inputs.shape[1]   # 输入维度 = 3
d_out = 2                # 输出维度，这里故意设成 2，方便区分前后

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query   # journey 的"查询"身份
keys    = inputs @ W_key   # 所有词的"键"身份   -> [6,2]
values  = inputs @ W_value # 所有词的"值"身份   -> [6,2]
```

**逐行拆**：
- 每个词的原始 3 维向量，分别乘三个不同的矩阵 `[3,2]`，投影成三个 2 维向量——**同一个词的三种"身份"**：当查询用的 Q、当被查时挂的标签 K、被选中时交出的内容 V。
- `torch.manual_seed(123)`：固定随机种子，保证你我跑出来的随机权重一样，结果可复现。
- `requires_grad=False`：暂时**关掉梯度**，只是为了让打印干净。真要训练时设成 `True`，这三个矩阵就会在训练中被一点点调好。

> **"权重参数" ≠ "注意力权重"，别混！**
> - **权重参数**＝ `W_query` 这种矩阵里的数，是训练中要优化的、固定下来的"连接系数"。
> - **注意力权重**＝ softmax 出来的那串"该关注谁多少"的比例，是**随输入动态变化**的。
> 一个是"模型的脑回路"，一个是"此刻在看哪"。

<a name="加油站4"></a>
> ### 🐍 加油站 ④ — `nn.Module` / `nn.Parameter` / `nn.Linear`
> - **`nn.Module`**：PyTorch 里所有模型/网络层的**基类**。你自己写的网络都继承它，套路是：`__init__` 里"备好零件"，`forward` 里"定义数据怎么流过这些零件"。
> - **`nn.Parameter(...)`**：把一个张量登记成"模型的可训练参数"。一旦登记，PyTorch 训练时就会自动算它的梯度、帮你更新它。`W_query` 就是这么来的。
> - **`nn.Linear(d_in, d_out, bias=False)`**：一个现成的"全连接层"，作用就是 `x @ W`（关掉 bias 时正好是一次矩阵乘）。比手写 `nn.Parameter` 更好，因为它自带**更聪明的初始值**，训练更稳。下一节的 v2 版就改用它。

### 2.2 算分数、缩放、softmax、加权——和玩具版同构

```python
attn_scores_2 = query_2 @ keys.T          # journey 对所有词的分数 -> [6]
d_k = keys.shape[-1]                       # 键的维度 = 2
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
context_vec_2 = attn_weights_2 @ values    # -> [2]
print(context_vec_2)   # tensor([0.3061, 0.8210])
```

**逐行拆**（和第 1 节三步完全对应，只是把 input 换成了 Q/K/V）：
- `query_2 @ keys.T`：用 `journey` 的 **Q** 去和所有词的 **K** 比对，得分数。
- `/ d_k**0.5`：**这是新东西——缩放**。把分数除以"键维度的平方根"（`d_k**0.5` 即 √2）。
- `softmax`：归一化成注意力权重。
- `@ values`：按权重把所有词的 **V** 混合起来，得上下文向量。

> **为什么要除以 √d_k（"缩放"那一步）？** 维度一大，点积的数值会变得很大；很大的数喂进 softmax，会让它输出接近"非 0 即 1"的极端分布，反向传播时**梯度几乎消失**，模型学不动。除以 √d_k 把数值压回温和区间，训练才稳。"缩放点积注意力"的名字就来自这一步。

### 2.3 收进一个类：`SelfAttention_v2`（代码清单 3.2）

把上面散装的步骤，**封装成一个可复用的层**。书里先给了用 `nn.Parameter` 的 v1，再给改用 `nn.Linear` 的 v2（更好）。我们直接看 v2：

```python
import torch.nn as nn

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)
        attn_scores  = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec  = attn_weights @ values
        return context_vec

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))     # 一次性输出全部 6 个上下文向量 -> [6,2]
```

**逐行拆**：
- `class ... (nn.Module)` + `super().__init__()`：标准开头，声明"我是一个网络层"。
- `__init__` 里备好三个 `nn.Linear` 零件（就是 Q/K/V 三个投影）。
- `forward(self, x)`：定义数据流——投影出 Q/K/V → 算分数 → 缩放+softmax → 加权求和。**和 2.2 节那几行一字不差，只是搬进了类里。**
- 用的时候 `sa_v2(inputs)` 会自动调用 `forward`（这是 `nn.Module` 的约定，不用你手动写 `.forward`）。

> **为什么 v2 比 v1 好？** v1 用 `nn.Parameter(torch.rand(...))`，初始值是纯随机；v2 用 `nn.Linear`，自带经过设计的初始化方案，训练更稳定有效。代价：两者初始权重不同，所以**输出数值不同**（这正是练习 3.1 要你打通的点）。

🔧 **亲手改一下**：把 `d_out = 2` 改成 `d_out = 3`，重跑 `sa_v2(inputs)`。预测：输出形状会从 `[6,2]` 变成什么？为什么？（提示：上下文向量的维度由谁决定？）

---

## 3. 因果掩码：不许偷看未来（书 3.5）

### 3.1 为什么必须挡住未来

模型的任务是**预测下一个词**。处理第 3 个词时，它只能看第 1、2、3 个词；**绝不能看到第 4 个及以后**——那是答案，看了就等于作弊（也叫"信息泄露"）。所以要在注意力里加一道**因果掩码（causal mask）**：把每个词"看向未来"的那些注意力，强行抹掉。

直觉图（左下三角保留，右上三角抹零）：

```
        Your journey starts with one step
Your    [ ✓    ·      ·     ·    ·   ·  ]
journey [ ✓    ✓      ·     ·    ·   ·  ]
starts  [ ✓    ✓      ✓     ·    ·   ·  ]
with    [ ✓    ✓      ✓     ✓    ·   ·  ]
one     [ ✓    ✓      ✓     ✓    ✓   ·  ]
step    [ ✓    ✓      ✓     ✓    ✓   ✓  ]    (· = 被掩掉)
```

### 3.2 高效写法：把"未来"填成 −∞，再 softmax

朴素做法是"乘以 0/1 下三角矩阵再重新归一化"，两步。书里给了更漂亮的**一步法**：在 softmax **之前**，把上三角（未来位置）填成负无穷 `-inf`：

```python
context_length = attn_scores.shape[0]
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
```

**逐行拆**：
- `torch.triu(..., diagonal=1)`：取**上三角**（diagonal=1 表示不含主对角线）的 1，其余为 0——这正好标出"未来位置"。
- `masked_fill(mask.bool(), -torch.inf)`：凡是 mask 为真（未来）的格子，填成 `-inf`。
- `softmax`：`e^(-inf) = 0`，所以那些未来位置的权重**自动变成 0**，且每行剩下的部分仍自动加起来为 1——一步到位，不用再手动重新归一化。**这就是用 softmax 的数学性质省掉一步的巧思。**

> **为什么不直接乘 0？** 乘 0 之后每行的和就不是 1 了，还得再除以行和归一化（两步）。填 `-inf` 让 softmax 一步把"归零"和"重新归一"同时做了，更省也更不易错。

### 3.3 再加一点 dropout（防过拟合）

```python
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)        # 训练时随机丢一半
print(dropout(attn_weights))
```

**逐行拆**：`Dropout(0.5)` 在训练时**随机把一半的注意力权重置 0**，并把剩下的放大 2 倍（补偿总量）。目的：逼模型不要过度依赖某几个固定位置，**防止过拟合**。注意：dropout **只在训练时开，推理时自动关**。真实 GPT 里一般用 0.1~0.2，这里用 0.5 只是为了演示看得清楚。

### 3.4 收进一个类：`CausalAttention`（代码清单 3.3）

把"投影 + 因果掩码 + dropout"打包，并且让它能处理**一批（batch）**输入：

```python
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)
        attn_scores = queries @ keys.transpose(1, 2)         # 注意：转的是后两维
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec
```

**逐行拆（只看新增/变化处，其余和前面同构）**：
- 输入 `x` 现在是 **3 维** `[b, num_tokens, d_in]`：`b` 是这一批有几句话，`num_tokens` 是每句几个词，`d_in` 是每个词几维。（书里用 `torch.stack((inputs, inputs))` 造了个 `[2,6,3]` 的假 batch 来测。）
- `keys.transpose(1, 2)`：因为多了 batch 维，转置不能再用 `.T`（那会把 batch 维也转乱），而要**只对最后两维转**——第 1 维和第 2 维对调。
- `masked_fill_`（带下划线）：**原地**修改，省内存。`[:num_tokens, :num_tokens]` 把预存的大掩码裁到当前句长。
- `register_buffer('mask', ...)`：把掩码登记成"缓冲区"。它不是可训练参数（不需要梯度），但**会随模型一起搬到 GPU**——省得你手动管设备，避免"张量在 CPU、模型在 GPU"这类报错。

<a name="加油站5"></a>
> ### 🐍 加油站 ⑤ — 形状操作三件套：`.T` / `.transpose` / `.view`
> - **`.T`**：整体转置，只适合 2 维。3 维以上别用它。
> - **`.transpose(i, j)`**：把第 `i` 维和第 `j` 维对调，其余不动。带 batch 的张量要用它，比如 `transpose(1, 2)`。
> - **`.view(新形状)`**：**不改数据、只换"怎么切"**。比如把 `[2,6,4]`（最后一维 4）重新看成 `[2,6,2,2]`（把 4 拆成 2×2）。多头注意力全靠它把"一大维"切成"多个头"。要求数据在内存里连续，不连续时先 `.contiguous()`。
> 一句话：`.transpose` 换轴的**顺序**，`.view` 换轴的**划分**，数据本身都没动。

> ### 🐍 加油站 ⑥ — `nn.ModuleList` 与 `torch.cat`（下一节要用）
> - **`nn.ModuleList([...])`**：装一串子层的"正规列表"。和普通 Python list 的区别：放进去的层会被 PyTorch **正确登记**（参数能被训练、能搬 GPU）。多头注意力的"简单版"用它装 N 个 `CausalAttention`。
> - **`torch.cat([a, b], dim=-1)`**：把多个张量沿某一维**拼接**。比如两个 `[6,2]` 沿最后一维拼成 `[6,4]`。多头把各头的输出 `cat` 到一起。

---

## 4. 多头注意力：同时用很多套眼睛看（书 3.6）

**多头（multi-head）** 的想法：与其只用一套 Q/K/V"环顾全场"，不如**并行用很多套**（比如 2 套、12 套），各看各的关系——有的盯语法、有的盯指代、有的盯主题——最后把结果拼起来。

### 4.1 最直观的实现：堆 N 个 `CausalAttention`（代码清单 3.4）

```python
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
```

**逐行拆**：
- `__init__` 里用列表推导造了 `num_heads` 个独立的 `CausalAttention`（每个有自己的一套 Q/K/V 权重），装进 `nn.ModuleList`。
- `forward` 里让每个头各自处理 `x`，再用 `torch.cat(..., dim=-1)` 把它们的输出**沿最后一维拼起来**。
- 所以若每个头输出 `[b,6,2]`、有 2 个头，拼完就是 `[b,6,4]`——`d_out * num_heads`。
- **缺点**：N 个头是**一个接一个**算的（`for head in self.heads`），慢。下一版要把它们**并行**起来。

🔧 **亲手改一下**：把 `num_heads=2` 改成 `num_heads=4`（`d_out` 保持 2），输出最后一维会变成几？（答：2×4=8。）这就是练习 3.2 的反向思考——见文末练习。

### 4.2 高效实现：一个矩阵乘搞定所有头（代码清单 3.5）

真实 GPT 用的是这一版。思路变了：不再堆 N 个独立模块，而是**用一套大的 Q/K/V 投影，再把输出"切"成 num_heads 份**，靠张量变形让所有头在**一次**矩阵乘里并行算完。

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads          # 每个头分到的维度
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)     # 合并各头后再过一层
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys    = self.W_key(x)        # [b, num_tokens, d_out]
        queries = self.W_query(x)
        values  = self.W_value(x)

        # ① 把 d_out 这一大维，切成 (num_heads, head_dim)
        keys    = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values  = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # ② 把"头"这一维换到前面，方便每个头各算各的
        keys    = keys.transpose(1, 2)      # [b, num_heads, num_tokens, head_dim]
        queries = queries.transpose(1, 2)
        values  = values.transpose(1, 2)

        # ③ 每个头内部：和单头注意力一模一样
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # ④ 加权求和，再把各头拼回去
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec
```

**逐行拆（盯住形状变化，这是全章最绕的一段）**：
- `assert d_out % num_heads == 0`：`d_out` 必须能被 `num_heads` 整除，因为要平分给每个头。`head_dim = d_out // num_heads` 是每个头的维度。
- **① `.view(...)`**：把投影后的 `[b, num_tokens, d_out]` 重新切成 `[b, num_tokens, num_heads, head_dim]`——同一批数据，只是把"`d_out` 这一大列"重新理解成"`num_heads` 个、每个 `head_dim` 维"。数据没动，只是换了划分（见加油站 ⑤）。
- **② `.transpose(1, 2)`**：换轴顺序，变成 `[b, num_heads, num_tokens, head_dim]`。把 `num_heads` 提到前面，相当于"按头分组"，这样后面一次矩阵乘就能让每个头各算各的、互不串味。
- **③** 这几行**和单头的因果注意力一字不差**（算分数→掩码→缩放 softmax→dropout），只不过现在是在 `[..., num_heads, ...]` 这个多出来的维度上**批量**地对每个头同时做。
- **④** `(attn_weights @ values)` 得到每个头的上下文向量；`.transpose(1,2)` 把头维换回去，`.contiguous().view(...)` 把各头的结果**拼平**回 `[b, num_tokens, d_out]`；最后 `out_proj` 再过一层线性，把各头信息**融合**一下。
- **大局**：`Wrapper` 是"先有 N 个头、再合起来"；这个 `MultiHeadAttention` 是"先有一大层、再在内部切成 N 个头"——**结果等价，但只用一次大矩阵乘，快得多**。这就是真实 GPT 里那一块。

🔧 **亲手改一下**：在 `forward` 里加一行 `print('attn_scores', attn_scores.shape)`，用 `num_heads=2`、`d_out=4`、batch 为 `[2,6,...]` 跑一次。预测它会打印什么形状？（答：`[2, 2, 6, 6]`＝`[b, num_heads, num_tokens, num_tokens]`——每个头一张 6×6 的关系表。）跑出来对上没？

---

## 5. 练习

> 📓 **对照官方答案**：卡住了别硬磕——[在 Colab 打开本章练习解答 notebook ↗](https://colab.research.google.com/github/rasbt/LLMs-from-scratch/blob/main/ch03/01_main-chapter-code/exercise-solutions.ipynb)

### 书上原题

**练习 3.1：让 `SelfAttention_v1` 和 `v2` 输出一致。**
两者输出不同，只因初始权重不同（`v2` 的 `nn.Linear` 用了不同初始化）。把 `v2` 的权重搬到 `v1` 里，让两者输出相同。
<details><summary>解题思路（点开）</summary>

关键提示（书里给的）：**`nn.Linear` 内部以"转置形式"存权重**。所以 `v1.W_query`（形状 `[d_in, d_out]`）应当等于 `v2.W_query.weight` 的**转置**。逐个赋值：

```python
sa_v1.W_query = nn.Parameter(sa_v2.W_query.weight.T)
sa_v1.W_key   = nn.Parameter(sa_v2.W_key.weight.T)
sa_v1.W_value = nn.Parameter(sa_v2.W_value.weight.T)
```
赋完再跑，两者输出应完全一致。**收获**：理解了 `nn.Linear` 存的 `weight` 和"我们心里那个 `x @ W`"差了一个转置——这是以后读各种源码会反复遇到的坑。
</details>

**练习 3.2：让多头输出变回 2 维。**
用 `MultiHeadAttentionWrapper(..., num_heads=2)`，但希望输出的上下文向量是 **2 维**而不是 4 维。提示：**不用改类，只改一个传入参数**。
<details><summary>解题思路（点开）</summary>

输出维度 = `d_out * num_heads`。要 `num_heads=2` 不变、输出为 2，就得让 `d_out = 1`。所以把构造时的 `d_out` 从 2 改成 **1** 即可：每个头输出 1 维，两个头拼起来正好 2 维。**收获**：把"输出维度由谁决定"这件事彻底想清楚了。
</details>

### 本册自测题（改了会怎样）

1. **关掉缩放**：在 `SelfAttention_v2.forward` 里把 `/ keys.shape[-1]**0.5` 删掉，输出会变吗？把 `d_out` 调到很大（比如 100）再对比，你能观察到"分布变极端"的迹象吗？（联系 2.2 节"为什么要缩放"。）
2. **掩码方向反了会怎样**：把 `CausalAttention` 里的 `torch.triu(..., diagonal=1)` 改成 `diagonal=0`，对角线会被一起掩掉。想一想：这会让每个词连"看自己"都不行——预测还合理吗？跑一下看权重矩阵长啥样。
3. **头数与维度**：用 `MultiHeadAttention`，设 `d_out=6`，分别试 `num_heads=1, 2, 3, 6`。哪些能跑、哪些报错？报错信息和那句 `assert` 对得上吗？

---

## 6. 本章小结

1. 注意力永远是同一套三步：**算分数（Q·K）→ softmax 归一化 → 按权重把 V 加权求和**。后面再花哨的版本都没逃出这三步。
2. 玩具版（无权重）→ 加可训练的 Q/K/V → 加因果掩码（不许看未来）→ 加 dropout → 扩成多头——书是**一层层加料**爬上来的，每一层只动一点点。
3. **缩放**（除以 √d_k）是为了训练稳定；**因果掩码**（填 −∞ 再 softmax）是为了不泄露未来；**多头**是为了同时从多个角度看关系。
4. 工程上反复出现的动作：用 **`@` 矩阵乘消掉 for 循环**、用 **`.view`/`.transpose` 把维度切成多个头**、用 **`register_buffer` 让掩码跟着模型跑**。
5. 你刚刚走过的 `MultiHeadAttention` 类，**就是真实 GPT 里原封不动的那一块**——第 4 章会把它当积木塞进 Transformer block。

---

## 7. 能改自检清单（全勾＝过关）

- [ ] 我能不看书说出注意力的三步，并指出每步对应代码哪一行。
- [ ] 给我一段注意力代码，我能 `print` 出每个中间张量的形状并解释为什么是这个形状。
- [ ] 我能解释"为什么除以 √d_k"和"为什么填 −∞ 而不是乘 0"。
- [ ] 我能说清因果掩码挡的是什么、不挡会出什么问题。
- [ ] 我能讲明白多头的两种实现（堆 N 个 vs 切成 N 份）为什么等价。
- [ ] 我能独立完成练习 3.1、3.2，并预测自测题里每个改动的结果再验证。

---

## 8. 通往下一章

注意力是"零件"，但单独一个注意力还不是 GPT。**第 4 章**会给它配上前馈网络、残差连接、层归一化，打包成一块**Transformer 积木**，然后**重复堆 N 层**——你这一章造的 `MultiHeadAttention`，会被原样 import 进去当心脏。

> 带走主线：注意力让每个词"环顾全场、拉取相关信息"，**最终目的仍然是——把下一个词预测得更准。**
