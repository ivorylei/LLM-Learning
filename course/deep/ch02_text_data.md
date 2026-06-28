# 深化册 · 第 2 章　文字变数字：把一句话变成模型能吃的张量

> **对应**：概念课 **第 2 天** ｜ 书 **第 2 章（处理文本数据）**
> **一句话直觉**：计算机只会算数字、不认字。所以第一步永远是把文字**切成块（token）→ 查成编号（ID）→ 变成高维空间里的点（嵌入向量）→ 再标上位置**。意义，就是位置。
> **这一章你将亲手得到**：一条完整的输入流水线——给它一段文本，吐出 `input_embeddings` 张量 `[批, 词数, 维度]`，这正是第 3 章注意力的进料口。
>
> **前置**（卡住就翻 `ch00`）：张量与 `.shape`、`nn.Embedding` 是什么、广播相加。

---

[![在 Colab 中打开本章](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rasbt/LLMs-from-scratch/blob/main/ch02/01_main-chapter-code/ch02.ipynb)

> 配套可运行 notebook（rasbt/LLMs-from-scratch）。**边读边跑、跟着每节的「🔧 亲手改一下」改一改。**

## 0. 三分钟回顾（概念课直觉）

模型不能直接吃文字，要走一条四步流水线（概念课第 2 天那张图）：

```
文字 ──①分词──► token ──②查表──► token ID ──③嵌入──► 向量 ──④加位置──► 带位置的向量
```

- **token** 不完全等于"词"——可能是词、半个词、一个字、一个标点。
- **嵌入**之后，意思相近的词，向量自动靠得近（"国王"和"女王"是邻居）——**意义即位置**。
- **位置编码**告诉模型词的先后（"狗咬人"≠"人咬狗"）。

本章把这四步**一段段写成代码**。

---

## 1. 最朴素的分词器：先体会"切词 + 建表"（书 2.2–2.4）

### 1.1 切词、建词表、编码/解码

先用最土的办法理解原理：按标点和空格把文本切开，给每个不同的 token 编个号，存成一本"字典"。书里在小说《The Verdict》上建了个 1130 词的小词表，分词器长这样（处理未知词的 V2 版）：

```python
import re

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab                              # 词 -> 编号
        self.int_to_str = {i: s for s, i in vocab.items()}   # 编号 -> 词
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int       # 没见过的词
                        else "<|unk|>" for item in preprocessed]  # 换成 <|unk|>
        return [self.str_to_int[s] for s in preprocessed]
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        return re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)     # 标点前去空格
```

**逐行拆**：
- `encode`：① `re.split(...)` 按标点/空格切开；② 去掉空白碎片；③ 把词表里没有的词换成特殊 token `<|unk|>`（unknown）；④ 逐个查 `str_to_int` 变成编号。
- `decode`：反过来，把编号查回词、拼成句子，再用正则把标点前多余的空格去掉。
- **文字到这里就变成了一串整数。** 这就是流水线的前两步（分词 + 查表）。

### 1.2 朴素分词器的两个毛病 → 引出 BPE

```python
text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace."
ids = tokenizer.encode(text)
print(tokenizer.decode(ids))
# <|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>.
```

`Hello` 和 `palace` 没在小说里出现过，全变成了 `<|unk|>`——**信息丢了**。这暴露了朴素方法的死穴：词表是固定的，**碰到没见过的词就抓瞎**。真实的 GPT 用更聪明的办法——**BPE**。

> **顺带认识两个特殊 token**（GPT 也在用）：
> - `<|endoftext|>`：拼接多篇独立文档时插在中间，告诉模型"这里是一篇的结束、下一篇的开始"。也兼作填充用。
> - `<|unk|>`：未知词占位（但下面会看到，真正的 GPT 用 BPE，根本不需要它）。

---

## 2. BPE：GPT 真正用的分词法（书 2.5）

**BPE（字节对编码）** 的聪明之处：词表里不光有完整的词，还有大量**子词碎片**和单个字符。碰到没见过的词，就拆成认识的碎片拼出来——**所以任何词都能编码，永远不会抓瞎**。GPT-2/3、ChatGPT 用的就是它。

书里不手写 BPE（实现复杂），而是用 OpenAI 开源的高效库 **tiktoken**：

```python
# 一次性安装：pip install tiktoken
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")        # 加载 GPT-2 用的 BPE 分词器

text = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces"
        " of someunknownPlace.")
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
# [15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252,
#  18250, 8812, 2114, 286, 617, 34680, 27271, 13]
print(tokenizer.decode(integers))                # 完美还原，连生造词都行
```

**逐行拆**：
- `get_encoding("gpt2")`：拿到和 GPT-2 一模一样的分词器（词表大小 **50257**）。
- `encode(...)`：编码成 token ID。注意那个生造的词 `someunknownPlace` 也被正确编码了——BPE 把它拆成了几个子词碎片（`34680`、`27271` 等）。
- `<|endoftext|>` 的 ID 是 **50256**（词表里最大的那个号）。
- **对比第 1 节**：朴素分词器遇到生词只能 `<|unk|>`；BPE 把生词拆成碎片，**信息不丢**。这就是 GPT 不需要 `<|unk|>` 的原因。

🔧 **亲手改一下 / 练习 2.1**：用这个分词器编码生造词 `"Akwirw ier"`，打印每个 token ID；再对每个 ID 单独 `decode`，看它被拆成了哪些碎片；最后整体 `decode` 看能不能拼回原词。（体会"BPE 怎么用碎片拼出没见过的词"。）

<a name="g-tiktoken"></a>
> ### 🐍 加油站 — 为什么 token ≠ 字数
> 概念课"上下文窗口 128K""按 token 收费""数不清 strawberry 里几个 r"，根都在这里：模型的计数单位是 **token（碎片）**，不是字、不是词。一段中文的 token 数和字数也不一样。你跑几次 `tokenizer.encode` 数数长度，就对"token 到底多大"有手感了——这比记定义有用得多。

---

## 3. 滑动窗口：造"输入→目标"训练对（书 2.6）

模型靠"猜下一个词"训练。那"题目"和"答案"长什么样？很简单：**输入是一段 token，目标就是它整体右移一位**——每个位置的"答案"就是它的下一个 token。

```python
enc_sample = enc_text[50:]          # 一段已经 BPE 编码好的 token
context_size = 4
x = enc_sample[:context_size]       # 输入：前 4 个
y = enc_sample[1:context_size+1]    # 目标：右移一位的 4 个
# x: [290, 4920, 2241, 287]
# y:      [4920, 2241, 287, 257]

for i in range(1, context_size+1):
    print(enc_sample[:i], "---->", enc_sample[i])
# [290]                 ----> 4920
# [290, 4920]           ----> 2241
# [290, 4920, 2241]     ----> 287
# [290, 4920, 2241, 287]----> 257
```

**逐行拆**：箭头左边是"已经看到的词"，右边是"该预测的下一个词"。**一句话天然自带海量练习题**——这正是概念课第 5 天说的"自监督"：不需要人工标注，文本自己就是题目+答案。

把这套逻辑包成标准的 `Dataset`（结构和 `ch00` §4 的 `ToyDataset` 一模一样）：

```python
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids, self.target_ids = [], []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i:i + max_length]))
            self.target_ids.append(torch.tensor(token_ids[i + 1: i + max_length + 1]))
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128,
                         shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last, num_workers=num_workers)
```

**逐行拆**：
- `__init__` 里用**滑动窗口**切：每次取 `max_length` 个 token 当输入、右移一位的 `max_length` 个当目标，窗口每次往后挪 `stride` 步。
- `target_chunk` 比 `input_chunk` 整体右移一位——这就是上面"输入→目标"的批量版。
- `max_length`＝一次喂多少 token（上下文长度），`stride`＝窗口步长。**`stride = max_length` 时各窗口不重叠**（不浪费也不重复，书里推荐这样以减少过拟合）。
- 这两段就是 `ch00` 的 `Dataset/DataLoader` 套路在文本上的应用——**模板完全一样**。

🔧 **亲手改一下 / 练习 2.2**：用 `max_length=2, stride=2` 和 `max_length=8, stride=2` 各跑一次 `create_dataloader_v1`，打印第一批，看输入/目标的形状和重叠情况怎么变。

---

## 4. 词嵌入：把编号变成"空间里的点"（书 2.7）

光有 token ID 还不够——编号 5 和 6 并不意味着意思相近。**嵌入层**把每个 ID 映射成一串浮点数（向量），训练后意思相近的词向量会自动靠近。

```python
input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6        # 玩具词表（真实 BPE 是 50257）
output_dim = 3        # 每个词 3 维（GPT-3 是 12288 维）

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)        # 6×3 的随机权重矩阵
print(embedding_layer(torch.tensor([3])))   # 查第 3 号 -> [-0.4015, 0.9666, -1.1481]
print(embedding_layer(input_ids))    # 一次查 4 个 -> 4×3 矩阵
```

**逐行拆**：
- `nn.Embedding(vocab_size, output_dim)`：一张 `[词表大小, 维度]` 的权重表，每个 token ID 占一行。一开始是**小随机数**，训练中被优化。
- `embedding_layer(torch.tensor([3]))`：**本质就是查表**——拿 ID 当行号，取出对应那一行。结果恰好等于权重矩阵的第 4 行（ID 3，从 0 数起）。
- 喂一串 ID，就返回对应的一摞向量。**这就是"意义即位置"落地**：每个词成了高维空间里的一个点。

<a name="g-embedding"></a>
> ### 🐍 加油站 — `nn.Embedding` 就是"可训练的查表"
> 别被名字唬住。它干的事就是"按行号取行"，和查字典一样。特别之处只有两点：① 这张表的数字是**可训练参数**，会在训练中被慢慢调好，让相近的词靠近；② 它等价于"one-hot 编码 + 矩阵乘"，但快得多。所以嵌入层既是"查表"，也是一个能被 `.backward()` 优化的神经网络层。

---

## 5. 位置编码：告诉模型谁先谁后（书 2.8）

嵌入层有个缺点：**同一个词不管排在第几位，向量都一样**——可注意力机制本身也不分先后（第 3 章会看到）。"狗咬人"和"人咬狗"用词相同、意思相反，模型得知道顺序。办法：再加一层**位置嵌入**，按位置给每个 token 加上"它排第几"的信息。

```python
vocab_size, output_dim = 50257, 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8,
                                  max_length=max_length, stride=max_length, shuffle=False)
inputs, targets = next(iter(dataloader))
print(inputs.shape)                       # torch.Size([8, 4])  8 句、每句 4 token

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)             # torch.Size([8, 4, 256])  每 token 变 256 维

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)               # torch.Size([4, 256])  4 个位置各一个向量

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)             # torch.Size([8, 4, 256])
```

**逐行拆**：
- `token_embeddings`：把 `[8, 4]` 的 ID 嵌成 `[8, 4, 256]`——8 句、每句 4 词、每词 256 维。
- `pos_embedding_layer`：另一张嵌入表，但按**位置**（0,1,2,3）查。`torch.arange(context_length)` 生成 `[0,1,2,3]`，查出 4 个位置向量 `[4, 256]`。
- `token_embeddings + pos_embeddings`：**广播相加**——`[4, 256]` 的位置向量自动加到 8 句里每一句的对应位置上。结果 `input_embeddings` 形状仍是 `[8, 4, 256]`。
- **这就是终点**：`input_embeddings` 就是模型的进料口，下一章注意力直接吃它。

> GPT 用的是**绝对位置嵌入**（每个位置一个可学习的向量，随训练优化），不是原始 Transformer 那种固定公式。书里这样实现，和真实 GPT 一致。

<a name="g-broadcast"></a>
> ### 🐍 加油站 — 广播（broadcasting）
> `[8,4,256] + [4,256]` 为什么能加？PyTorch 的**广播**会自动把小张量"复制对齐"到大张量的形状上：这里把 `[4,256]` 的位置向量，原样加到 8 句话的每一句上。规则：从最后一维往前比，维度相等或其中一个为 1 就能广播。拿不准时——又是那句话——`print(shape)` 看结果对不对。

---

## 6. 练习

> 📓 **对照官方答案**：卡住了别硬磕——[在 Colab 打开本章练习解答 notebook ↗](https://colab.research.google.com/github/rasbt/LLMs-from-scratch/blob/main/ch02/01_main-chapter-code/exercise-solutions.ipynb)

### 书上原题
- **练习 2.1（BPE 拆生词）**：见 §2 的 🔧。重点体会 BPE 怎么用碎片拼出没见过的词。
- **练习 2.2（不同步长/窗口）**：见 §3 的 🔧。重点体会 `max_length` 和 `stride` 怎么影响输入/目标的形状与重叠。

### 本册自测题
1. 把 §4 的 `output_dim=3` 改成 `output_dim=5`，`embedding_layer.weight` 形状会变成什么？查一个 ID 得到几维向量？
2. §5 里如果**只用 `token_embeddings`、不加 `pos_embeddings`**，模型还能区分"狗咬人 / 人咬狗"吗？为什么？
3. 用 `tiktoken` 编码同一句话的中文和英文版本，比较 token 数。能解释"为什么按 token 收费时，中文有时更贵/更便宜"吗？

---

## 7. 本章小结

1. 输入流水线四步：**分词 → 查 ID → 嵌入 → 加位置**，产物是 `input_embeddings` 张量 `[批, 词数, 维度]`。
2. 朴素分词器遇生词就 `<|unk|>` 丢信息；**BPE（tiktoken）把生词拆成子词碎片**，任何词都能编码，这是 GPT 的做法（词表 50257）。
3. **滑动窗口**把文本切成"输入→目标右移一位"的训练对——这就是"自监督"，文本自带答案。
4. **`nn.Embedding` 是可训练的查表**，把 token ID 变成空间里的点；**位置嵌入**再补上先后顺序，两者相加（广播）得到最终输入。
5. token 的计数单位不是字——上下文窗口、按 token 收费、数错字母，根都在这。

---

## 8. 能改自检清单（全勾＝过关）

- [ ] 我能说出输入流水线四步，并指出每步对应代码哪一段。
- [ ] 我能解释 BPE 比朴素分词器强在哪（用"生词"举例）。
- [ ] 我能说清"输入→目标"训练对是怎么从一段文本切出来的，以及 `stride` 的作用。
- [ ] 我能解释 `nn.Embedding` 本质是查表、为什么它是可训练的。
- [ ] 我能说清为什么要加位置嵌入、以及广播相加是怎么对齐形状的。
- [ ] 我能独立完成练习 2.1、2.2，并预测自测题里每个改动的结果。

---

## 9. 通往下一章

`input_embeddings` 已经备好——一摞带位置的向量。**第 3 章**（你已经做过的注意力）就是吃这摞向量、让每个词"环顾全场、拉取相关信息"。入口造完，下一步直奔心脏。

> 带走主线：把文字变成向量，只是为了让那台机器能开始干它唯一的活——**预测下一个词**。
