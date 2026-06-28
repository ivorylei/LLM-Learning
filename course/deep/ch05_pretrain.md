# 深化册 · 第 5 章　预训练：让 GPT 自己"滚下山"变聪明

> **对应**：概念课 **第 5 天** ｜ 书 **第 5 章（在未打标签的数据上进行预训练）**
> **一句话直觉**：用亿万次"猜下一个词"把半个互联网压进参数；文本自带答案＝**自监督**；损失一路**滚下山**，模型就一点点变聪明。
> **这一章你将亲手得到**：一个**能训练、会随损失下降而变好**的 GPT——以及一手"作弊技巧"：**直接加载 OpenAI 的 GPT-2 权重**，让你的模型瞬间拥有几十万美元算力训出来的本事。
>
> **前置**（卡住就翻回去）：
> - 训练循环五步母版（前向→算损失→清梯度→反向→更新）、交叉熵 → [第 0 章 §5](ch00_python_pytorch.md)
> - `GPTModel`、`generate_text_simple`（逐词生成）→ [第 4 章](ch04_gpt.md)
> - 本章新出现的工具（交叉熵/采样/`view`/`device`/`.eval()`）→ 见各节「🐍 加油站」

---

[![在 Colab 中打开本章](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rasbt/LLMs-from-scratch/blob/main/ch05/01_main-chapter-code/ch05.ipynb)

> 配套可运行 notebook（rasbt/LLMs-from-scratch）。**边读边跑、跟着每节的「🔧 亲手改一下」改一改。**

## 0. 三分钟回顾（概念课直觉）

前四章你已经把 GPT 这台机器**造**出来了：文字变数字（第 2 章）、注意力（第 3 章）、堆成完整的 `GPTModel`（第 4 章）。但它现在是**一台没学过任何东西的空壳**——给它"Every effort moves you"，它会吐出一串乱码。

这一章解决的就是一件事：**怎么让这台空壳变聪明**。答案是概念课第 5 天那条主线——

> **预训练 = 让模型做亿万次"猜下一个词"的填空题，猜错就挨罚（损失变大），据此微调参数，直到猜得越来越准。**

最妙的地方在于：**这些填空题的答案，文本自己就带着**。给模型看 "Every effort moves"，正确答案 "you" 就是原文里的下一个词——不需要任何人工标注。这就是**自监督学习（self-supervised learning）**：数据自己监督自己。半个互联网的文本，就是几亿道自带答案的填空题。

整章的剧情线（也是书 5.1→5.5 的顺序）：

1. **先有一把尺**：怎么用数字衡量"模型生成的文本有多烂"？→ **交叉熵损失**。
2. **再让它滚下山**：写一个训练循环，让损失一路往下掉。→ `train_model_simple`。
3. **让它会说人话**：训练后，怎么生成不那么死板的文本？→ **温度 + top-k 采样**。
4. **存下来 / 站在巨人肩上**：保存权重；以及——**别自己练了**，直接加载 OpenAI 练好的 GPT-2 权重。

你会发现第 2 步那个训练循环，和第 0 章 §5 的"母版"**结构一模一样**——只是模型从玩具网络换成了 GPT，数据从几个点换成了文本。**全书所有训练都是那五步。**

---

## 1. 先把"文本 ⇄ Token"的来回路打通（书 5.1.1）

### 要解决什么

训练前先热身：把第 4 章的 `generate_text_simple` 拿出来跑一遍，确认这台空壳确实"还不会说话"。但每次手动 `encode`、`decode` 很啰嗦，所以书里先封装两个小助手，**专管文本和 Token ID 之间的来回翻译**。

先把模型立起来（沿用第 4 章的配置，只把上下文长度从 1024 缩到 256，好让笔记本也能训）：

```python
import torch
from chapter04 import GPTModel          # 第 4 章造好的 GPT

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # 词汇表大小
    "context_length": 256,  # 上下文长度（原版 1024，这里缩短省算力）
    "emb_dim": 768,         # 嵌入维度
    "n_heads": 12,          # 注意力头数
    "n_layers": 12,         # Transformer 层数
    "drop_rate": 0.1,       # dropout 比例
    "qkv_bias": False,
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()                # 先设成评估模式（关掉 dropout）
```

### 贴书真实代码（代码清单 5.1）

```python
import tiktoken
from chapter04 import generate_text_simple   # 第 4 章的逐词生成函数

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)   # [n] -> [1, n] 加一个批维
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)                # [1, n] -> [n] 去掉批维
    return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"],
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
```

跑出来是一串乱码：

```
Output text:
 Every effort moves you rentingetic wasn? refres RexMeCHicular stren
```

### 逐行拆

- `tokenizer.encode(...)`：分词器把字符串切成一串整数 Token ID（第 2 章那套 BPE 分词）。`allowed_special` 允许出现 `<|endoftext|>` 这个特殊标记。
- `.unsqueeze(0)`：模型吃的是**带批维**的张量 `[批大小, 词数]`。一句话只有 `[n]` 个 token，`unsqueeze(0)` 在最前面塞一个长度为 1 的维度，变成 `[1, n]`——相当于"这一批只有 1 句话"。
- `token_ids_to_text` 反着来：`.squeeze(0)` 把那个长度 1 的批维去掉，再 `.decode` 翻回人话。
- `generate_text_simple`（第 4 章写的）做的事：每次拿当前序列喂模型 → 取最后一个位置概率最高的 token → 接到序列尾巴 → 重复 `max_new_tokens` 次。**这就是"预测下一个词"循环本身。**

> **为什么输出是乱码？** 因为 `model` 刚用随机数初始化，啥也没学过。它现在每一步"猜下一个词"基本是瞎猜。这正是我们要训练它的起点——**先承认它很烂，再想办法量化"有多烂"。**

🔧 **亲手改一下**：把 `start_context` 换成别的英文短句（比如 `"The sky is"`），`max_new_tokens` 改成 20。预测：输出会更连贯吗？（不会——没训练前换什么开头都是乱码。这一步是为了让你**亲眼确认起点的烂**，训练后再回头对比。）

---

## 2. 造一把尺：交叉熵损失（书 5.1.2）

### 要解决什么

"乱码"是肉眼判断，机器学不了。我们需要一个**数字**，能说"模型现在有多烂"，而且这个数字要**可以被优化**（越小越好，训练就朝它最小化）。这把尺就是**交叉熵损失（cross-entropy loss）**——它也正是第 0 章训练循环里那个 `F.cross_entropy`，只是这次喂的是文本。

书里没有一上来就调 PyTorch 的现成函数，而是**手动把交叉熵拆成 6 步走一遍**，让你看清它到底在算什么。我们跟着走。

### 第一步：准备"输入"和"答案"

```python
inputs = torch.tensor([[16833,  3626,  6100],   # ["every effort moves",
                       [   40,  1107,   588]])  #  "I really like"]
targets = torch.tensor([[3626,  6100,   345],   # [" effort moves you",
                        [1107,   588, 11311]])  #  " really like chocolate"]
```

**逐行拆**：`inputs` 是两句话（每句 3 个 token）。`targets` 是**同样的句子，但整体向右挪了一个位置**——`inputs` 第一个词是 16833（every），`targets` 第一个词是 3626（effort），正好是 inputs 的第二个词。

> **这就是"自监督"的全部秘密。** 答案不用人标，**把原文往左错一位就是答案**：看到 "every"，该猜 "effort"；看到 "every effort"，该猜 "moves"……targets 永远是 inputs 的"下一个词"。第 2 章的数据加载器干的就是这个错位。

### 第二步：模型给出概率，挑出"答案那一格"的概率

```python
with torch.no_grad():               # 只看不学，不用算梯度
    logits = model(inputs)
probas = torch.softmax(logits, dim=-1)
print(probas.shape)   # torch.Size([2, 3, 50257])
```

`probas` 形状 `[2, 3, 50257]`：2 句话 × 每句 3 个位置 × 每个位置在 **50257 个词**上的概率分布。

现在，对每个位置，**只挑出"正确答案那个词"被分到多少概率**：

```python
text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 1:", target_probas_1)
# Text 1: tensor([7.4541e-05, 3.1061e-05, 1.1563e-05])

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2:", target_probas_2)
# Text 2: tensor([1.0337e-05, 5.6776e-05, 4.7559e-06])
```

**逐行拆**：`probas[0, [0,1,2], targets[0]]` 是"花式索引"——对第 0 句的第 0、1、2 个位置，分别取出 `targets` 指定的那个词的概率。结果都小到 `0.00007` 这种数量级。

> **为什么这么小？** 50257 个词，瞎猜每个约 `1/50257 ≈ 0.00002`。模型现在给正确答案的概率就在这个"瞎猜水平"附近。**训练的目标，就是把"答案那一格"的概率一路抬高。** 抬得越高，模型越确信下一个词该是它——这就是概念课说的"让正确 token 的概率最大化"。

### 第三步到第六步：取对数 → 求平均 → 取负 = 交叉熵

```python
# 步骤 4：取对数（概率连乘在数学上不好处理，取对数变成连加）
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)
# tensor([ -9.5042, -10.3796, -11.3677, -11.4798,  -9.7764, -12.2561])

# 步骤 5：求平均
avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)   # tensor(-10.7940)

# 步骤 6：取负 —— 这个值就叫"交叉熵损失"
neg_avg_log_probas = avg_log_probas * -1
print(neg_avg_log_probas)   # tensor(10.7940)
```

**逐行拆**：
- **取对数**：概率都在 0~1 之间，连乘会越乘越小到电脑存不下。取对数后，"乘"变"加"，数值好管理（这是数学优化的常规操作）。概率越接近 1，对数越接近 0；概率越小，对数是越负的大负数。
- **求平均**：把 6 个位置的对数概率平均成一个数 `-10.7940`。
- **取负**：深度学习的惯例是"**最小化**一个损失"，而我们想"**最大化**对数概率"。给它乘 -1，最大化就变成了最小化。这个 `10.7940` 就是**交叉熵损失**。目标：把它一路压向 0。

### 一行顶六步：PyTorch 的 `cross_entropy`

手动那 6 步只是为了讲清原理。实战直接调 PyTorch，**一行搞定全部六步**：

```python
# cross_entropy 要求 logits 是二维 [样本数, 类别数]，targets 是一维 [样本数]
logits_flat = logits.flatten(0, 1)   # [2, 3, 50257] -> [6, 50257]
targets_flat = targets.flatten()     # [2, 3]        -> [6]

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss)   # tensor(10.7940)  ← 和手动六步的结果一模一样
```

**逐行拆**：
- `logits.flatten(0, 1)`：把"批维"和"词数维"**揉成一维**。原来 `[2句, 3词, 50257]`，揉完是 `[6个token, 50257]`——把"2 句各 3 个词"摊平成"6 个待预测的 token"。
- `targets.flatten()`：`[2, 3]` 摊平成 `[6]`，对上那 6 个 token 的正确答案。
- `cross_entropy(logits_flat, targets_flat)`：注意**直接喂 logits（没过 softmax）**——这个函数内部自带 softmax、取对数、挑答案、求平均、取负，**那六步全包了**。结果 `10.7940`，和手算分毫不差。

> ### 🐍 加油站 ① — 交叉熵到底在算什么？为什么喂 logits 不喂概率？
> **交叉熵（cross-entropy）** 衡量两个概率分布有多"不像"：一个是"真实答案"（正确 token 概率 100%，其余 0），一个是"模型的预测分布"。模型把概率压在正确答案上越多，交叉熵越小。所以它天然就是"猜下一个词"任务的打分尺。
> **三个等价说法**（同一个东西的不同名字，源码里换着用，别被绕晕）：交叉熵损失 ＝ 负平均对数概率 ＝ negative log likelihood。
> **为什么 `cross_entropy` 要喂 logits 而不是 softmax 后的概率？** 因为它**内部自己做 softmax**。如果你先 softmax 再喂进去，等于做了两次，不仅多算还**数值不稳**（容易溢出）。记一条铁律：`cross_entropy` 永远喂**原始 logits**。这和第 0 章 §5 母版里那行 `F.cross_entropy(logits, labels)` 是同一个道理。

> **困惑度（perplexity）顺带一提**：`perplexity = torch.exp(loss)`，是损失的另一种说法，更直观。`exp(10.7940) ≈ 48725`，意思是"模型此刻纠结得相当于在 4.8 万个词里瞎挑下一个"。损失越低，困惑度越低，模型越笃定。

🔧 **亲手改一下**：把 `targets` 里某个数字改成 `inputs` 对应位置完全对不上的词，重算 `loss`。预测：loss 会变大还是变小？（变大——你把"正确答案"换成了模型更不看好的词，惩罚就更重。）

---

## 3. 把尺架到整个数据集上：`calc_loss_loader`（书 5.1.3）

### 要解决什么

上面只量了两句话的损失。真正训练要在**整个训练集**上算损失，还要留一份**验证集**看模型有没有"光会背、不会举一反三"（过拟合）。这一节把"算一个批次的损失"和"算整个加载器的损失"封装成两个函数。

数据用伊迪丝·华顿的短篇小说《The Verdict》——一个**只有 5145 个 token 的迷你数据集**，小到笔记本几分钟能训完（真实预训练要烧几十万美元，见下方加油站）。按 9:1 切成训练 / 验证：

```python
file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]   # 前 90% 训练
val_data   = text_data[split_idx:]   # 后 10% 验证

from chapter02 import create_dataloader_v1   # 第 2 章的数据加载器
torch.manual_seed(123)
train_loader = create_dataloader_v1(
    train_data, batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True, shuffle=True, num_workers=0,
)
val_loader = create_dataloader_v1(
    val_data, batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False, shuffle=False, num_workers=0,
)
```

每个批次出来是 `x.shape = [2, 256]`、`y.shape = [2, 256]`——2 句、每句 256 个 token，`y` 还是 `x` 错位一格的"答案"。

### 贴书真实代码（代码清单 5.2）

```python
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)      # 把数据搬到模型所在的设备
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches
```

### 逐行拆

- `calc_loss_batch`：**就是上一节那一行 `cross_entropy`**，只是前面加了 `.to(device)` 把数据搬到模型所在的硬件（CPU 或 GPU），免得"数据在 CPU、模型在 GPU"对不上而报错。这个函数是整个训练的**心脏**——后面训练循环、评估全靠它。
- `calc_loss_loader`：遍历加载器里每个批次，对每批调 `calc_loss_batch`，把损失**累加再求平均**，得到"整个数据集上的平均损失"。
  - `loss.item()`：把只含一个数的张量取成普通 Python 浮点数（累加用，省得拖着计算图占内存）。
  - `num_batches` 参数：可以只算前几个批次。训练中频繁评估时，用它**抽查几批就好**，不用每次跑完整个集，省时间。

跑一下看初始损失：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
with torch.no_grad():                # 评估只看不学
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss   = calc_loss_loader(val_loader, model, device)
print("Training loss:", train_loss)    # 10.98...
print("Validation loss:", val_loss)    # 10.98...
```

两个损失都约 `10.98`——**未训练的模型在训练集和验证集上一样烂**，符合预期。训练后我们就盯着这俩数字往下掉。

> ### 🐍 加油站 ② — `model.to(device)` 与 `with torch.no_grad()`
> - **`device` / `.to(device)`**：`device` 是"在哪块硬件上算"——有显卡就 `"cuda"`，没有就 `"cpu"`。`model.to(device)` 把模型搬过去，`数据.to(device)` 把数据搬过去。**两边必须在同一块设备**，否则报 "expected all tensors to be on the same device"——新手高频错误。
> - **`with torch.no_grad():`**：括起来的代码"只前向、不记账"。平时 PyTorch 会偷偷记录每步运算好待会儿算梯度（占内存）；评估 / 生成时我们**不需要**算梯度，用 `no_grad` 关掉它，**省内存、跑更快**。规律：**训练时不用它，评估和生成时一定用。**

> **预训练有多贵？** 书里给了个数字让你感受规模：训练一个 70 亿参数的 Llama 2，要 **18.4 万 GPU 小时**，在 A100 上按云价算大约 **69 万美元**。所以本章用 5145 个 token 的迷你集只是"教学演示"——真到生产级，没人从零练，**直接加载现成权重**（§7 就干这事）。这正是概念课说的"站在巨人肩上"。

---

## 4. 让损失滚下山：训练循环 `train_model_simple`（书 5.2）

### 要解决什么

万事俱备：有模型、有数据、有损失这把尺。现在写**训练循环**——让模型反复看数据、算损失、调参数，把损失一路压下去。

**关键认知：这个循环和第 0 章 §5 那个"母版"结构完全一样。** 你已经会了，这里只是换上 GPT 和文本。先把母版五步默念一遍：① 前向 ② 算损失 ③ 清梯度 ④ 反向 ⑤ 更新。

### 贴书真实代码（代码清单 5.3）

```python
def train_model_simple(model, train_loader, val_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):            # 把整个数据集过 num_epochs 遍
        model.train()                          # 进入训练模式（开 dropout）
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()                          # ③ 清梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # ①②
            loss.backward()                                # ④ 反向：算梯度
            optimizer.step()                               # ⑤ 更新参数
            tokens_seen += input_batch.numel()             # 记一共看了多少 token
            global_step += 1

            if global_step % eval_freq == 0:               # 每隔几步评估一次
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample(model, tokenizer, device, start_context)  # 每个 epoch 末打印一段样本
    return train_losses, val_losses, track_tokens_seen
```

### 逐行拆（盯住"和母版一样"的那五步）

把它和第 0 章 §5 并排看，**核心五行一字不差**：

| 母版（第 0 章 §5） | 本章 `train_model_simple` | 在干什么 |
|---|---|---|
| `logits = model(features)` | （藏在）`calc_loss_batch` 里的 `model(input_batch)` | ① 前向 |
| `loss = F.cross_entropy(...)` | （藏在）`calc_loss_batch` 里的 `cross_entropy` | ② 算损失 |
| `optimizer.zero_grad()` | `optimizer.zero_grad()` | ③ 清梯度 |
| `loss.backward()` | `loss.backward()` | ④ 反向 |
| `optimizer.step()` | `optimizer.step()` | ⑤ 更新 |

剩下的全是**外围装饰**：
- `for epoch ...`：把整个数据集反复过 `num_epochs` 遍（小数据集才这么干，大数据集通常只过 1 遍）。
- `model.train()`：进训练模式，**打开 dropout**（随机丢神经元防过拟合）。和评估时的 `model.eval()` 配对。
- `tokens_seen += input_batch.numel()`：累计"模型一共看了多少个 token"——规模法则里横轴就是它，看得越多通常越强。
- `if global_step % eval_freq == 0`：每隔 `eval_freq` 步，跑一次评估、记一笔损失、打印进度。**这就是你能在终端看着损失往下掉的那行字。**

`zero_grad` 那行**绝不能省**——PyTorch 梯度是累加的，不清零会把上一批的梯度也算进来，训练直接崩（第 0 章自测题让你亲手删掉体验过）。

### 两个配角：`evaluate_model` 和 `generate_and_print_sample`

```python
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()                          # 评估模式：关 dropout
    with torch.no_grad():                 # 不算梯度
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss   = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()                         # 评估完切回训练模式
    return train_loss, val_loss
```

**逐行拆**：评估时三件套——`model.eval()` 关 dropout、`torch.no_grad()` 关梯度、`num_batches=eval_iter` 只抽查几批（快）。**算完务必 `model.train()` 切回去**，否则接下来的训练就没 dropout 了。这是个容易忘的小坑。

```python
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()
```

**逐行拆**：`evaluate_model` 给你**一个数字**（损失降了没），`generate_and_print_sample` 给你**一段真实文本**（让你肉眼看模型现在会不会说人话）。它复用 §1 的 `text_to_token_ids` + 第 4 章的 `generate_text_simple`，每个 epoch 末吐一段样本。

### 开训！

```python
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer)
```

在 MacBook Air 上约 5 分钟，终端会滚动打印（截取首尾）：

```
Ep 1 (Step 000000): Train loss 9.781, Val loss 9.933
Ep 1 (Step 000005): Train loss 8.111, Val loss 8.339
Every effort moves you,,,,,,,,,,,,.
...
Ep 9 (Step 000080): Train loss 0.541, Val loss 6.393
Every effort moves you?" "Yes--quite insensible to the irony. She wanted
him vindicated--and by me!" ...
Ep 10 (Step 000085): Train loss 0.391, Val loss 6.452
```

**这就是"损失滚下山"的真实样子**：训练损失从 9.781 一路掉到 0.391。生成文本也跟着进化——一开始只会堆逗号（`you,,,,,`），然后重复 "and, and, and"，最后能写出语法正确的句子。**你亲眼看着一台空壳学会了说话。**

> **为什么用 `AdamW` 不用 `SGD`？** AdamW 是 Adam 优化器的改良版，给每个参数自动调学习率、还带"权重衰减"（`weight_decay=0.1`，惩罚过大的权重防过拟合）。它比第 0 章玩具例用的 SGD 更聪明、更稳，是训练 LLM 的标配。换句话说：**母版没变，只是把 `optimizer.step()` 背后那个优化器换了个更好的。**

> **注意验证损失的"分叉"**：训练损失掉到 0.391，验证损失却卡在 6.452 下不去，第 2 个 epoch 后两条线就**分叉**了——这是典型的**过拟合**：模型把这本只有 5145 token 的小说**背下来了**，而不是学会普遍规律（你甚至能在原文里搜到它生成的句子）。这在如此小的数据集上是必然的。真实预训练用海量数据、只过 1 遍，就不会这样。

🔧 **亲手改一下**：把 `num_epochs` 从 10 改成 3，重训。预测：训练损失会停在更高还是更低？生成的文本会更连贯还是更烂？（更高、更烂——滚下山的时间不够。这能让你直观感受 epoch 数和损失的关系。）

---

## 5. 让它别那么死板：温度与 top-k 采样（书 5.3）

### 要解决什么

训练后的模型用 `generate_text_simple` 生成时，每步都用 `argmax` 挑**概率最高**的那个词（叫**贪婪解码**）。问题：同样的开头，它**永远吐一模一样的句子**，而且常常是从训练集里**背出来的原文**。怎么让它生成得更多样、更有"创意"？两个旋钮：**温度（temperature）** 和 **top-k**。

先用一个 9 词的迷你词汇表演示（看得清）：

```python
vocab = {"closer":0, "every":1, "effort":2, "forward":3, "inches":4,
         "moves":5, "pizza":6, "toward":7, "you":8}
inverse_vocab = {v: k for k, v in vocab.items()}

# 假设模型给 "every effort moves you" 之后的下一个词打了这些分（logits）
next_token_logits = torch.tensor([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])
```

`argmax` 永远选最高分的第 3 个词 "forward"。要打破这个死板，第一招：**用采样代替 argmax**。

### 第一招：概率采样（`torch.multinomial`）

```python
probas = torch.softmax(next_token_logits, dim=0)
torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item()
print(inverse_vocab[next_token_id])   # forward
```

**逐行拆**：`torch.multinomial` **按概率比例抽签**——概率高的词更容易被抽中，但不是每次都抽它。"forward" 概率最高，所以**大多数**时候还是它，但偶尔会蹦出别的词。重复抽 1000 次看分布：

```python
def print_sampled_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")

print_sampled_tokens(probas)
# 582 x forward, 343 x toward, 73 x closer, 2 x inches ...
```

1000 次里 "forward" 抽中 582 次，但 "toward""closer" 也各分到一些。**这就是多样性的来源**：模型不再死认一个答案。

> ### 🐍 加油站 ③ — `torch.multinomial`：按概率"抽签"
> `torch.multinomial(probas, num_samples=1)` 把 `probas` 当成一个"加权骰子"：每一面（每个词）被掷中的概率＝它的 softmax 概率。概率 0.58 的面，约 58% 掷中；概率 0.07 的面，约 7% 掷中。
> 对比 `argmax`：`argmax` 是"永远选最大面"（确定、死板），`multinomial` 是"按权重随机掷"（有变化、有惊喜）。LLM 想要"会聊天而不是复读机"，就得用后者。`torch.manual_seed(...)` 固定随机种子，保证你我掷出的序列一样、结果可复现。

### 第二招：温度——调节"骰子有多偏"

```python
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)
```

**逐行拆**：温度就是"**softmax 前把 logits 除以一个数**"，就这么简单，但效果很妙：

- **温度 < 1（如 0.1）**：放大差距，分布变**尖**——几乎总选最高分那个，接近 argmax。**更保守、更确定。**
- **温度 = 1**：原样，等于不缩放。
- **温度 > 1（如 5）**：拉平差距，分布变**平**——各词机会更均等。**更随机、更有创意，但也更容易胡说**（比如蹦出 "every effort moves you pizza"）。

一句话记法：**温度低＝稳重复读，温度高＝放飞自我。**

### 第三招：top-k——只在"靠谱的前 k 个"里抽

温度高了容易胡说。top-k 给它上个保险：**只保留概率最高的 k 个词，其余全部掐死，再在这 k 个里采样。**

```python
top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)   # 取最高的 3 个
# top_logits: [6.7500, 6.2800, 4.5100]  top_pos: [3, 7, 0]

# 把"没进前 3 名"的 logit 全设成 -inf
new_logits = torch.where(
    condition=next_token_logits < top_logits[-1],   # 小于第 3 名的分数
    input=torch.tensor(float('-inf')),              # 就设成负无穷
    other=next_token_logits)                         # 否则保留原值

topk_probas = torch.softmax(new_logits, dim=0)
# tensor([0.0615, 0., 0., 0.5775, 0., 0., 0., 0.3610, 0.])
```

**逐行拆**：
- `torch.topk(logits, 3)`：揪出分数最高的 3 个及其位置。
- `torch.where(条件, A, B)`：条件为真填 A、否则填 B。这里把"分数 < 第 3 名"的位置全填 `-inf`。
- 为什么填 `-inf`？因为 `softmax` 里 `e^(-inf) = 0`——那些被掐死的词概率**自动归零**，剩下 3 个重新归一化为 1。**这跟第 3 章因果掩码"填 -inf 再 softmax"是同一个巧思！**

### 合体：改进版 `generate`（代码清单 5.4）

把温度和 top-k 缝进生成函数，得到真实可用的 `generate`：

```python
def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]          # 只取最近 context_size 个 token
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]                  # 只要最后一个位置的预测

        if top_k is not None:                      # ① top-k 过滤
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits)

        if temperature > 0.0:                      # ② 温度 + 采样
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:                                      # 温度=0 退化成贪婪解码
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:                     # 生成到结束符就停
            break
        idx = torch.cat((idx, idx_next), dim=1)    # 接到序列尾巴
    return idx
```

**逐行拆**：核心还是第 4 章 `generate_text_simple` 那个"取最后位置 → 挑词 → 接上去"的循环，只是中间塞了两道工序——先 top-k 过滤、再温度采样。`temperature=0.0` 时直接走 `argmax`，**退化回原来的贪婪解码**（所以这个函数向下兼容）。`eos_id` 让它遇到结束符提前收工。

```python
torch.manual_seed(123)
token_ids = generate(
    model=model, idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15, context_size=GPT_CONFIG_124M["context_length"],
    top_k=25, temperature=1.4)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
# Every effort moves you stand to work on surprise, a one of us had gone with random-
```

输出明显比贪婪解码更"放飞"，不再是从训练集背原文。

🔧 **亲手改一下**：把 `temperature=1.4` 改成 `0.3`、`top_k=25` 改成 `5`，重跑几次。预测：输出会更稳定（更接近背原文）还是更天马行空？（更稳定——低温 + 小 k 把模型按回"只选最稳的几个词"。这正是练习 5.2 让你体会的：什么场景要低温、什么场景要高温。）

---

## 6. 把成果存下来 / 取回来（书 5.4）

### 要解决什么

训练烧了 5 分钟（真实场景是几天几周）。**绝不能关机就没了**。PyTorch 存模型的推荐做法：保存 `state_dict`——一个"层名 → 参数张量"的字典。

```python
# 只存模型权重
torch.save(model.state_dict(), "model.pth")

# 取回来：先造一个同样结构的空壳，再灌权重
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()
```

**逐行拆**：
- `model.state_dict()`：返回一个字典，键是层名（如 `trf_blocks.0.att.W_query.weight`），值是该层的参数张量。`torch.save` 把它序列化进文件。`.pth` 只是约定俗成的后缀。
- 加载时**先 `GPTModel(GPT_CONFIG_124M)` 造一个结构相同的空壳**（权重是随机的），再 `load_state_dict` 把存下来的权重**覆盖**进去。`map_location=device` 解决"在 GPU 上存、想在 CPU 上读"的设备错位。
- `model.eval()`：加载后若要推理，记得切评估模式关 dropout。

### 想接着训？连优化器一起存

如果以后还要**继续训练**，光存模型不够——`AdamW` 给每个参数都攒了"历史动量"，丢了它优化器会"失忆"、模型可能学崩。所以把模型和优化器一起存：

```python
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, "model_and_optimizer.pth")

# 恢复
checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train()
```

**逐行拆**：把两个 `state_dict` 装进一个字典一起存。恢复时分别灌回 model 和 optimizer，再 `model.train()` 切训练模式，就能从断点无缝续训。

> ### 🐍 加油站 ④ — `.eval()` / `.train()` 到底切了什么？
> 这俩**只影响"训练和推理时行为不同"的零件**，对 GPT 来说主要是 **dropout**：
> - **`.train()`**：dropout **开**——训练时随机丢一部分神经元，逼模型别死记，防过拟合。
> - **`.eval()`**：dropout **关**——推理 / 评估时要稳定、可复现的输出，不能随机丢信息。
> 配套口诀：**训练 `.train()`，评估和生成 `.eval()`，评估完记得切回 `.train()`**（§4 的 `evaluate_model` 就是这么干的）。注意它和 `torch.no_grad()` 是**两件事**：`.eval()` 管 dropout 行为，`no_grad()` 管要不要算梯度——推理时两个通常一起上。

---

## 7. 站在巨人肩上：加载 OpenAI 的 GPT-2 权重（书 5.5）

### 要解决什么

我们自己练的模型只看过一本小说，水平有限。但 **OpenAI 把 GPT-2 的权重公开了**——那是在海量语料上、烧了几十万美元训出来的。我们的 `GPTModel` 架构和 GPT-2 **一模一样**，所以可以**直接把人家的权重灌进我们的模型**，瞬间白嫖一个会说人话的 GPT。这就是概念课第 5 天的收尾直觉：**别自己从零练，加载现成权重。**

> **思路链**：① 下载 OpenAI 权重 → ② 我们的模型结构与之对齐（开 bias、改回 1024 上下文）→ ③ 把权重逐层"搬"进我们的 `GPTModel`。难点全在 ③——因为 OpenAI 的命名和我们的不一样。

### 第一步：下载权重

OpenAI 的权重是用 TensorFlow 存的，下载代码又长又枯燥，书里直接让你下一个现成模块用：

```python
# 需要先 pip install tensorflow>=2.15.0 tqdm>=4.66
from gpt_download import download_and_load_gpt2     # 书附带的下载模块
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

print("Settings:", settings)
# {'n_vocab':50257, 'n_ctx':1024, 'n_embd':768, 'n_head':12, 'n_layer':12}
print("Parameter dictionary keys:", params.keys())
# dict_keys(['blocks', 'b', 'g', 'wpe', 'wte'])
```

**逐行拆**：`settings` 是架构配置（和我们手写的 `GPT_CONFIG_124M` 对应），`params` 是**实打实的权重张量字典**——`wte` 是 token 嵌入、`wpe` 是位置嵌入、`blocks` 是每层 Transformer 的权重。（`download_and_load_gpt2` 是书附带的 `gpt_download.py` 模块，本节只用它、不展开它的下载细节——**书中代码，本节只讲思路**。）

### 第二步：让我们的配置和 GPT-2 对齐

```python
model_configs = {
    "gpt2-small (124M)":  {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)":    {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024})   # OpenAI 用 1024，不是我们缩的 256
NEW_CONFIG.update({"qkv_bias": True})          # OpenAI 的注意力带 bias，得开

gpt = GPTModel(NEW_CONFIG)
gpt.eval()
```

**逐行拆**：要让权重对得上，配置必须和 OpenAI 完全一致。两处关键调整：
- **`context_length` 改回 1024**：之前为省算力缩成 256，但人家是按 1024 训的。
- **`qkv_bias` 开成 True**：现代 LLM 大多不用 Q/K/V 的偏置项，但 GPT-2 用了。为了对齐**必须开**，否则少了一批参数对不上。

> **规模法则一瞥**：那张 `model_configs` 表就是规模法则的缩影——从 124M 到 1558M，**架构一模一样，只是把同样的积木堆更多层（`n_layers`）、每块做得更宽（`emb_dim`）**。参数越多，模型越强。本章代码对四个尺寸都通用，换个 `model_size` 就能加载更大的。

### 第三步：逐层搬运权重（代码清单 5.5）

这是最细的活：OpenAI 的命名（`c_attn`、`c_proj`、`wte`…）和我们的（`W_query`、`out_proj`、`tok_emb`…）对不上，得**手工一一对应**。先写个安全检查小工具：

```python
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))
```

**逐行拆**：`assign` 在搬之前**先核对形状**——形状不一致立刻报错。这是救命的护栏：搬错一个权重，模型就废了，但你可能直到生成乱码才发现。形状检查能当场拦住大部分错位。

然后逐层赋值：

```python
import numpy as np

def load_weights_into_gpt(gpt, params):
    # 两个嵌入层
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):       # 逐个 Transformer 块
        # —— 注意力的 Q/K/V：OpenAI 把三者合存在 c_attn 里，要先切成三份 ——
        q_w, k_w, v_w = np.split(
            params["blocks"][b]["attn"]["c_attn"]["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight   = assign(gpt.trf_blocks[b].att.W_key.weight,   k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            params["blocks"][b]["attn"]["c_attn"]["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias   = assign(gpt.trf_blocks[b].att.W_key.bias,   k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(gpt.trf_blocks[b].att.W_value.bias, v_b)

        # —— 注意力输出投影 ——
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias, params["blocks"][b]["attn"]["c_proj"]["b"])

        # —— 前馈网络两层 ——
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, params["blocks"][b]["mlp"]["c_proj"]["b"])

        # —— 两个层归一化 ——
        gpt.trf_blocks[b].norm1.scale = assign(gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"])

    # 最后的归一化 + 输出头（输出头复用 token 嵌入权重，叫"权重绑定"）
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight  = assign(gpt.out_head.weight,  params["wte"])
```

### 逐行拆（抓三个要点，别被长度吓到）

这个函数虽长，但**重复度极高**——`for b` 循环里把每一层的同一批权重搬一遍。盯住三个关键点：

1. **`np.split(... c_attn ..., 3, axis=-1)`**：OpenAI 把 Q、K、V 三个投影矩阵**合并存成一个大矩阵** `c_attn`，而我们的 `GPTModel` 是三个独立的 `W_query/W_key/W_value`。所以要先把大矩阵**沿最后一维切成三等份**，再分别灌进去。
2. **到处的 `.T`（转置）**：OpenAI 用 TensorFlow，权重存的是"我们的转置"。所以搬 `weight` 时几乎都要 `.T` 翻一下方向（这和第 3 章练习 3.1 里 "`nn.Linear` 以转置形式存权重" 是同一个坑）。`assign` 的形状检查就是用来抓"该转没转"的。
3. **`gpt.out_head.weight = params["wte"]`**：输出头直接**复用 token 嵌入**的权重——这叫**权重绑定（weight tying）**，省一大笔参数，GPT-2 就这么干。

最后一步，搬完就能生成：

```python
load_weights_into_gpt(gpt, params)
gpt.to(device)

torch.manual_seed(123)
token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25, context_size=NEW_CONFIG["context_length"],
    top_k=50, temperature=1.5)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
# Every effort moves you toward finding an ideal new way to practice
# something! What makes us want to be on top of that?
```

**生成的文本通顺、连贯、有逻辑**——这就是几十万美元算力的成果，被你一行 `load_weights_into_gpt` 白嫖到了。能生成连贯文本，本身就证明权重**搬对了**（搬错哪怕一个，输出就会崩成乱码，没有中间状态）。

🔧 **亲手改一下**：把 `model_size="124M"` 改成 `"355M"`（中号 GPT-2），同时把 `model_name` 换成 `"gpt2-medium (355M)"` 重跑整个加载流程。预测：生成文本会更连贯吗？（通常会——参数更多、见过的更多。这就是练习 5.6，也是规模法则的亲手验证。注意：模型更大，下载和运行更慢。）

---

## 8. 练习

> 📓 **对照官方答案**：卡住了别硬磕——[在 Colab 打开本章练习解答 notebook ↗](https://colab.research.google.com/github/rasbt/LLMs-from-scratch/blob/main/ch05/01_main-chapter-code/exercise-solutions.ipynb)

### 书上原题

**练习 5.1：`pizza` 被采样的频率。**
用 `print_sampled_tokens` 在不同温度下统计 "pizza" 被抽中的次数（对应图 5.14）。再想：有没有**更快更准**的办法直接算出这个频率，而不用抽 1000 次？
<details><summary>解题思路（点开）</summary>

抽样统计是"近似"。**精确**做法：直接看 `pizza` 在该温度下 softmax 后的概率值——那个概率就是它被采样的理论频率，乘以采样次数即得期望次数，不用真抽。即 `softmax_with_temperature(next_token_logits, T)[vocab["pizza"]]`。**收获**：理解"采样频率"本质就是"那个词的概率"，1000 次抽样只是在用蒙特卡洛近似一个你本可以直接读出的数。
</details>

**练习 5.2：调温度和 top-k，各适合什么场景？**
试不同 `temperature` 和 `top_k` 组合。想几个"低温 + 小 top-k 更合适"的场景，再想几个"高温 + 大 top-k 更合适"的。
<details><summary>解题思路（点开）</summary>

**低温 + 小 k（保守）**：要求准确、稳定、可复现的任务——代码生成、事实问答、数学、提取信息。你要的是"最可能正确的答案"，不要惊喜。
**高温 + 大 k（放飞）**：要创意和多样性的任务——写诗、头脑风暴、起名、编故事。重复和死板才是缺点，偶尔的"意外"反而是亮点。**收获**：解码参数不是玄学，是按任务在"稳"和"野"之间调旋钮。
</details>

**练习 5.3：怎么让 `generate` 退回确定性输出？**
什么参数组合能关掉随机性，让 `generate` 每次产出和贪婪解码一样的固定结果？
<details><summary>解题思路（点开）</summary>

设 `temperature=0.0`。看 `generate` 源码：`temperature > 0.0` 才走 `multinomial` 采样，否则走 `else` 分支的 `argmax`——这就是贪婪解码、完全确定。（`top_k` 设不设都行，argmax 本就只取第一名。）**收获**：看懂了那个 `if temperature > 0.0` 分支是"随机 / 确定"的总开关。
</details>

**练习 5.4：续训。**
保存权重后，在**新的 Python 会话**里加载模型和优化器，用 `train_model_simple` 再续训 1 个 epoch。
<details><summary>解题思路（点开）</summary>

用 §6 那段"连优化器一起存/取"的代码：`torch.save({...两个 state_dict...})` → 新会话里 `torch.load` → 分别 `load_state_dict` 灌回 model 和 optimizer → `model.train()` → 调 `train_model_simple(..., num_epochs=1, ...)`。**关键**：必须连优化器状态一起恢复，否则 AdamW 的动量丢失，续训效果会变差。**收获**：打通了"断点续训"这个实战刚需。
</details>

**练习 5.5 / 5.6（加载权重后）**：用 OpenAI 权重的模型，算它在《The Verdict》上的训练/验证损失（应该比我们自己训的低很多）；再换更大的 GPT-2（如 1558M）对比生成质量。

### 本册自测题（改了会怎样）

1. **关掉自监督的错位**：把 §2 的 `targets` 改成和 `inputs` **完全一样**（不错位）。重算 loss。想一想：此时模型在学"预测下一个词"还是"原样复制当前词"？这还能叫语言模型吗？
2. **学习率调极端**：把 `AdamW` 的 `lr=0.0004` 改成 `lr=0.1`（大 250 倍），重训几步。预测损失会平稳下降、剧烈震荡、还是直接发散（变 `nan`）？跑出来对上没？（体会"滚下山步子太大会冲出山谷"。）
3. **温度趋近 0**：在 `softmax_with_temperature` 里把温度设成 `0.01`，和 `argmax` 的结果比。它俩会越来越像吗？为什么温度越低越接近贪婪解码？（联系 §5"温度低＝分布变尖"。）

---

## 9. 本章小结

1. **预训练 = 反复"猜下一个词"**：把文本错位一格当答案（**自监督**，答案自带，不用人标），猜错就用**交叉熵损失**罚一下，据此调参数。
2. **交叉熵损失**就是第 0 章母版里那个 `cross_entropy`：手动六步（softmax→挑答案→取对数→平均→取负）= PyTorch 一行，记得**喂原始 logits**。损失高＝模型烂，目标是把它一路压向 0。
3. **训练循环 `train_model_simple` 和第 0 章母版结构一字不差**：核心永远是五步（前向→算损失→清梯度→反向→更新），外面只是套了 epoch 循环、评估打印、AdamW 优化器。损失"滚下山"你能在终端亲眼看见。
4. **解码策略**让模型别当复读机：**温度**调"稳 vs 野"，**top-k** 只在靠谱的前几名里采样（填 -inf 再 softmax，和第 3 章因果掩码同款巧思），`generate` 把两者缝进生成循环。
5. **存取权重**用 `state_dict`；要续训就连**优化器状态**一起存。
6. **站在巨人肩上**：自己预训练要烧几十万美元，所以直接 `load_weights_into_gpt` 加载 **OpenAI 的 GPT-2 权重**——架构相同，逐层对齐命名（注意 `c_attn` 要切三份、几乎处处要 `.T` 转置）即可白嫖。规模法则：更大模型只是把同样的积木堆更多、做更宽。

---

## 10. 能改自检清单（全勾＝过关）

- [ ] 我能解释"自监督"为什么不需要人工标注——答案从哪来。
- [ ] 我能说出交叉熵的"手动六步"，并知道为什么 `cross_entropy` 要喂 logits 而不是概率。
- [ ] 给我 `train_model_simple`，我能指出哪五行就是第 0 章的母版五步。
- [ ] 我能解释训练损失降、验证损失不降（过拟合）是怎么回事，以及小数据集为什么必然过拟合。
- [ ] 我能说清温度和 top-k 各调什么、`temperature=0` 时 `generate` 为什么退化成贪婪解码。
- [ ] 我能讲明白为什么加载 OpenAI 权重时要开 `qkv_bias`、要 `np.split` 切 c_attn、要 `.T` 转置。
- [ ] 我能独立完成练习 5.3、5.4，并预测每个自测题改动的结果再验证。

---

## 11. 通往下一章

你现在手里有一个**会说人话的 GPT-2**了——但它只会"接着往下写"，你问它问题，它不会"回答"，只会续写。**第 6 章**就解决这个：用一小批带标签的数据**微调（fine-tuning）**这个大脑，让它学会做具体任务（先是判断垃圾邮件的分类微调）。你这一章加载的预训练权重，正是第 6 章微调的**起点**。

> 带走主线：预训练让模型把"猜下一个词"练到炉火纯青——**这身本事，就是后面所有微调、对齐、Agent 的地基。一切的底层，仍然是预测下一个词。**
