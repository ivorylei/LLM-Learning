# 深化册 · 第 6 章　微调分类器：把"续写大脑"改造成"垃圾邮件判官"

> **对应**：概念课 **第 6 天** ｜ 书 **第 6 章（微调分类器）**
> **一句话直觉**：同一个预训练大脑，把"预测下一个词"的**输出头**换成"是/否两类"的头，再用少量标注数据训练几步，就成了垃圾邮件分类器——这就是**分类微调（classification fine-tuning）**。
> **这一章你将亲手得到**：一个**能判断一条短信是不是垃圾邮件**的 GPT 分类器，测试集准确率约 **96%**——而你只训练了它一小部分参数、花了几分钟。
>
> **前置**（卡住就翻回去）：
> - 训练循环五步母版（前向→算损失→清梯度→反向→更新）、`cross_entropy`、准确率怎么算 → [第 0 章 §5](ch00_python_pytorch.md)
> - `GPTModel` 的结构（嵌入层 + N 个 TransformerBlock + final_norm + out_head）、因果掩码 → [第 4 章](ch04_gpt.md)、[第 3 章](ch03_attention.md)
> - 加载 OpenAI 的 GPT-2 预训练权重（`download_and_load_gpt2` / `load_weights_into_gpt`）→ [第 5 章](ch05_pretrain.md)
> - 本章新出现的小工具（`requires_grad` / `torch.no_grad` / Dataset 与 DataLoader）→ 见各节「🐍 加油站」

---

[![在 Colab 中打开本章](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rasbt/LLMs-from-scratch/blob/main/ch06/01_main-chapter-code/ch06.ipynb)

> 配套可运行 notebook（rasbt/LLMs-from-scratch）。**边读边跑、跟着每节的「🔧 亲手改一下」改一改。**

## 0. 三分钟回顾（概念课直觉）

前五章你做完了一件大事：**造出并训练好一个会说人话的 GPT-2**。它的本事只有一个——**预测下一个词**，然后一个接一个地往下写。

但概念课第 6 天讲了一个关键转折：**这个预训练好的大脑，不只是会"续写"。它在亿万次猜词里，已经顺带学会了"读懂语言"——什么是句子、什么是语气、什么词常和什么词一起出现。**这身"语言理解力"是通用的，可以拿去干各种活。

这一章干的，就是**复用这身理解力，但换一种产出形式**：

> 不再让它"接着往下写"，而是让它对整段文本下一个判断——**这是垃圾邮件，还是正常短信？**

做法出奇地简单，可以浓缩成一句话：

> **同一个大脑，把最后那层"预测下一个词（5 万多类）"的输出头，换成"两类（是/否）"的小头，冻住大部分参数、只训练最后一小撮，喂几百条标好的短信，几分钟就练成了。**

这正是概念课第 6 天的主线落点：**预训练是地基，微调是低成本地把地基改造成一栋具体的房子**。指令微调（让它变成 ChatGPT 那样会聊天）是下一章的事；这一章先做更简单、更省的那种——**分类微调**。

本章的爬坡阶梯（和书 6.1→6.8 一致）：

```
准备 spam 数据集 (6.2)
   → 打包成 DataLoader (6.3)
   → 加载预训练 GPT-2 权重 (6.4)
   → 替换输出头 + 冻结大部分参数 (6.5)
   → 写"算准确率 / 算损失"的评估函数 (6.6)
   → 训练循环微调 (6.7)
   → 拿它分类新短信 (6.8)
```

---

## 1. 两种微调：分类 vs 指令（书 6.1）

> **这一节只讲思路，没有代码。** 但它决定了你后面写的每一行属于哪条路。

微调（fine-tuning）最常见的两种：

| | **分类微调**（本章）| **指令微调**（第 7 章/书后续）|
|---|---|---|
| 模型学会干什么 | 把输入归到**固定的几个类别**（如"垃圾/正常"）| 听懂并执行**各种自然语言指令**（"帮我翻译""总结这段"）|
| 产出形式 | 一个类别标签 | 一段自由文本 |
| 能力范围 | **窄而专**：只会它训练过的那几类 | **宽而通**：能应付没见过的任务 |
| 要多少数据/算力 | **少**（几百条就够，几分钟）| **多**（大数据集、更大算力）|
| 典型场景 | 垃圾邮件检测、情感分析、肿瘤良恶性判别 | ChatGPT 那种通用助手 |

书里的一句话总结很到位：

> 分类微调出来的是**高度专业化**的模型——它能判断某条短信是不是垃圾邮件，但你问它别的，它**一句别的也答不上来**。开发一个专才，通常比开发一个通才容易得多。

我们这一章走的就是"专才"这条便宜路。**记住这条主线：不管哪种微调，起点都是同一个预训练大脑——区别只在你给它换什么输出头、喂什么数据。**

---

## 2. 准备数据集：把短信变成模型能吃的样子（书 6.2）

### 2.1 下载并加载数据（代码清单 6.1）

我们用一个公开的短信数据集：每条短信都标好了 `ham`（正常）或 `spam`（垃圾）。

```python
import urllib.request
import zipfile
import os
from pathlib import Path

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(
        url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download "
              "and extraction.")
        return
    with urllib.request.urlopen(url) as response:           # 下载
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())
    with zipfile.ZipFile(zip_path, "r") as zip_ref:         # 解压
        zip_ref.extractall(extracted_path)
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)           # 改名加 .tsv
    print(f"File downloaded and saved as {data_file_path}")

download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
```

**逐行拆**：这段就是常规的"下载 zip → 解压 → 把文件改个名"。值得记的只有一处工程习惯：开头 `if data_file_path.exists(): return`——**文件已经在了就别重复下载**。

下载完是一个 **TSV**（用制表符分隔的表格），用 pandas 读进来：

```python
import pandas as pd

df = pd.read_csv(
    data_file_path, sep="\t", header=None, names=["Label", "Text"]
)
print(df)
```

`df` 是一张两列的表：`Label`（`ham`/`spam`）和 `Text`（短信内容）。

### 2.2 让两类一样多：欠采样（代码清单 6.2）

先看看两类各有多少：

```python
print(df["Label"].value_counts())
# Label
# ham     4825
# spam     747
```

正常短信（ham）有 4825 条，垃圾（spam）只有 747 条——**严重不平衡**。如果直接拿去训练，模型会偷懒：反正全猜"正常"也能对 87%，它就懒得真学。

书里用最简单的办法摆平：**欠采样（undersampling）**——把多的那类（ham）随机抽出 747 条，和 747 条 spam 凑成一份 1:1 的平衡数据。

```python
def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]      # spam 有多少条 = 747
    ham_subset = df[df["Label"] == "ham"].sample(
        num_spam, random_state=123                     # 从 ham 里随机抽同样多
    )
    balanced_df = pd.concat([
        ham_subset, df[df["Label"] == "spam"]          # 抽出的 ham + 全部 spam
    ])
    return balanced_df

balanced_df = create_balanced_dataset(df)
print(balanced_df["Label"].value_counts())
# Label
# ham     747
# spam    747
```

**逐行拆**：
- `df[df["Label"] == "spam"].shape[0]`：数出 spam 有多少条（747）。
- `.sample(num_spam, random_state=123)`：从 ham 里**随机抽** 747 条。`random_state=123` 固定随机种子，保证你我抽到的是同一批，结果可复现。
- `pd.concat([...])`：把抽出的 ham 和全部 spam **拼成一张表**。

接着把字符串标签转成整数——模型只认数字：

```python
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
```

> **这一步和第 2 章"文字变 TokenID"是同一个动作**：把人看的符号换成机器算的数字。区别只在于，这里的"词汇表"只有两个词——`0`（正常）和 `1`（垃圾）。

### 2.3 切成训练/验证/测试三份（代码清单 6.3）

机器学习的老规矩：数据要分三份——**训练**（拿来学）、**验证**（学的过程中盯着调）、**测试**（最后才看一眼，验真本事）。书里按 70% / 10% / 20% 切：

```python
def random_split(df, train_frac, validation_frac):
    df = df.sample(
        frac=1, random_state=123                    # frac=1 = 整体打乱
    ).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]
    return train_df, validation_df, test_df

train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

train_df.to_csv("train.csv", index=None)
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)
```

**逐行拆**：
- `df.sample(frac=1, ...)`：`frac=1` 表示"抽取 100%"——其实就是**整体洗牌**，把 ham 和 spam 打散混匀（不然前一半全是 ham、后一半全是 spam，切出来就废了）。
- `train_end` / `validation_end`：算出两个切割点。前 70% 给训练，接下来 10% 给验证，剩下 20% 给测试。
- 三个 `.to_csv(...)`：存成三个文件，后面随时复用，不用每次重切。

> **为什么必须留出测试集、训练时绝不看它？** 因为你要的是模型在**没见过的新短信**上的本事，而不是"把训练题背熟"。测试集就是模拟"上线后遇到的陌生短信"。第 5 章过拟合那一课讲的就是这件事。

🔧 **亲手改一下**：把 `random_split(balanced_df, 0.7, 0.1)` 改成 `0.8, 0.1`（训练 80%、验证 10%、测试 10%）。想一想：训练数据变多，最后测试准确率会更高还是更低？跑完整章后回来验证。

---

## 3. 打包成 DataLoader：长短不一的短信怎么对齐（书 6.2–6.3）

### 3.1 难点：每条短信长度都不一样

第 2 章里我们用滑动窗口，把长文切成**等长**的块，天然好打包成批。但短信不一样——有的三五个词，有的好几十个词。要把它们堆成一个整齐的张量批次（batch），必须先**对齐长度**。两个办法：

- **截断**：都砍到最短那条的长度——省算力，但会丢信息。
- **填充（padding）**：都补到最长那条的长度——保住全部内容。

书里选**填充**。用什么补？用 GPT-2 的特殊 token **`<|endoftext|>`**，它的 Token ID 是 **50256**：

```python
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
# [50256]
```

所以"对齐"就是：把每条短信编码成一串 Token ID，**不够长的，在后面补 50256，直到和最长那条一样长**。

### 3.2 `SpamDataset` 类（代码清单 6.4）

PyTorch 训练数据走"两层"：先写一个 **`Dataset`**（说清"第 index 条数据长啥样、怎么取"），再用 **`DataLoader`** 把它**自动分批、打乱**。先写 `Dataset`：

```python
import torch
from torch.utils.data import Dataset

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None,
                 pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        # ① 把每条短信编码成 Token ID 列表
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]
        # ② 决定统一长度
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # 超过上限的截断
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]
        # ③ 不够长的，用 pad_token_id 补到 max_length
        self.encoded_texts = [
            encoded_text + [pad_token_id] *
            (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),       # 输入：一串 Token ID
            torch.tensor(label, dtype=torch.long)          # 标签：0 或 1
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length
```

**逐行拆**（盯住三件事：编码、定长、补齐）：
- **`__init__` ①**：`tokenizer.encode(text)` 把每条短信变成 Token ID 列表（和第 2 章一样的分词器）。
- **`__init__` ②**：`max_length=None` 时，调 `_longest_encoded_length()` 找出"最长那条有多少 token"，以它为统一长度；否则用你指定的 `max_length`，并把超长的截断到这个长度。
- **`__init__` ③**：`encoded_text + [pad_token_id] * (差多少)`——这是最关键的一行，给短的那些**在尾部补 50256**，让每条都正好 `max_length` 个 token。
- **`__getitem__(self, index)`**：`DataLoader` 取第 `index` 条时会调它，返回 `(输入张量, 标签张量)` 一对。注意这里返回的标签不再是"下一个词"，**而是整条短信的类别 0/1**——这正是分类和预训练在数据上的根本区别。
- **`__len__`**：告诉 `DataLoader` 一共多少条。

`__getitem__` / `__len__` 是 PyTorch `Dataset` 的**两个约定方法**，名字不能改——`DataLoader` 全靠这两个名字来取数据和数总数。

> ### 🐍 加油站 ① — `Dataset` 与 `DataLoader` 是怎么配合的
> 把它俩想成"仓库"和"传送带"：
> - **`Dataset`（仓库）**：你只需告诉它两件事——`__len__`（一共多少件货）和 `__getitem__(index)`（第 index 件货长啥样）。它**不管批次、不管打乱**，只负责"按编号取一件"。
> - **`DataLoader`（传送带）**：把仓库包起来，自动帮你**攒成一批一批**、可选**每轮打乱顺序**、还能多进程并行取货。你训练时 `for batch in loader` 拿到的就是一摞已经堆好的张量。
> 分工清晰：**数据"长什么样"写在 Dataset，"怎么喂"交给 DataLoader。** 这套模式你以后写任何 PyTorch 训练都会反复用到。

实例化三个数据集——**注意验证集和测试集都用训练集的 `max_length` 来对齐**，保证三份数据的张量宽度一致：

```python
train_dataset = SpamDataset(
    csv_file="train.csv", max_length=None, tokenizer=tokenizer
)
print(train_dataset.max_length)   # 120：最长那条短信有 120 个 token

val_dataset = SpamDataset(
    csv_file="validation.csv",
    max_length=train_dataset.max_length,     # 跟训练集对齐
    tokenizer=tokenizer
)
test_dataset = SpamDataset(
    csv_file="test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)
```

`train_dataset.max_length` 是 **120**——说明最长的短信也就 120 个 token，远在 GPT-2 能吃的 1024 上限之内。

🔧 **亲手改一下**：这正是书里的**练习 6.1**——把 `max_length=None` 改成 `max_length=1024`（填充到模型支持的最大长度），重训一遍，看准确率和训练时间怎么变。预测：每条都补到 1024 个 token，算力会大涨，但准确率不见得更好（大量内容都是没意义的 padding）。

### 3.3 创建数据加载器（代码清单 6.5）

```python
from torch.utils.data import DataLoader

num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,                 # 训练集每轮打乱
    num_workers=num_workers,
    drop_last=True,               # 丢掉凑不满一批的尾巴
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)
```

**逐行拆**：
- `batch_size=8`：每批 8 条短信一起算，比一条条算高效得多。
- `shuffle=True`（只在训练集开）：每轮把顺序打乱，防止模型记住"题目的排列"。验证/测试不需要打乱。
- `drop_last=True`（只在训练集开）：如果最后剩下的不够 8 条，**丢掉**这半批——避免最后一个批次大小不齐影响训练稳定。验证/测试要看全部数据，所以 `False`。

验证一下批次形状对不对：

```python
for input_batch, target_batch in train_loader:
    pass
print("Input batch dimensions:", input_batch.shape)
print("Label batch dimensions", target_batch.shape)
# Input batch dimensions: torch.Size([8, 120])
# Label batch dimensions  torch.Size([8])
```

读这两个形状：输入是 `[8, 120]`——**8 条短信、每条 120 个 token**；标签是 `[8]`——**8 个类别标签（每条一个 0/1）**。再数数总共多少批：

```python
print(f"{len(train_loader)} training batches")     # 130 training batches
print(f"{len(val_loader)} validation batches")     # 19 validation batches
print(f"{len(test_loader)} test batches")          # 38 test batches
```

数据这一关通了。下面把预训练大脑请进来。

---

## 4. 把预训练的 GPT-2 请进来（书 6.4）

微调的起点，**永远是一个已经预训练好的模型**——我们不从零训。这里直接复用第 5 章的成果：把 OpenAI 的 GPT-2（124M）权重加载进我们自己的 `GPTModel`。

### 4.1 配置 + 加载权重（代码清单 6.6）

```python
CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True            # 加载 GPT-2 权重必须开，见第 5 章
}
model_configs = {
    "gpt2-small (124M)":  {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)":    {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
```

`BASE_CONFIG` 就是第 4/5 章那套 GPT 配置；`update(...)` 把选中型号的尺寸（嵌入维 768、12 层、12 个头）填进去。

```python
from gpt_download import download_and_load_gpt2
from chapter05 import GPTModel, load_weights_into_gpt

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")   # "124M"
settings, params = download_and_load_gpt2(
    model_size=model_size, models_dir="gpt2"
)

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()
```

> **`GPTModel`、`download_and_load_gpt2`、`load_weights_into_gpt` 都是前几章的代码，这里只是 `import` 进来直接用，不重新定义。** `GPTModel` 来自第 4 章（结构），`load_weights_into_gpt` 来自第 5 章（把 OpenAI 权重一块块搬进我们的模型，含 `np.split` 切 `c_attn`、`.T` 转置那些细节）。忘了的话翻回[第 5 章](ch05_pretrain.md)。

`model.eval()` 把模型切到**评估模式**（关掉 dropout 等只在训练时启用的随机行为）。

### 4.2 先验一下：它现在还只会"续写"

加载对不对，跑一段生成看看（复用第 4/5 章的生成函数）：

```python
from chapter04 import generate_text_simple
from chapter05 import text_to_token_ids, token_ids_to_text

text_1 = "Every effort moves you"
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_1, tokenizer),
    max_new_tokens=15,
    context_size=BASE_CONFIG["context_length"]
)
print(token_ids_to_text(token_ids, tokenizer))
# Every effort moves you forward. The first step is to understand the importance of your work
```

输出通顺——权重加载正确。但如果你**直接拿提示词让它分类垃圾邮件**呢？

```python
text_2 = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.'"
)
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_2, tokenizer),
    max_new_tokens=23,
    context_size=BASE_CONFIG["context_length"]
)
print(token_ids_to_text(token_ids, tokenizer))
# ...它只是把你的问题又重复了一遍，根本没回答 yes/no
```

模型**听不懂"请回答 yes/no"这种指令**——它只会"接着往下写"。这完全在意料之中：它只预训练过，没做过指令微调。所以我们不走"提示"这条路，而是动手**改造它的输出头**，让它天生就输出"两类"。

---

## 5. 替换输出头：把"预测下一个词"改成"判两类"（书 6.5）

这是全章的**心脏**。一句话：**砍掉那个对着 5 万多个词的大输出头，换上一个只对着 2 个类别的小输出头。**

### 5.1 先看清楚要动哪里

`print(model)` 看模型结构，重点看最后一行：

```
GPTModel(
  (tok_emb): Embedding(50257, 768)
  (pos_emb): Embedding(1024, 768)
  (drop_emb): Dropout(p=0.0, inplace=False)
  (trf_blocks): Sequential(
    ...
    (11): TransformerBlock( ... )      # 12 个块，只显示最后一个
  )
  (final_norm): LayerNorm()
  (out_head): Linear(in_features=768, out_features=50257, bias=False)   ← 就是它
)
```

`out_head` 是一个 `Linear(768 → 50257)`：把每个位置的 768 维隐藏向量，映射成 **50257 个分数**（词汇表里每个词一个分数，用来选下一个词）。我们要把它换成 `Linear(768 → 2)`。

### 5.2 先冻结全部，再只解冻该训的部分

我们从一个**已经很会读语言**的模型出发，没必要把它整个重训一遍——那既慢又容易把学好的本事搞坏。策略是：

> **先把所有参数冻住（不训练），再只解冻"靠近输出"的那几层。**

底层网络学的是通用的语言结构（什么是词、什么是句法），各种任务都用得上，不用动；越靠近输出的层越"任务专属"，这些才需要针对垃圾邮件微调。

第一步——**全部冻结**：

```python
for param in model.parameters():
    param.requires_grad = False
```

> ### 🐍 加油站 ② — `requires_grad = False` 是什么意思（冻结参数）
> 每个参数张量都有个开关 `requires_grad`：
> - **`True`（默认）**：训练时 PyTorch 会给它算梯度、`optimizer.step()` 会更新它——这个参数**会被学**。
> - **`False`**：不算梯度、不更新——这个参数**被冻住了**，训练时纹丝不动。
> 把整个模型 `requires_grad = False`，等于告诉 PyTorch："这些本事先保留原样，别动。"然后我们再单独打开几处需要学的地方。好处：**要训练的参数少了，训练又快又省、还不容易破坏预训练学到的东西。**

### 5.3 换上新的两类输出头（代码清单 6.7）

```python
torch.manual_seed(123)
num_classes = 2
model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"],     # 768
    out_features=num_classes                # 2
)
```

**逐行拆**：
- 直接给 `model.out_head` **重新赋一个新的 `nn.Linear`**——旧的 `768→50257` 就被这个 `768→2` 顶替了。
- `in_features` 用 `BASE_CONFIG["emb_dim"]`（不写死 768），这样换更大的 GPT-2 型号时这行代码不用改。
- **关键细节**：新建的 `nn.Linear` 的 `requires_grad` **默认就是 `True`**。所以即使前面把全模型冻死了，这个**新头是活的、会被训练**——这正是我们要的。

> ### 🐍 加油站 ③ — 输出头形状从 `[.., 50257]` 变成 `[.., 2]`
> 输出头本质是一个矩阵乘 `隐藏向量 @ W`。`W` 的形状决定了输出有几个分数：
> - **预训练时** `W: [768, 50257]` → 每个位置吐 **50257 个分数**，含义是"下一个词是词表里第几个词"，配 softmax 就是一个超大词表上的概率分布。
> - **分类时** `W: [768, 2]` → 每个位置只吐 **2 个分数**，含义是"这是第 0 类（正常）还是第 1 类（垃圾）"。
> 注意：**前面那一整套（嵌入 + 12 层 Transformer）完全没变**——它照样把短信"读"成一串 768 维的隐藏向量。**变的只有最后这一下"读完之后怎么表态"。** 这就是"同一个大脑、换个出口干不同的活"最具体的体现。

光训新头其实就能用，但书里发现：**多解冻一点点，效果显著更好**。所以再把**最后一个 Transformer 块**和**最终的 LayerNorm** 也解冻：

```python
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True
```

`model.trf_blocks[-1]` 是第 12 个（最后一个）Transformer 块，`model.final_norm` 是输出前的最后一道归一化。**最终会被训练的，就只有：新输出头 + 最后一个块 + 最后的 LayerNorm**——占全模型参数的一小撮。

### 5.4 看看新模型现在吐什么

随便喂一句进去：

```python
inputs = tokenizer.encode("Do you have time")
inputs = torch.tensor(inputs).unsqueeze(0)        # 加一个 batch 维 -> [1, 4]
print("Inputs:", inputs)                          # tensor([[5211, 345, 423, 640]])
print("Inputs dimensions:", inputs.shape)         # torch.Size([1, 4])

with torch.no_grad():
    outputs = model(inputs)
print("Outputs:\n", outputs)
print("Outputs dimensions:", outputs.shape)
# Outputs dimensions: torch.Size([1, 4, 2])
```

读输出形状 `[1, 4, 2]`：**1 句、4 个 token、每个 token 2 个分数**。对比一下——换头之前同样的输入会输出 `[1, 4, 50257]`。**行数（4）没变（还是每个输入 token 一行），变的是每行的宽度：从 50257 缩成了 2。** 换头成功。

> ### 🐍 加油站 ④ — `torch.no_grad()`：只看不学的时候用它
> `with torch.no_grad():` 包起来的代码，PyTorch **不会记录梯度信息**。两个好处：**省内存、跑得快**。
> 什么时候用：**任何"只想要模型输出、不打算反向传播"的场合**——比如这里的试跑、后面算准确率、最终分类新短信。
> 什么时候**别**用：真正训练那一步（要靠梯度更新参数），那里必须让它记录梯度。
> 一句话记法：**"要不要学"——学就别包，不学就包上 `no_grad`。**

### 5.5 关键抉择：只用**最后一个** token 的输出

输出有 4 行（4 个 token 各一行），但分类只需要**一个**判断。书里明确：**只取最后一个 token 那一行**。

```python
print("Last output token:", outputs[:, -1, :])
# Last output token: tensor([[-3.5983, 3.9902]])
```

`outputs[:, -1, :]` 的意思是"所有 batch、**最后一个位置** `-1`、那 2 个分数全要"。这就把 `[1, 4, 2]` 取成了 `[1, 2]`——**一句话一个判断、含 2 个类别分数**。

> ### 🐍 加油站 ⑤ — 为什么偏偏用**最后一个** token 来分类？
> 这要回到第 3 章的**因果掩码**：GPT 里每个 token 只能"看见自己和它前面的"，看不见后面的。
> 那么在一条短信里，**哪个 token 看到的信息最全？**——**最后一个**！只有它能"回望"整条短信的所有词。前面的 token 都只看到了部分内容。
> 所以最后一个 token 的隐藏向量，是**整条短信信息的最完整汇总**。拿它去判断"垃圾还是正常"，自然最靠谱。这就是为什么我们盯着 `outputs[:, -1, :]`，而不是第一个 token 或所有 token 的平均。
> （书里还留了**练习 6.3**：改成用第一个 token 试试，看准确率掉多少——你会直观看到"信息不全"的代价。）

🔧 **亲手改一下**：把 `outputs[:, -1, :]` 改成 `outputs[:, 0, :]`（取第一个 token），打印看看。它对应的是模型只看到 `"Do"` 这一个词时的判断——信息少得可怜。记住这个差别，5.5 加油站讲的就是它。

---

## 6. 怎么算"准确率"和"损失"（书 6.6）

要训练，先得有"尺子"：一把量**准确率**（多少条判对了，给人看），一把量**损失**（给优化器滚下山用）。

### 6.1 从 2 个分数到一个类别标签

和预训练里"50257 个分数选下一个词"一模一样的套路，只是这回只有 2 个分数：

```python
logits = outputs[:, -1, :]            # 最后一个 token 的 2 个分数
label = torch.argmax(logits)          # 哪个分数大就选哪类
print("Class label:", label.item())   # 1  -> 预测为 spam
```

`torch.argmax` 返回"最大值在第几个位置"。位置 0 = 正常，位置 1 = 垃圾。**这里不用先 softmax**——因为最大的分数对应最大的概率，谁大谁大的顺序不会因 softmax 改变，直接对 logits 取 argmax 就行（省一步）。

### 6.2 算整个数据集的准确率（代码清单 6.8）

把上面这招套到一整个 DataLoader 上，数出"判对的比例"：

```python
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]     # 只取最后 token
            predicted_labels = torch.argmax(logits, dim=-1)
            num_examples += predicted_labels.shape[0]
            correct_predictions += (
                (predicted_labels == target_batch).sum().item()
            )
        else:
            break
    return correct_predictions / num_examples
```

**逐行拆**：
- `model.eval()` + `with torch.no_grad()`：评估不学习，关随机、不记梯度。
- `model(input_batch)[:, -1, :]`：跑模型，**只取最后一个 token** 的 2 个分数（注意这个 `[:, -1, :]` 在本章会反复出现，它是分类和预训练的分水岭）。
- `torch.argmax(logits, dim=-1)`：每条短信选出预测类别。`dim=-1` 表示沿"2 个分数"那一维比大小。
- `(predicted_labels == target_batch).sum().item()`：预测和真实标签**逐个比对**，`==` 得到一串 True/False，`.sum()` 数出几个 True（即判对几条）。
- `num_batches` 参数：允许只看前几批（训练时为了快，常只抽 10 批估一下）。

> **`device` 是什么？** `"cuda"`（有 NVIDIA GPU 时）或 `"cpu"`。`.to(device)` 把数据搬到和模型相同的设备上——否则会报"一个在 CPU 一个在 GPU"的错。第 0 章 / 第 5 章讲过这个套路。

微调**之前**先量一下（抽 10 批）：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
torch.manual_seed(123)

train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
val_accuracy   = calc_accuracy_loader(val_loader,   model, device, num_batches=10)
test_accuracy  = calc_accuracy_loader(test_loader,  model, device, num_batches=10)
print(f"Training accuracy:   {train_accuracy*100:.2f}%")   # ~46%
print(f"Validation accuracy: {val_accuracy*100:.2f}%")     # ~45%
print(f"Test accuracy:       {test_accuracy*100:.2f}%")    # ~49%
```

约 46% / 45% / 49%——**接近瞎猜**（两类各 50%）。完全正常：新输出头才刚随机初始化，啥也没学。下面就靠训练把它拉上去。

### 6.3 分类损失：还是交叉熵（calc_loss_batch / 代码清单 6.9）

准确率是给人看的，但它**不可微**（不能对它求梯度让模型滚下山）。所以训练时用**交叉熵损失**当代理——和第 5 章预训练**用的是同一个损失函数**，唯一改动就一处：

```python
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]          # ← 唯一的改动：只取最后 token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss
```

**逐行拆**：和第 5 章的 `calc_loss_batch` 几乎一字不差，**唯一区别是这个 `[:, -1, :]`**——预训练对**所有** token 算损失（每个位置都要预测下一个词），分类只对**最后一个** token 算损失（整句一个判断）。`cross_entropy` 吃的是 `logits`（原始分数，**不是** softmax 后的概率），它内部会自己做 softmax，这点第 5 章加油站讲过。

整套 DataLoader 上的平均损失（代码清单 6.9，结构和第 5 章完全相同）：

```python
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

微调前量一下初始损失（抽 5 批）：

```python
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss   = calc_loss_loader(val_loader,   model, device, num_batches=5)
    test_loss  = calc_loss_loader(test_loader,  model, device, num_batches=5)
print(f"Training loss:   {train_loss:.3f}")    # 2.453
print(f"Validation loss: {val_loss:.3f}")      # 2.583
print(f"Test loss:       {test_loss:.3f}")     # 2.322
```

损失 2.x，挺高——和"准确率约等于瞎猜"对得上。该训练了。

---

## 7. 微调：训练循环跑起来（书 6.7）

### 7.1 训练函数（代码清单 6.10）

这个 `train_classifier_simple` 和第 5 章预训练的 `train_model_simple` **几乎一模一样**——同一套训练循环母版。只有两处不同：① 现在数"见过多少**样本**"（`examples_seen`）而不是"多少 token"；② 每个 epoch 后**算准确率**，而不是生成一段示例文本。

```python
def train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()                                 # 切回训练模式
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()                     # ① 清空旧梯度
            loss = calc_loss_batch(                   # ② 前向 + 算损失
                input_batch, target_batch, model, device
            )
            loss.backward()                           # ③ 反向传播
            optimizer.step()                          # ④ 更新参数
            examples_seen += input_batch.shape[0]     # 累计见过多少条
            global_step += 1

            if global_step % eval_freq == 0:          # 每隔几步评估一次
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}")

        # 每个 epoch 末尾，算一次准确率
        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen
```

**逐行拆**——盯住 ①②③④ 这四步，**它就是第 0 章那条训练循环母版，原封不动**：
- **① `optimizer.zero_grad()`**：清掉上一轮的梯度（梯度默认会累加，不清就乱了）。
- **② `calc_loss_batch(...)`**：前向跑模型、算这一批的交叉熵损失。
- **③ `loss.backward()`**：反向传播，算出"每个**可训练**参数该往哪挪"。注意——被我们冻住的参数 `requires_grad=False`，这一步**自动跳过**它们。
- **④ `optimizer.step()`**：照梯度更新参数（只更新没冻的那些）。
- 其余都是"记账和汇报"：每 `eval_freq` 步打印一次损失，每个 epoch 末算一次准确率。

`evaluate_model` 也和第 5 章完全相同——临时切 `eval()`、不记梯度地量训练/验证损失，再切回 `train()`：

```python
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss
```

### 7.2 开练

```python
import time
start_time = time.time()
torch.manual_seed(123)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 5

train_losses, val_losses, train_accs, val_accs, examples_seen = \
    train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50, eval_iter=5
    )

end_time = time.time()
print(f"Training completed in {(end_time - start_time)/60:.2f} minutes.")
```

**逐行拆**：
- `AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)`：优化器。注意虽然把**全部** `model.parameters()` 交给了它，但被冻住的参数没有梯度，`step()` 时根本不会动——所以**实际只更新解冻的那一小撮**。`lr=5e-5` 是个温和的小学习率（微调要轻手轻脚，别把预训练学的本事冲掉）。
- `num_epochs=5`：整个训练集过 5 遍。书里说 5 轮通常是个不错的起点。
- `eval_freq=50, eval_iter=5`：每 50 步打印一次损失；每次评估只抽 5 批估算（图快）。

训练过程（书里的真实输出）：

```
Ep 1 (Step 000000): Train loss 2.153, Val loss 2.392
Ep 1 (Step 000050): Train loss 0.617, Val loss 0.637
Ep 1 (Step 000100): Train loss 0.523, Val loss 0.557
Training accuracy: 70.00% | Validation accuracy: 72.50%
...
Ep 4 (Step 000500): Train loss 0.222, Val loss 0.137
Training accuracy: 100.00% | Validation accuracy: 97.50%
Ep 5 (Step 000600): Train loss 0.083, Val loss 0.074
Training accuracy: 100.00% | Validation accuracy: 97.50%
Training completed in 5.65 minutes.
```

**看这条曲线**：损失从 2.15 一路滚到 0.08，准确率从约 50%（瞎猜）冲到 97%+——**这就是第 5 天那张"损失滚下山"的图，只不过这回滚的是分类损失**。一台 M3 MacBook Air 约 6 分钟、A100 不到半分钟。**这就是"微调很便宜"的实感**：不用几十万美元算力，几分钟就把通用大脑改造成了专用判官。

### 7.3 画曲线（代码清单 6.11，只讲思路）

> 书里给了一个 `plot_values` 函数，用 Matplotlib 把 `train_losses`/`val_losses`（以及准确率）画成随 epoch 变化的曲线。**它纯粹是画图、不参与训练**，这里不逐行拆。关键是看图判断：训练损失和验证损失都一路向下、且**没明显拉开差距**，说明学得好、几乎没过拟合。

### 7.4 在**整个**测试集上验真本事

训练时为了快，评估只抽了 5 批。最后摘掉 `num_batches`，在**全部**数据上量一遍：

```python
train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy   = calc_accuracy_loader(val_loader,   model, device)
test_accuracy  = calc_accuracy_loader(test_loader,  model, device)
print(f"Training accuracy:   {train_accuracy*100:.2f}%")   # 97.21%
print(f"Validation accuracy: {val_accuracy*100:.2f}%")     # 97.32%
print(f"Test accuracy:       {test_accuracy*100:.2f}%")    # 95.67%
```

**测试集 95.67%**——这才是真本事（模型从没见过这些短信）。训练 97% vs 测试 96%，差距很小，说明几乎没过拟合。漂亮。

🔧 **亲手改一下**：这接近书里的**练习 6.2**——把 5.2 节里那两段"解冻最后一个块 / final_norm"删掉（变成**只训练新输出头**），重训。预测：准确率会掉一些（书里说多解冻几层效果明显更好）。或者反过来，**解冻整个模型**（去掉全部冻结那段），看准确率和耗时怎么变。

---

## 8. 拿它来分类新短信（书 6.8）

模型练好了，封装一个"输入一句话 → 输出 spam / not spam"的函数（代码清单 6.12）。它的预处理和 `SpamDataset` 一脉相承：编码 → 截断/填充 → 取最后 token → argmax。

```python
def classify_review(
        text, model, tokenizer, device, max_length=None,
        pad_token_id=50256):
    model.eval()

    input_ids = tokenizer.encode(text)                       # 编码
    supported_context_length = model.pos_emb.weight.shape[1]
    input_ids = input_ids[:min(                              # 截断到上限
        max_length, supported_context_length
    )]
    input_ids += [pad_token_id] * (max_length - len(input_ids))   # 填充

    input_tensor = torch.tensor(
        input_ids, device=device
    ).unsqueeze(0)                                           # 加 batch 维

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]              # 只取最后 token
    predicted_label = torch.argmax(logits, dim=-1).item()
    return "spam" if predicted_label == 1 else "not spam"
```

**逐行拆**：
- `tokenizer.encode(text)`：和训练时同一个分词器，把短信变 Token ID。
- 截断 + 填充：和 `SpamDataset` 里一样，保证长度合法（不超模型上限、补到 `max_length`）。
- `.unsqueeze(0)`：模型要的是带 batch 维的输入 `[1, 长度]`，所以给单句加一维。
- `model(...)[:, -1, :]` → `argmax`：**又是这套"取最后 token、选最大分数"** ——和算损失、算准确率时一模一样。
- 最后把 `1/0` 翻译回 `"spam"/"not spam"`。

试两条：

```python
text_1 = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)
print(classify_review(text_1, model, tokenizer, device,
                      max_length=train_dataset.max_length))
# spam

text_2 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)
print(classify_review(text_2, model, tokenizer, device,
                      max_length=train_dataset.max_length))
# not spam
```

中奖诈骗判成 `spam`、约饭日常判成 `not spam`——**都对**。你亲手把一个"续写大脑"改造成了"垃圾邮件判官"。

最后存盘，下次直接加载、不用重训：

```python
torch.save(model.state_dict(), "review_classifier.pth")

# 下次加载：
model_state_dict = torch.load("review_classifier.pth", map_location=device)
model.load_state_dict(model_state_dict)
```

> **`state_dict` 是什么？** 模型里所有参数的一本"字典账本"（每层的权重）。`torch.save` 把账本存成文件，`load_state_dict` 把账本灌回一个同结构的模型——结构代码 + 这本账本 = 完整复活的模型。第 0 章存取模型那节讲过。

---

## 9. 练习

> 📓 **对照官方答案**：卡住了别硬磕——[在 Colab 打开本章练习解答 notebook ↗](https://colab.research.google.com/github/rasbt/LLMs-from-scratch/blob/main/ch06/01_main-chapter-code/exercise-solutions.ipynb)

### 书上原题

**练习 6.1：增大上下文长度。**
把输入填充到模型支持的最大 token 数（`max_length=1024`），观察对预测性能的影响。
<details><summary>解题思路（点开）</summary>

在建 `SpamDataset` 时把 `max_length=None` 改成 `max_length=1024`（三个数据集都改），其余不变，重训。
**预期**：每条短信都补到 1024 个 token，绝大部分是无意义的 padding，**计算量大幅上升、训练变慢**，而准确率不见得提高（甚至可能略降）——因为有用信息就那么几十个 token，硬撑到 1024 只是在烧算力。**收获**：体会到"填充到多长"是个**性价比**权衡，不是越长越好。
</details>

**练习 6.2：微调整个模型。**
不只解冻最后一个 transformer 块，而是**解冻全部层**一起训，看性能变化。
<details><summary>解题思路（点开）</summary>

把"全部冻结 + 只解冻最后一块/final_norm"那几段去掉，直接让所有参数 `requires_grad=True` 再训。
**预期**：能训的参数变多，**训练更慢、更吃显存**，准确率通常会再高一点点，但提升有限——因为底层学的通用语言结构本就不太需要为"判垃圾邮件"而改。**收获**：理解"微调多少层"是**效果 vs 成本**的取舍，书里默认"只调最后几层"正是这个甜点位。
</details>

**练习 6.3：微调第一个 token vs 最后一个 token。**
把分类用的输出从最后一个 token 改成**第一个** token，看准确率变化。
<details><summary>解题思路（点开）</summary>

把代码里所有 `[:, -1, :]` 改成 `[:, 0, :]`（损失、准确率、分类函数都要改），重训。
**预期**：准确率明显下降。原因见加油站 ⑤——**因果掩码下，第一个 token 只看到了它自己一个词，信息量极少**；最后一个 token 才能"回望"整条短信。**收获**：把"为什么用最后一个 token"从背诵变成亲眼所见。
</details>

### 本册自测题（改了会怎样）

1. **不冻结会怎样**：把 5.2 节"全部冻结"那段整段删掉（不冻任何参数），但保留换头。训练耗时和测试准确率各怎么变？联系"微调多少层"的取舍。
2. **数据不平衡回来**：跳过 `create_balanced_dataset`，直接拿原始 4825:747 的数据训。准确率数字可能很高（比如 87%），但它真的"会判垃圾邮件"吗？想一想：如果模型对**所有**短信都猜"正常"，准确率是多少？（这就是为什么 6.2 要先平衡。）
3. **`[:, -1, :]` 漏改一处**：故意把 `calc_loss_batch` 里的 `[:, -1, :]` 删掉（变成对所有 token 算损失），但 `calc_accuracy_loader` 不动。会报错还是会训出个怪东西？为什么训练目标和评估口径必须一致？

---

## 10. 本章小结

1. **微调 = 站在预训练大脑的肩膀上。** 起点永远是一个已经会读语言的模型；分类微调只是**换个出口**：把"预测下一个词（50257 类）"的输出头，换成"判类别（2 类）"的小头。
2. **冻结大部分、只训一小撮。** `requires_grad=False` 冻住底层通用能力，只解冻"新输出头 + 最后一个 block + final_norm"——又快又省，还不破坏预训练学到的东西。
3. **盯住最后一个 token。** 因果掩码下，只有最后一个 token 看遍了全句，信息最全；所以损失、准确率、分类全都取 `outputs[:, -1, :]`。这是分类微调代码里反复出现的"暗号"。
4. **训练循环和预训练同一套母版。** `train_classifier_simple` 就是第 0 章那五步（清梯度→前向→算损失→反向→更新），唯一改动是损失只看最后 token、评估改算准确率。
5. **损失还是交叉熵。** 准确率不可微，用交叉熵当代理优化；`cross_entropy` 喂的是 logits 不是概率。
6. **微调很便宜，这正是第 6 天的主线**：几百条短信、几分钟、一小撮参数，就把通用大脑改造成 95%+ 准确率的专用判官——**同一个大脑能干各种活，区别只在换什么头、喂什么数据。**

---

## 11. 能改自检清单（全勾＝过关）

- [ ] 我能说清分类微调和指令微调的区别，以及为什么本章走"专才"这条便宜路。
- [ ] 我能解释为什么要先 `requires_grad=False` 冻全模型、再只解冻最后几层。
- [ ] 给我换头那行 `model.out_head = nn.Linear(768, 2)`，我能说出输出形状从 `[.., 50257]` 变成 `[.., 2]`、而前面整套网络没动。
- [ ] 我能不看书解释"为什么分类用最后一个 token 的输出"（联系因果掩码）。
- [ ] 我能指出 `calc_loss_batch` 和第 5 章相比唯一改的那一处（`[:, -1, :]`），并说清为什么。
- [ ] 我能在 `train_classifier_simple` 里圈出第 0 章训练循环的五步母版。
- [ ] 我能说明 `torch.no_grad()` 什么时候该用、什么时候不能用。
- [ ] 我能独立完成练习 6.1、6.2、6.3，并预测每个自测题改动的结果再验证。

---

## 12. 通往下一章

你现在有了一个**专才**——只会判垃圾邮件。但概念课第 6 天还有更激动人心的一步：**指令微调**——同样是这个预训练大脑，喂上"指令→回答"这种数据，它就学会**听懂人话、按要求干各种活**，这正是 ChatGPT 诞生的那一步。下一章就做这件事：从"判两类的专才"，走向"会聊天的通才"。

> 带走主线：这一章你看清了一件事——**预训练把"预测下一个词"练到炉火纯青之后，那身语言理解力可以被极低成本地改造成任何具体任务。换个输出头、喂点标注数据，专才就成了。而这一切的底座，始终是——预测下一个词。**
