# 深化册 · 第 7 章　指令微调：把"自动续写机"调教成"听话的助手"

> **对应**：概念课 **第 6 天（指令微调那部分）** ｜ 书 **第 7 章（指令微调）**
> **一句话直觉**：给模型成千上万条「指令 → 理想回答」样例，训练它**接到指令就给出合适回答**——把"自动续写机"变成"听话的问答助手"。**这就是 ChatGPT 诞生的那一步。**
> **这一章你将亲手得到**：一个**经指令微调、能听懂指令并回答**的 GPT-2 medium（355M）——你问它"把这句话改成被动语态"，它真的会照做，而不是傻乎乎地把你的话再抄一遍。
>
> **前置**（卡住就翻回去）：
> - 训练循环 `train_model_simple`、`calc_loss_loader`、`generate`、加载 OpenAI 权重 `load_weights_into_gpt` → [第 5 章](ch05_pretrain.md)
> - 微调的套路（造一个 `Dataset`、用 `DataLoader` 喂批次、复用训练循环）→ [第 6 章（分类微调）](ch06_classify.md)
> - 本章新出现的工具（padding 对齐 / `-100` 掩码 / `torch.stack` / `functools.partial` / REST API）→ 见各节「🐍 加油站」

---

[![在 Colab 中打开本章](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/ch07.ipynb)

> 配套可运行 notebook（rasbt/LLMs-from-scratch）。**边读边跑、跟着每节的「🔧 亲手改一下」改一改。**

## 0. 三分钟回顾（概念课直觉）

前面五章，你已经把一台 GPT **造出来、喂聪明、还白嫖了 OpenAI 的权重**——它现在是个很能"接着往下写"的**续写机**。但续写机有个尴尬毛病：**你给它一条指令，它不会"回答"，只会"接龙"。**

书里有个活生生的例子。把指令喂给微调前的 GPT-2 medium：

> **指令**：Convert the active sentence to passive: 'The chef cooks the meal every day.'（把这句主动句改成被动语态）
>
> **它的回答**：`The chef cooks the meal every day. ### Instruction: Convert the active sentence to passive: 'The chef cooks the...`

看见了吗？它**只是把你的话又抄了一遍**，还顺手续写了下一条"指令"的开头。它根本没"听懂"你要它做什么——因为预训练只教过它一件事：**猜下一个词**。它从没见过"指令→回答"这种对话格式。

这一章干的事，概念课第 6 天讲过，就一句话——

> **指令微调（instruction fine-tuning，也叫监督微调 SFT）= 给模型看成千上万条「指令 → 理想回答」的样例，让它学会"接到指令该怎么回"。**

这就是 **GPT-3 → ChatGPT 的那道分水岭**：底层那台"预测下一个词"的机器**几乎没变**，变的是**这层调教**——先指令微调（本章），再加一层人类偏好对齐（RLHF，书里作为可选下一步）。**底座是同一个，调教方式不同，就有了会聊天的助手。**

整章的流程（书里图 7.3 的三个阶段）：

```
阶段 1：准备数据   下载「指令-回答」数据集 → 套 Alpaca 模板 → 切成训练批次（本章最难的 custom_collate_fn 在这）
阶段 2：训练       加载 GPT-2 medium 预训练权重 → 复用第 5 章训练循环 → 损失滚下山
阶段 3：评估       让模型回答测试集 → 用另一个更大的 LLM（Llama 3）给回答打分
```

我们就按这条线走。**注意：本章几乎不写新模型代码**——`GPTModel`、`train_model_simple`、`generate` 全是第 4、5 章的老朋友，原样复用。本章真正的新东西，**全在"怎么把数据喂对"**。这也呼应了一条行业真相：**做 SFT，七成功夫在数据。**

---

## 1. 指令微调到底在调什么（书 7.1）

先把概念钉死，免得后面看代码犯迷糊。

- **预训练**教的是"**世界知识 + 语言能力**"——靠在半个互联网上猜下一个词，模型学会了语法、事实、推理的雏形。但它的输出形态是"**续写**"。
- **指令微调**不教新知识，它**重塑输出形态**：把"看到一段文字就续写"改成"看到一条指令就给出该指令对应的回答"。

打个比方：预训练像是让一个人读完了整个图书馆（满肚子学问），但他只会喃喃自语地接话；指令微调则是请他做了几千道"**别人提问、他作答**"的练习题，于是他学会了"**有人问，我就答**"这个**交互规矩**。**知识没变多，但变得听话、能用了。**

> 书里特别点明：指令微调用的预训练模型，**从 124M（small）换成了 355M（medium）**。原因——"124M 容量太小，学不动指令跟随这种复杂细腻的行为"。**这是个重要工程经验：指令微调对模型容量有下限要求，太小的模型怎么调都调不出像样的助手。**

数据集长什么样？书里专门为这本书做了一个 **1,100 条**的小数据集，每条是一个 Python 字典，三个字段：

```python
# data[50]
{'instruction': 'Identify the correct spelling of the following word.',
 'input': 'Ocassion',
 'output': "The correct spelling is 'Occasion.'"}

# data[999]  —— 注意 input 可以为空
{'instruction': "What is an antonym of 'complicated'?",
 'input': '',
 'output': "An antonym of 'complicated' is 'simple'."}
```

- **instruction（指令）**：要模型做的事。
- **input（输入）**：指令要操作的对象——**可有可无**（像"反义词是什么"这种就不需要额外输入）。
- **output（输出）**：我们希望模型给出的**标准答案**。这就是训练时的"**正确答案**"，模型猜得不像就挨罚。

**🐍 中英对照**：instruction＝指令，input＝（指令的）输入，output＝（理想）输出/回答。下文也叫 response（响应）。

> ### 🐍 加油站 ① — JSON 与 Python 字典：数据集为什么长这样
> 这个数据集存成 **JSON** 文件（`.json`）。JSON 是一种纯文本的数据格式，长得**几乎和 Python 字典/列表一模一样**：`{}` 是字典（键值对），`[]` 是列表。
> 用 `json.load(file)` 读进来，就直接变成一个 Python **列表**，里面 1,100 个**字典**，每个字典就是上面那种三字段结构。
> 为什么用 JSON？因为它**人能读、机器也好解析**，是数据交换的"普通话"。你以后下载任何指令数据集，八成都是这个格式。
> 下载这一步书里用 `download_and_load_file`（代码清单 7.1，本节只讲思路）：文件在本地就直接读，不在就用 `urllib.request` 从 GitHub 拉下来——纯粹是个"有就读、没有就下"的工具函数，不必细抠。

---

## 2. 套模板：把字典变成模型能读的一段提示词（书 7.2）

模型不吃 Python 字典，它只吃**一段连续的文字**。所以我们得把 `{instruction, input, output}` 这三个字段，**拼成一段格式固定的文字**——这套固定格式叫**提示词模板（prompt template / prompt style）**。

本章用的是 **Alpaca 风格**（最早公开指令微调细节的模型之一，也是最流行的模板）。代码清单 7.2 实现这个拼接：

```python
# 代码清单 7.2　实现提示词格式功能
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )

    return instruction_text + input_text
```

**逐行拆**：

- `instruction_text`：先是一句**固定的开场白**（"下面是一条描述任务的指令，请写出恰当完成请求的回答"），然后跟一个 `### Instruction:` 小标题，把这条样例的 `instruction` 字段填进去。
- `input_text`：这一行是**条件表达式**（行尾的 `if entry["input"] else ""`）。如果 `input` 字段**非空**，就拼一段 `### Input:\n{...}`；如果**为空**，就拼一个空字符串 `""`——**整段 `### Input:` 直接消失**。这正好对应"input 可有可无"。
- 最后把两段**字符串相加**（`+` 拼接）返回。注意：`format_input` 只拼到指令和输入为止，**不含答案**——答案 `output` 是另外拼的（见下文）。

拼出来长这样（以 `data[50]` 为例，再手动补上答案部分）：

```python
model_input = format_input(data[50])
desired_response = f"\n\n### Response:\n{data[50]['output']}"
print(model_input + desired_response)
```

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Identify the correct spelling of the following word.

### Input:
Ocassion

### Response:
The correct spelling is 'Occasion.'
```

而 `data[999]`（input 为空）拼出来就**没有 `### Input:` 那一段**，`### Instruction:` 后面直接接 `### Response:`。

> ### 🐍 加油站 ② — 为什么要"套模板"？`### Instruction:` 这种标记图什么
> 模型靠**模式**学东西。如果每条样例都用**同一套分隔标记**（`### Instruction:` / `### Input:` / `### Response:`）把"指令区""输入区""回答区"框得清清楚楚，模型很快就能学会一条铁律：
> **"看到 `### Response:`，接下来就该我写回答了。"**
> 这就是模板的全部魔力——它给了模型一个**稳定的信号**，告诉它"输入到此为止，该你登场了"。推理时我们也用 `format_input` 拼出指令、停在 `### Response:`，模型就知道该接着往下生成回答。
> **f-string 小抄**：`f"...{变量}..."` 里花括号会被替换成变量的值；`\n` 是换行。模板里那些 `\n\n` 就是空一行，让格式更清爽、分隔更明显。
> **练习 7.1（书）**：改用 Phi-3 模板（另一种风格）重训，看回答质量有没有变化——模板不是唯一的，但同一次训练里必须**首尾一致**。

拼完模板，按老规矩把 1,100 条**切成训练 / 验证 / 测试三份**（代码清单 7.3，和第 6 章分类微调时一样的套路）：

```python
# 代码清单 7.3　数据集分区
train_portion = int(len(data) * 0.85)   # 85% 训练
test_portion = int(len(data) * 0.1)     # 10% 测试
val_portion = len(data) - train_portion - test_portion  # 剩下 5% 验证

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]
# Training set length: 935 / Validation: 55 / Test: 110
```

**逐行拆**：用切片把列表分三段。85% 拿去训练，10% 留作最后考试（测试集），5% 在训练途中盯着防过拟合（验证集）。比例是经验值，不必死记。

🔧 **亲手改一下**：把 `data[999]`（input 为空那条）传进 `format_input`，`print` 出来。**预测**：输出里还会有 `### Input:` 这一段吗？为什么？（提示：盯住那行 `if entry["input"] else ""`。）

---

## 3. `InstructionDataset`：把每条样例预先编码成 token（书 7.3）

和第 6 章的 `SpamDataset` 一个思路：包一个 PyTorch `Dataset` 类，在 `__init__` 里**一次性把所有样例套模板 + 编码成 token ID**，存起来备用。

```python
# 代码清单 7.4　实现指令数据集类
import torch
from torch.utils.data import Dataset

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)
```

**逐行拆**：

- `class InstructionDataset(Dataset)`：继承 PyTorch 的 `Dataset`，约定要实现 `__getitem__`（按下标取一条）和 `__len__`（共多少条），`DataLoader` 才会用。
- `__init__` 里循环每条 `entry`：
  - `format_input(entry)`：套模板得到"指令+输入"部分（**不含答案**）。
  - `response_text`：单独拼上 `### Response:` + 标准答案。
  - `full_text = ... + ...`：**指令+输入+答案，整条拼成一段完整文字**。注意这里和 `format_input` 不同——**训练样本必须包含答案**，因为答案就是模型要学着生成的"正确下一词序列"。
  - `tokenizer.encode(full_text)`：用第 2 章的 BPE 分词器把整段文字编码成一串 **token ID**（整数列表），存进 `self.encoded_texts`。
- `__getitem__` 直接返回**已编码好的那条**（一串整数）。**注意它返回的是个 Python 列表，长度不一**——这就埋下了下一节要解决的大问题。

> **关键观察**：每条样例编码后**长度都不一样**（指令有长有短、答案有详有简）。但神经网络要把多条样例**叠成一个整齐的张量批次**（matrix）一起算——张量要求**每行一样长**。长短不齐怎么叠成矩形？这就是下一节 `custom_collate_fn` 要啃的硬骨头。

数据集还要约定一个**填充 token（padding token）**——用来把短句"垫长"。本章沿用 GPT-2 的 `<|endoftext|>`，它的 ID 是 **50256**：

```python
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
# [50256]
```

---

## 4. ⭐ 全章最难：`custom_collate_fn`——批内填充、右移目标、`-100` 掩码（书 7.3）

这一节是整章的**心脏**，慢慢来。要解决三件环环相扣的事：

1. **填充对齐**：把一批里长短不一的样例，垫到**同样长**，才能叠成张量。
2. **造目标**：训练要"猜下一个词"，所以每个 `input` 都要配一个**右移一格**的 `target`（答案）。
3. **掩掉填充**：垫进去的那些填充 token 是**凑数的废料**，绝不能让模型"学着去生成它们"——用 `-100` 把它们从损失计算里**屏蔽掉**。

书里很贴心，**分三稿**把这个函数搭起来，每稿只加一件事。我们就跟着爬。

> ### 🐍 加油站 ③ — 什么是 collate function（裁剪/整理函数）？
> `DataLoader` 每次从 `Dataset` 抓出 `batch_size` 条样例（一个 list），然后要把它们**合并成一个批次张量**交给模型——干这个合并活的函数就叫 **collate function（整理函数）**。
> PyTorch 有个**默认**的 collate，但它假设每条样例**已经一样长**。我们这里长短不一，默认的会直接报错。所以**得自己写一个**，塞给 `DataLoader` 的 `collate_fn=` 参数。它接收"一批样例的列表"，吐出"整理好的批次张量"。本节就是一步步把它写出来。

### 4.1 第一稿：只做填充对齐

```python
def custom_collate_draft_1(
    batch,
    pad_token_id=50256,
    device="cpu"
):
    batch_max_length = max(len(item)+1 for item in batch)   # 本批最长 + 1
    inputs_lst = []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]                          # 先在末尾加一个 50256
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))              # 再垫到本批最长
        )
        inputs = torch.tensor(padded[:-1])                  # 去掉最后一个，得到 inputs
        inputs_lst.append(inputs)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    return inputs_tensor
```

**逐行拆**（盯住"垫到一样长"这个目标）：

- `batch_max_length = max(len(item)+1 for item in batch)`：找出**本批**里最长那条的长度，**再 +1**。为什么 +1？因为下面每条都要先**额外加一个填充 token**（这个 token 后面造 target 时有用，见 4.2）。
- 循环每条 `item`：
  - `new_item = item.copy()`：复制一份，别改原数据。
  - `new_item += [pad_token_id]`：先在末尾**加一个 50256**。
  - `padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))`：再用 50256 **重复垫**到 `batch_max_length` 那么长。`[x] * n` 是"把这个列表重复 n 次"，所以这是垫上"差多少补多少"个填充 token。
  - `inputs = torch.tensor(padded[:-1])`：**去掉最后一个**元素，转成张量。（为什么去掉一个？还是为了和 target 对齐，下一稿揭晓。）
- `torch.stack(inputs_lst)`：把这一批长度已经**全部相等**的张量，**叠成一个二维批次张量** `[batch_size, 序列长]`。

跑一下感受："`inputs_1=[0,1,2,3,4]`、`inputs_2=[5,6]`、`inputs_3=[7,8,9]`"三条拼一批：

```
tensor([[ 0,  1,  2,  3,  4],
        [ 5,  6, 50256, 50256, 50256],
        [ 7,  8,  9, 50256, 50256]])
```

短的两条被 **50256 垫到和最长那条一样宽**，于是叠成了一个整齐的 3×5 矩形。**填充对齐，搞定。**

> ### 🐍 加油站 ④ — 为什么要 padding？`torch.stack` 是什么
> **神经网络一次处理一批样例，要求这批数据是个规整的矩形张量**——每行一样长。可真实句子长短不一，怎么办？**把短的用一个"无意义的填充 token"垫到和最长的一样长**，凑成矩形。这就是 **padding（填充）**。
> 本章的填充 token 用 `<|endoftext|>`（50256），它本来就表示"文本结束"，拿来当垫片很自然。
> **`torch.stack([a, b, c])`**：把若干个**形状相同**的张量**摞成一摞**，多出一个新维度。三个长度都是 5 的一维张量，stack 后变成 `[3, 5]` 的二维张量。**前提是它们形状必须一致**——这正是我们费劲做填充对齐的原因：不先垫齐，`stack` 直接报错。
> **省填充的小聪明**：注意 `batch_max_length` 是按**本批**算的，不是按整个数据集最长算的。**每批只垫到这批最长**，不同批可以不同宽（后面你会看到 loader 打印出 `[8,61]`、`[8,76]` 等不同宽度）。这样**最大限度少垫废料**，省算力。

### 4.2 第二稿：再造一个"右移一格"的目标

填充对齐只给了 `inputs`。但训练要算损失，得有**正确答案 `targets`**。和预训练时一模一样的招（第 5 章）：**目标就是输入向右移一格**——位置 `i` 的输入，要去预测位置 `i` 的目标，而那个目标正是"原序列里的下一个 token"。

```python
def custom_collate_draft_2(
    batch,
    pad_token_id=50256,
    device="cpu"
):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])    # 去掉最后一个 -> 输入
        targets = torch.tensor(padded[1:])    # 去掉第一个 -> 目标（右移一格）
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor
```

**逐行拆**（关键就两行）：

- `inputs = torch.tensor(padded[:-1])`：去掉**最后一个**。
- `targets = torch.tensor(padded[1:])`：去掉**第一个**。

这一去头、一去尾，`targets` 就比 `inputs` **整体右移了一格**：`inputs[i]` 对应的正确"下一个词"正好是 `targets[i]`。现在也明白 4.1 里那两处怪操作了——**先 +1 个填充、再 `[:-1]`/`[1:]` 错位切**，就是为了能切出这对错开一格的孪生张量。

跑出来：

```
inputs:                              targets（右移一格）:
tensor([[ 0,  1,  2,  3,  4],        tensor([[ 1,  2,  3,  4, 50256],
        [ 5,  6, 50256, 50256, 50256],        [ 6, 50256, 50256, 50256, 50256],
        [ 7,  8,  9, 50256, 50256]])          [ 8,  9, 50256, 50256, 50256]])
```

对一下第一行：input `[0,1,2,3,4]`，target `[1,2,3,4,50256]`——看到 0 该猜 1，看到 1 该猜 2……看到 4 该猜 50256（文本结束）。**完美错开一格。**

### 4.3 第三稿（最终版）：用 `-100` 把多余的填充掩掉

第二稿有个隐患：`targets` 里**那一堆 50256 填充**，会被当成"正确答案"去算损失——等于在**逼模型学着生成一连串填充 token**，纯属带歪。我们要把这些**多余的填充从损失里剔除**。

办法很巧：把目标里**多余的填充 token 替换成 `-100`**。PyTorch 的交叉熵损失**默认会忽略值为 `-100` 的目标**（下面加油站细讲）。但要**留一个 50256 不替换**——因为我们**希望模型学会"回答完了就输出一个结束 token"**，这个本事得留着。

```python
# 代码清单 7.5　实现自定义批量整理功能
def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id              # 标出 targets 里所有 50256
        indices = torch.nonzero(mask).squeeze()     # 这些 50256 的位置
        if indices.numel() > 1:                     # 如果不止一个
            targets[indices[1:]] = ignore_index     # 从第二个起，全改成 -100

        if allowed_max_length is not None:          # 可选：截断到上限长度
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor
```

**逐行拆（只看新增的掩码三行，这是精华）**：

- `mask = targets == pad_token_id`：逐元素比较，得到一个**布尔张量**——`targets` 里凡是 50256 的位置标 `True`，其余 `False`。
- `indices = torch.nonzero(mask).squeeze()`：`nonzero` 找出所有 `True` 的**位置下标**，`squeeze()` 把多余的维度压平成一串下标。
- `if indices.numel() > 1: targets[indices[1:]] = ignore_index`：如果填充 token **多于一个**，就**从第二个开始（`indices[1:]`）全部改成 `-100`**。**精髓在这里——保留第一个 50256（让模型学"该收尾了"），其余全部掩掉（不算损失）。**
- `allowed_max_length`：可选参数。如果设了，就把 `inputs`/`targets` **截断**到这个长度。后面 loader 会把它设成 1024（GPT-2 的上下文上限），防止超长样本撑爆模型。

跑出来，对比第二稿：

```
inputs（没变）:                      targets（多余填充已变 -100）:
tensor([[ 0,  1,  2,  3,  4],        tensor([[ 1,  2,  3,  4, 50256],
        [ 5,  6, 50256, 50256, 50256],        [ 6, 50256, -100, -100, -100],
        [ 7,  8,  9, 50256, 50256]])          [ 8,  9, 50256, -100, -100]])
```

看第二行 target `[6, 50256, -100, -100, -100]`：**第一个 50256 留着**（教模型收尾），**后面的填充全变 `-100`**（损失里直接无视）。这正是我们要的效果。

> ### 🐍 加油站 ⑤ — `-100` 凭什么能"被损失忽略"？（本章最该懂的机制）
> 这不是魔法，是 **PyTorch 交叉熵的一个默认设置**：`cross_entropy(..., ignore_index=-100)`。`ignore_index` 默认就是 `-100`——意思是"**目标里凡是等于 -100 的位置，直接跳过、不计入损失**"。
> 书里用一个小实验把这事钉死：
>
> ```python
> import torch
> logits = torch.tensor([[-1.0, 1.0], [-0.5, 1.5], [-0.5, 1.5]])
> targets_a = torch.tensor([0, 1, 1])
> print(torch.nn.functional.cross_entropy(logits, targets_a))   # tensor(0.7936)
>
> targets_b = torch.tensor([0, 1, -100])   # 把第三个目标换成 -100
> print(torch.nn.functional.cross_entropy(logits, targets_b))   # tensor(1.1269)
> ```
> 把第三个目标从 `1` 换成 `-100`，损失就**完全等于只算前两个**的结果（`1.1269`，正是只有 `[0,1]` 两个目标时的损失）——**第三个被彻底无视了**。（顺带：换成 `-100` 以外的越界值会直接报错，因为那不是合法类别下标。）
> **为什么是 -100 而不是别的数？** 纯粹是 PyTorch 选的默认哨兵值。负数天然不可能是合法的 token 下标（token ID 都 ≥ 0），拿它当"请忽略我"的暗号最安全。
> **一句话**：我们借 `ignore_index=-100` 这个机制，让**填充 token 不污染训练**——模型只为"真正该生成的内容"挨罚。

> **延伸（书里提到，不在本章实现）**：除了掩掉填充，还可以**连指令部分的 target 也掩成 -100**，让损失**只算到回答（response）那一段**——逼模型专注"生成好回答"而非"背诵指令"。本章为简单起见**不掩指令**（学界对该不该掩还有分歧）。这是**练习 7.2** 留给你的可选实验。

🔧 **亲手改一下**：把 `custom_collate_fn` 里那行 `if indices.numel() > 1:` 改成 `if indices.numel() > 0:`（从 0 开始就掩），重跑示例批次。**预测**：target 里那个**留着的 50256 会不会也变成 -100**？这样模型还学得会"何时收尾"吗？（联系上面"为什么要保留第一个结束 token"。）

---

## 5. 数据加载器：把整理函数插进 `DataLoader`（书 7.4）

零件齐了：`InstructionDataset`（出样例）+ `custom_collate_fn`（整理成批）。现在把它们插进 `DataLoader`，让它自动洗牌、分批、整理。

但 `custom_collate_fn` 有几个参数（`device`、`allowed_max_length`）要**预先固定**，`DataLoader` 的 `collate_fn=` 只接受"一个吃 batch 的函数"。怎么把多余参数提前填好？用 `functools.partial`：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from functools import partial
customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024     # 截断到 GPT-2 的上下文上限
)
```

**逐行拆**：

- `device = ...`：有 NVIDIA GPU 就用 `cuda`，否则 `cpu`（Apple Silicon 可用 `mps`，但书里提醒 mps 实验性、数值可能有出入）。
- `partial(custom_collate_fn, device=..., allowed_max_length=1024)`：**把 `device` 和 `allowed_max_length` 这两个参数"焊死"**，造出一个**只剩 `batch` 一个参数**的新函数 `customized_collate_fn`——正好能塞给 `DataLoader`。
- **顺带一个工程巧思**：把"搬到 device"放进 collate（在 `DataLoader` 的后台进程里做），而不是放训练循环里——这样**搬数据和 GPU 算训练能并行**，不互相堵着。

> ### 🐍 加油站 ⑥ — `functools.partial`：给函数"预填参数"
> `partial(f, x=1)` 的意思是"**拿函数 `f`，把它的参数 `x` 先填成 1，造一个新函数**"。以后调用这个新函数就不用再传 `x` 了。
> 这里我们的 `custom_collate_fn` 有 5 个参数，但 `DataLoader` 只肯喂它 1 个（`batch`）。`partial` 把另外几个**提前填好**，剩下那个 `batch` 交给 `DataLoader` 现填——天作之合。
> 记法：**`partial` = "锁住几个参数，留一个口子"。**

然后三个 loader 一气呵成（代码清单 7.6）：

```python
# 代码清单 7.6　初始化数据加载器
from torch.utils.data import DataLoader
num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,   # 关键：用我们的自定义整理函数
    shuffle=True,                       # 训练要打乱
    drop_last=True,
    num_workers=num_workers
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, collate_fn=customized_collate_fn,
    shuffle=False, drop_last=False, num_workers=num_workers
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, collate_fn=customized_collate_fn,
    shuffle=False, drop_last=False, num_workers=num_workers
)
```

**逐行拆**：套路和第 6 章一样——每个数据集包一个 `DataLoader`。三处要点：① `collate_fn=customized_collate_fn` 把我们的整理函数挂上去；② 训练集 `shuffle=True`（打乱防止学到顺序）、验证/测试 `False`；③ `drop_last=True` 让训练批大小恒定。

检查一下批次形状：

```python
for inputs, targets in train_loader:
    print(inputs.shape, targets.shape)
# torch.Size([8, 61]) torch.Size([8, 61])
# torch.Size([8, 76]) torch.Size([8, 76])
# torch.Size([8, 73]) torch.Size([8, 73])  ...
```

`8` 是 `batch_size`；后面那个数（61、76、73…）是**这一批的序列长度，每批不同**——正是 §4 那个"每批只垫到本批最长"的省料策略在起效。`inputs` 和 `targets` 形状永远一致（一对孪生张量）。

🔧 **亲手改一下**：如果你显存/内存吃紧，把 `batch_size` 从 8 改成 4 或 2 再看形状。**预测**：每行第一个数会变成几？序列长度（第二个数）会变吗？（提示：一个是你定的，一个是数据决定的。）

---

## 6. 加载预训练的 GPT-2 medium（355M）并摸个底（书 7.5）

数据备齐，该请出**被微调的主角**了。代码和第 5 章加载 GPT-2 权重**几乎一字不差**，唯一区别：把型号从 `"gpt2-small (124M)"` 换成 `"gpt2-medium (355M)"`。

```python
# 代码清单 7.7　加载预训练模型
from gpt_download import download_and_load_gpt2
from chapter04 import GPTModel
from chapter05 import load_weights_into_gpt

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}
model_configs = {
    "gpt2-small (124M)":  {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)":    {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()
```

**逐行拆**：

- `BASE_CONFIG`：通用配置。注意 `qkv_bias: True`、`drop_rate: 0.0`——和第 5 章加载 OpenAI 权重时一致（OpenAI 的 GPT-2 用了 QKV 偏置）。
- `model_configs`：四种规模的尺寸表。**medium 比 small 宽**（`emb_dim` 768→1024）**也更深**（`n_layers` 12→24）——这就是它"容量更大、学得动指令"的来源。
- `CHOOSE_MODEL = "gpt2-medium (355M)"`：选中 medium（下载约 1.42 GB，是 small 的三倍）。
- `GPTModel(BASE_CONFIG)`：第 4 章那个模型类，**原样复用**。
- `load_weights_into_gpt(model, params)`：第 5 章那个"把 OpenAI 权重逐层灌进我们模型"的函数，**原样复用**。
- `model.eval()`：切到评估模式（关 dropout 等），先用来看微调前的水平。

**微调前先摸个底**（这步很重要——有基线才知道微调到底有没有用）。拿验证集第一条试：

```python
torch.manual_seed(123)
input_text = format_input(val_data[0])

from chapter05 import generate, text_to_token_ids, token_ids_to_text
token_ids = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer),
    max_new_tokens=35,
    context_size=BASE_CONFIG["context_length"],
    eos_id=50256,
)
generated_text = token_ids_to_text(token_ids, tokenizer)
response_text = generated_text[len(input_text):].strip()
print(response_text)
```

**逐行拆**：

- `format_input(val_data[0])`：套模板生成指令（停在 `### Response:`，没有答案——让模型自己补）。
- `generate(...)`：第 5 章那个生成函数，原样复用，让模型续写 35 个新 token。
- **`generated_text[len(input_text):]`**：这是个**关键小技巧**——`generate` 返回的是"输入+输出**连在一起**"的整段文字。我们只想要模型**新生成的回答**，所以**按字符长度把开头的输入部分切掉**，剩下的 `.strip()` 去掉首尾空白，就是模型的纯回答。

结果就是 §0 那个尴尬现场：模型**只会复读指令、续写下一条指令**，根本没"听懂"。**这正是微调要修的病。**

---

## 7. 训练：复用第 5 章的训练循环（书 7.6）

最让人安心的一节——**指令微调的训练，和预训练用的是同一套代码**。因为本质没变：还是"喂一批 input/target，算交叉熵损失，反向传播，更新参数"。前面那些 `-100` 掩码、padding，都已经在数据这一层处理好了，训练循环**完全不用知道**。

```python
from chapter05 import calc_loss_loader, train_model_simple

model.to(device)

# 先看初始损失（基线）
torch.manual_seed(123)
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss   = calc_loss_loader(val_loader,   model, device, num_batches=5)
# Training loss: 3.826 / Validation loss: 3.762
```

```python
# 代码清单 7.8　指令微调预训练的 LLM
import time
start_time = time.time()
torch.manual_seed(123)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
num_epochs = 2

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[0]), tokenizer=tokenizer
)

end_time = time.time()
print(f"Training completed in {(end_time - start_time)/60:.2f} minutes.")
```

**逐行拆**：

- `calc_loss_loader(...)`（第 5 章函数）：先算个**初始损失**（≈3.8）当基线。`torch.no_grad()` 是"只看不学"，省内存。
- `optimizer = AdamW(..., lr=0.00005, weight_decay=0.1)`：优化器。**注意学习率很小（5e-5）**——微调是"轻轻地改"已经很会的模型，步子太大会把预训练学到的本事抖没了。
- `num_epochs = 2`：**只训 2 轮**。书里强调：2 轮损失就稳稳降下来了，**再多反而容易过拟合**（小数据集，背答案）。
- `train_model_simple(...)`：**第 5 章那个训练循环，原样复用**。它内部就是你在第 0 章背过的母版五步（前向→算损失→清梯度→反向→更新），外加按 `eval_freq` 定期打印损失、用 `start_context` 生成一段看看进步。
- `start_context=format_input(val_data[0])`：每隔一阵让模型试答验证集那条"改被动语态"，**亲眼看它一点点学会**。

训练时终端会打印（节选）：

```
Ep 1 (Step 000000): Train loss 2.637, Val loss 2.626
Ep 1 (Step 000005): Train loss 1.174, Val loss 1.103
...
Ep 2 (Step 000230): Train loss 0.300, Val loss 0.657
... ### Response: The meal is cooked every day by the chef.<|endoftext|> ...
Training completed in 0.87 minutes.   # A100 GPU；M3 MacBook Air CPU 约 15.8 分钟
```

损失从 ~2.6 一路降到 ~0.3，而且**那条被动语态题答对了**——"The meal is cooked every day by the chef."。**对比 §0 微调前的复读机，这就是 SFT 的威力：底层模型没大变，变的就是这层调教。**

> **再次扣主线**：训练目标**从头到尾还是"预测下一个词"**——交叉熵损失就是在罚"下一个词猜得不像标准答案"。指令微调只是**换了一批特殊的训练数据**（指令→回答对、且用 `-100` 把不该学的填充屏蔽掉），让"预测下一个词"这身本事，**对齐到"按指令作答"这个方向**。

🔧 **亲手改一下**：把 `num_epochs` 改成 1 重训，看最终验证损失和回答质量。再改成 5，观察验证损失到后面是不是**不降反升**（过拟合的征兆）。**预测**：哪个 epoch 数最划算？（联系书里"2 轮就够、再多弊大于利"。）

---

## 8. 提取并保存模型的回答（书 7.7）

训练完，要在**没见过的测试集**上验收。先把模型对每条测试样例的回答**生成出来、抽干净、存盘**，方便后面打分。

先看前 3 条对不对（人眼粗判）：

```python
torch.manual_seed(123)
for entry in test_data[:3]:
    input_text = format_input(entry)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )
    print(input_text)
    print(f"\nCorrect response:\n>> {entry['output']}")
    print(f"\nModel response:\n>> {response_text.strip()}")
    print("-------------------------------------")
```

**逐行拆**：

- `generate(..., max_new_tokens=256, eos_id=50256)`：让模型最多生成 256 个新 token，遇到结束 token（50256）就停——这就是为什么我们前面特意**保留了一个 50256 让它学收尾**（§4.3），现在它知道答完该停了。
- `generated_text[len(input_text):]`：老技巧，切掉输入只留回答。
- `.replace("### Response:", "")`：模型有时会把 `### Response:` 标记也吐出来，顺手抹掉。
- `.strip()`：去首尾空白。

抽样结果（节选）——模型答得**相当不错**：

```
Instruction: Rewrite the sentence using a simile.  Input: The car is very fast.
Correct: The car is as fast as lightning.    Model: The car is as fast as a bullet.   ✓（不同比喻但对）

Instruction: What type of cloud is typically associated with thunderstorms?
Correct: cumulonimbus.    Model: a cumulus cloud.    ≈（接近但不全对）

Instruction: Name the author of 'Pride and Prejudice.'
Correct: Jane Austen.    Model: The author of 'Pride and Prejudice' is Jane Austen.   ✓
```

然后把**全部 110 条测试**的回答生成出来，塞回字典、存成 JSON（代码清单 7.9）：

```python
# 代码清单 7.9　生成测试集响应
from tqdm import tqdm
for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
    input_text = format_input(entry)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )
    test_data[i]["model_response"] = response_text   # 把回答塞回这条字典

with open("instruction-data-with-response.json", "w") as file:
    json.dump(test_data, file, indent=4)
```

**逐行拆**：和上面 3 条一样的生成+抽取，只是**遍历全部 110 条**，把每条的回答存进 `test_data[i]["model_response"]`，最后 `json.dump` 整体写盘。`tqdm` 给个进度条（M3 Air 上约 6 分钟）。这样回答就**落盘留档**了，下一节打分时直接读。

最后，把微调好的模型权重也存下来，以后能直接复用：

```python
import re
file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-sft.pth"   # gpt2-medium355M-sft.pth
torch.save(model.state_dict(), file_name)
# 重载：model.load_state_dict(torch.load("gpt2-medium355M-sft.pth"))
```

**逐行拆**：`re.sub(r'[ ()]', '', ...)` 把型号名里的空格和括号去掉，拼成文件名（`-sft` 即 supervised fine-tuned）。`torch.save(model.state_dict(), ...)` 存的是**第 5 章学过的 `state_dict`**（模型所有参数的字典）。**这个 `.pth` 文件就是你亲手调教出的助手——存下来，下次直接加载就能用。**

---

## 9. 评估：请另一个更大的 LLM 来打分（书 7.8）

最后一关：**怎么量化"答得好不好"？** 分类微调时简单——对就对、错就错，算个准确率。但**对话/回答没有唯一标准答案**（"as fast as a bullet" 和 "as fast as lightning" 都对）。书里点出三种业界评估法：① 选择题基准（如 MMLU）；② 人类偏好对战（如 LMSYS Arena）；③ **用另一个 LLM 自动打分**（如 AlpacaEval）。

110 条让人一条条读太累，所以选第三种：**请一个更强的模型（Meta 的 Llama 3 8B）给我们的回答打 0–100 分**。用开源的 **Ollama** 在本地跑这个裁判模型（装好后命令行 `ollama run llama3` 即可，约需 16 GB 内存；`ollama serve` 要一直开着）。

先用 Python 通过 **REST API** 和 Ollama 对话（代码清单 7.10）：

```python
# 代码清单 7.10　本地 Ollama 模型查询
import urllib.request
import json

def query_model(
    prompt,
    model="llama3",
    url="http://localhost:11434/api/chat"
):
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {
            "seed": 123,
            "temperature": 0,     # 要确定性的评分，不要它发挥
            "num_ctx": 2048
        }
    }
    payload = json.dumps(data).encode("utf-8")
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    response_data = ""
    with urllib.request.urlopen(request) as response:
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]
    return response_data
```

**逐行拆**：

- `data = {...}`：按 Ollama 的接口约定打包请求——`model` 指定裁判（llama3），`messages` 放我们的提问（`role: user`），`options` 里 **`temperature: 0`** 让评分尽量稳定（不要它即兴发挥）、`seed: 123` 固定随机性。
- `payload = json.dumps(data).encode(...)`：把 Python 字典转成 JSON 字节串发出去。
- `urllib.request.Request(url, ..., method="POST")`：向本地 `localhost:11434`（Ollama 默认端口）发 POST 请求。
- 那个 `while True` 循环：Ollama **流式**返回（一行一小段），循环把每行的 `message.content` **拼接**成完整回答。

> ### 🐍 加油站 ⑦ — REST API：程序之间怎么"对话"
> **API** 是"程序对外的服务窗口"。Ollama 跑起来后，会在你电脑的 `localhost:11434` 开一个窗口（**本地服务**），别的程序可以往这个地址**发请求、收回复**。
> 我们的 Python 不直接 import Llama 3（它是另一个独立程序），而是**打包一段 JSON 请求 POST 过去**，Ollama 算完把结果**回传**。这种"发 JSON 请求、收 JSON 回复"的网络接口就叫 **REST API**——你平时调 OpenAI、调任何云端大模型，底层都是这一套。
> 这里只是把裁判模型当成一个"**外部服务**"来用，**和我们自己训练的 GPT-2 完全是两码事**：我们的模型是被考的考生，Llama 3 是请来的阅卷老师。

有了 `query_model`，就能让 Llama 3 当裁判。先看它怎么打分（节选）：

```python
for entry in test_data[:3]:
    prompt = (
        f"Given the input `{format_input(entry)}` "
        f"and correct output `{entry['output']}`, "
        f"score the model response `{entry['model_response']}`"
        f" on a scale from 0 to 100, where 100 is the best score. "
    )
    print(">>", query_model(prompt))
```

裁判会给出**带理由的评分**，比如对 "as fast as a bullet" 打了 85 分（"比喻用对了，子弹确实快；只是没 lightning 那么有画面感"），对 "cumulus cloud" 打了 40 分（"把积云说成雷暴云，事实错了"）——**点评得相当在理**。

但带理由的文字没法算平均分。所以加一句**"只回整数"**，造一个批量打分函数（代码清单 7.11）：

```python
# 代码清单 7.11　评估指令微调的 LLM
def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."     # 关键：只回一个整数
        )
        score = query_model(prompt, model)
        try:
            scores.append(int(score))                    # 转成整数
        except ValueError:
            print(f"Could not convert score: {score}")   # 偶尔没回纯数字就跳过
            continue
    return scores

scores = generate_model_scores(test_data, "model_response")
print(f"Number of scores: {len(scores)} of {len(test_data)}")
print(f"Average score: {sum(scores)/len(scores):.2f}")
# Number of scores: 110 of 110 / Average score: 50.32
```

**逐行拆**：

- 提示词比上面多了 `"Respond with the integer number only."`——逼裁判**只吐一个整数**，方便机器解析。
- `int(score)`：把裁判回的字符串转成整数。包在 `try/except ValueError` 里——万一裁判没听话、回了一段话而非纯数字，就**打印警告并跳过这条**，不让整个流程崩。
- 遍历全部 110 条打分，最后求**平均分 ≈ 50.32**。

**这个 50.32 怎么读？** 它本身不是"及格线"，而是一个**基准（baseline）**——以后你换更大的模型、调超参、加更多数据，**就拿这个分数比**，看有没有进步。书里给的参照：Llama 3 8B **基础模型**约 58.5 分，Llama 3 8B **指令版**（在大得多的指令数据上微调过）高达 **82.6** 分。**我们这个 355M 的小 GPT-2 用 935 条数据能到 50+，已经证明了整条 SFT 流水线是通的。**

> **注意**：Ollama 跨系统**不完全确定性**，你跑出来的分数可能略有出入；想要稳，可以多评几次取平均。

🔧 **亲手改一下**：把 `query_model` 里的 `temperature` 从 0 调到 1，对同一条回答连问三次，看评分稳不稳。**预测**：温度高了，裁判打分会更飘还是更稳？（联系第 5 章"温度调稳 vs 野"。）

---

## 10. 练习

> 📓 **对照官方答案**：卡住了别硬磕——[在 Colab 打开本章练习解答 notebook ↗](https://colab.research.google.com/github/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/exercise-solutions.ipynb)

### 书上原题

**练习 7.1：更换提示词模板。**
用 Alpaca 风格训完后，改用 Phi-3 风格（图 7.4 的另一种模板）重训，看回答质量有没有变化。
<details><summary>解题思路（点开）</summary>

改 `format_input` 里那串模板文字即可（Phi-3 用 `<|user|>` / `<|assistant|>` 这类标记，而非 `### Instruction:`）。**关键**：训练和推理必须用**同一套**模板——训的时候用 Phi-3，评估时也得用 Phi-3 拼提示，否则模型认不出"该我答了"的信号。**收获**：模板只是个"约定信号"，本身没有对错，但**全程必须一致**。
</details>

**练习 7.2：指令与输入掩码。**
训完后，把**指令和输入部分的 target 也换成 -100**（图 7.13 的指令掩码法），让损失只算回答部分，再看性能有没有提升。
<details><summary>解题思路（点开）</summary>

在 `custom_collate_fn` 里，除了掩填充，还要找出"`### Response:` 之前"那段的位置，把对应 target 设成 `ignore_index`(-100)。这样损失**只惩罚回答部分**，模型专注"答得好"而非"背指令"。**注意**：学界对掩不掩指令有分歧（Shi 等 2024 发现不掩反而更好），所以这是**实验题，不是标准答案**——亲手跑一遍对比平均分。**收获**：彻底理解 `-100` 这个机制能用来"选择性地决定哪些位置参与训练"。
</details>

**练习 7.3：在原始 Alpaca 数据集上微调。**
斯坦福的 Alpaca 数据集有 52,002 条（比本章数据多约 50 倍），换它来训。
<details><summary>解题思路（点开）</summary>

把下载的数据换成 Alpaca 的 52k 条即可（格式同样是 instruction/input/output）。**强烈建议用 GPU**，数据大了 50 倍。爆内存就把 `batch_size` 从 8 降到 4/2/1，或把 `allowed_max_length` 从 1024 降到 512/256。**收获**：体会"数据量、批大小、序列长度"如何共同决定显存占用。
</details>

**练习 7.4：用 LoRA 做参数高效微调。**
改用附录 E 的 LoRA 方法微调，对比训练时间和性能。
<details><summary>解题思路（点开）</summary>

LoRA 只训练注入的少量低秩矩阵、冻结原模型大部分参数，省显存、提速。这正好预告**概念课第 7 天"工程现实"**那一章——LoRA / RLHF / RAG / Agent 都是长在这套主干上的"枝"。**收获**：明白指令微调（全参数）只是众多微调法之一，工业界更常用 LoRA 这种"省着调"的方案。
</details>

### 本册自测题（改了会怎样）

1. **不留结束 token 会怎样**：把 §4.3 的 `if indices.numel() > 1` 改成把**所有** 50256 都掩成 -100（一个不留）。重训后用 `generate` 看模型——它还知道"答完该停"吗？回答会不会**停不下来、一直续写**？（联系"保留一个结束 token 教模型收尾"。）
2. **学习率调大**：把 `lr=0.00005` 改成 `lr=0.001`（大 20 倍）重训，盯住训练损失。它会更快降，还是**直接发散/崩坏**？为什么微调要用很小的学习率？（联系"轻轻地改已经很会的模型"。）
3. **换小模型**：把 `CHOOSE_MODEL` 换回 `"gpt2-small (124M)"` 重训重评，平均分会比 medium 高还是低？这能不能印证书里"小模型容量不够、学不动指令"的说法？

---

## 11. 本章小结

1. **指令微调（SFT）= 给模型看大量「指令→理想回答」样例，把"自动续写机"调教成"听话的问答助手"。** 底层那台"预测下一个词"的机器几乎没变，变的是**这层调教方向**——这就是 GPT-3 → ChatGPT 的关键一步（之后还可叠 RLHF 偏好对齐）。
2. **本章七成功夫在数据，几乎不写新模型代码**：`GPTModel`、`train_model_simple`、`generate`、`load_weights_into_gpt` 全是第 4、5 章原样复用。
3. **`format_input` 套 Alpaca 模板**（`### Instruction:` / `### Input:` / `### Response:`）给模型一个稳定信号——"看到 `### Response:` 就该我答了"。`input` 为空时整段跳过。
4. **`custom_collate_fn` 是全章心脏，干三件事**：① **padding** 把一批垫到等长好叠成张量（`torch.stack`，且**只垫到本批最长**省料）；② **右移一格**造出 target（`[:-1]` 配 `[1:]`）；③ 用 **`-100` 掩掉多余填充**（借 `cross_entropy(ignore_index=-100)` 让它们不计损失），但**保留一个结束 token** 教模型"何时收尾"。
5. **训练就是第 5 章那套**：小学习率（5e-5）、只 2 个 epoch（防过拟合），损失从 ~2.6 滚到 ~0.3，那条"改被动语态"从复读变答对——SFT 的威力肉眼可见。
6. **评估没有唯一答案，请更大的 LLM（Llama 3 via Ollama）当裁判打 0–100 分**，求平均分当基准（本章 ≈50；Llama 3 指令版可达 82.6）。打分模型是"外部服务"，通过 REST API 调用，和我们的 GPT-2 是两码事。
7. **主线贯穿到底**：指令微调的训练目标**仍然是"预测下一个词"**——只是换了特殊数据、并用 `-100` 屏蔽了不该学的部分，把这身本事**对齐到"按指令作答"**。

---

## 12. 能改自检清单（全勾＝过关）

- [ ] 我能解释"续写机"和"听话助手"的区别，以及为什么预训练完的模型不会"回答"只会"接龙"。
- [ ] 我能说清 `format_input` 套模板的作用——`### Response:` 这个标记到底给了模型什么信号。
- [ ] 给我 `custom_collate_fn`，我能逐段讲出它怎么 padding、怎么造右移 target、怎么用 -100 掩填充，以及**为什么要保留一个 50256**。
- [ ] 我能用书里那个小实验解释"为什么 `-100` 会被 `cross_entropy` 忽略"（`ignore_index` 默认值）。
- [ ] 我能说出为什么指令微调要换成 355M 而非 124M、为什么学习率很小、为什么只训 2 个 epoch。
- [ ] 我能解释"切掉输入只留回答"那行 `generated_text[len(input_text):]` 在干什么。
- [ ] 我能讲清为什么评估要请另一个 LLM 打分、平均分 50 是什么意思，以及它和 Llama 3 指令版 82.6 的差距说明了什么。
- [ ] 我能独立完成练习 7.1、7.2，并预测每个自测题改动的结果再验证。

---

## 13. 通往下一章

到这里，**这本书的"从零造一个 LLM"主线就走完了**——你亲手把一台 GPT 从架构搭起来、预训练喂聪明、再指令微调调教成了能听懂指令、会回答的助手。

但工业界的真实助手还多了几道工序，正是**概念课第 7 天「工程现实」**要讲的：**LoRA**（省着调，练习 7.4 已预告）、**RLHF / 偏好微调**（在 SFT 之后再用人类偏好对齐，书里作为可选下一步指了路）、**RAG**（让模型查资料再答）、**Agent**（让模型会用工具、自己规划）。**它们全都是长在你这一章造好的主干上的"枝"——主干没变，枝越加越多。**

> 带走主线：指令微调让"预测下一个词"这身本事**对齐到"按指令作答"**——你现在用的每一个 AI 助手，本质都还是那台**预测下一个词**的机器，只是被一层层调教得越来越听话。**一切的底层，仍然是预测下一个词。**
