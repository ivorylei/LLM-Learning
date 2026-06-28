# 深化册 · 附录 D　让训练更稳的三个小技巧：热身 / 余弦衰减 / 梯度裁剪

> **对应**：概念课 **第 7 天（工程现实）** ｜ 书 **附录 D（为训练循环添加技巧）**
> **一句话直觉**：三个让大模型训练更稳的实用技巧——**学习率热身、余弦衰减、梯度裁剪**——都是给 ch5 那个训练循环加的小补丁。原理一行没动，只是让它**更稳、更省**。
> **这一章你将亲手得到**：一个加了这三招、更稳的训练循环（`train_model`）——它就是 ch5 那个 `train_model_simple` 升级版，**多了三处、其余照旧**。
>
> **前置**（卡住就翻回去）：
> - ch05 的 `train_model_simple` 训练循环（取数据→算损失→`backward`→`optimizer.step`）→ 本章假设你已经读过它
> - 学习率是什么、优化器 `optimizer` 在干嘛 → 本章 [加油站 ①](#加油站1)
> - 张量、形状、`backward`、`.grad` → 见 ch03 加油站

---

[![在 Colab 中打开本章](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rasbt/LLMs-from-scratch/blob/main/appendix-D/01_main-chapter-code/appendix-D.ipynb)

> 配套可运行 notebook（rasbt/LLMs-from-scratch）。**边读边跑、跟着每节的「🔧 亲手改一下」改一改。**

## 0. 三分钟回顾（概念课第 7 天直觉）

概念课第 7 天讲过一句话：**主干已成，后面那些花活都是"枝"。** 模型结构（注意力、Transformer 块）和训练目标（猜下一个词）是主干；而工程师在真实训练里加的一堆小技巧，是为了让这棵树**长得更稳**。

本章的三招，正是这种"枝"——它们**完全不改模型、不改"预测下一个词"这个目标**，只动一件事：**每一步用多大的步子去更新参数（学习率），以及更新幅度别失控（梯度裁剪）**。

打个比方。训练就是"蒙着眼下山找最低点"（损失最小值）：

- **学习率** = 你每一步迈多大。
- **热身（warmup）** = 刚出发先小碎步，别一上来就大跨步摔跤。
- **余弦衰减（cosine decay）** = 快到谷底了就放慢脚步，免得一脚迈过头、又冲上对面山坡。
- **梯度裁剪（gradient clipping）** = 万一某一步算出来"要迈 100 米"，强行把它压回"最多迈 1 米"，防止一步把模型踹飞。

> 三招都是给 ch5 的训练循环打补丁。读完你会发现：**升级版 `train_model` 和原版 `train_model_simple` 逐行对照，只多了三块。** 这正是"读懂+能改"的最好练习——看清"原版"和"加料版"差在哪。

本章先把每招单独跑通（D.1～D.3），最后（D.4）把三者一起焊进训练函数。

> 全章沿用 ch5 的实验设置：一个 124M 的小 GPT，拿《The Verdict》这篇短篇小说当训练数据。数据太小，几轮就会过拟合——但这里的目的不是练出好模型，而是**看清这三招怎么改训练循环**。

---

## 0.5 先把舞台搭好（书中代码，本节只讲思路）

附录 D 开头先重新初始化了 ch5 那个模型和数据加载器，好让代码自成一体：

```python
import torch
from chapter04 import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
model.eval()
```

**逐行拆**（不用细抠，这就是 ch4/ch5 那个模型）：`GPTModel` 是你前面亲手搭过的那块 GPT；`GPT_CONFIG_124M` 是它的尺寸表（5 万词表、768 维、12 层、12 头）。`model.to(device)` 把模型搬到 GPU（有的话）。然后照 ch2 的老办法，把《The Verdict》读进来、切成 `train_loader` / `val_loader` 两个数据加载器（90% 训练、10% 验证）——**这些都是前几章的原班代码，本附录直接复用，不展开**。

舞台搭好，下面三招登场。

---

## 1. 学习率热身：开局先小步走（书 D.1）

### 1.1 要解决什么

直接拿一个**大学习率**开训复杂模型（比如 LLM），风险很高：训练刚开始时参数还是随机的、损失曲面很陡，一上来就大步更新，容易"一脚踩空"——损失突然飙升甚至变成 `NaN`，整轮训练废掉。

**热身（warmup）** 的办法很朴素：**别一上来就用最大学习率，而是从一个很小的初始值，在前若干步里线性地、一点点涨到最大值。** 开局小步走，等模型"找到点感觉"了，再放开步子。

### 1.2 贴书真实代码：先定参数

```python
n_epochs = 15
initial_lr = 0.0001       # 初始学习率（很小）
peak_lr = 0.01            # 热身要涨到的最大学习率（峰值）
warmup_steps = 20
```

热身要走多少步？书里的经验：**总步数的 0.1% ～ 20%**。可以这么算：

```python
total_steps = len(train_loader) * n_epochs   # 总步数 = 每轮批次数 × 轮数
warmup_steps = int(0.2 * total_steps)        # 取 20%
print(warmup_steps)   # 27
```

**逐行拆**：
- `total_steps`：一个 epoch 要走 `len(train_loader)` 步（数据被切成这么多批），乘以 `n_epochs` 轮，就是整个训练总共更新多少次参数。
- `int(0.2 * total_steps)`：取总步数的 20% 当热身步数，这里算出来是 **27**。意思是：**前 27 步**里，学习率从 `0.0001` 线性爬到 `0.01`；之后保持在 `0.01`。

### 1.3 贴书真实代码：把热身塞进循环

下面是一个"只演示学习率怎么变"的精简循环（还没真训练，只是记录每步的学习率）：

```python
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1)
lr_increment = (peak_lr - initial_lr) / warmup_steps   # 每步涨多少

global_step = -1
track_lrs = []
for epoch in range(n_epochs):
    for input_batch, target_batch in train_loader:
        optimizer.zero_grad()
        global_step += 1

        if global_step < warmup_steps:
            lr = initial_lr + global_step * lr_increment   # 热身期：线性往上爬
        else:
            lr = peak_lr                                    # 热身后：用峰值

        # 把算好的 lr 写回优化器（关键一步）
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        track_lrs.append(optimizer.param_groups[0]["lr"])
```

**逐行拆**：
- `lr_increment = (peak_lr - initial_lr) / warmup_steps`：把"要涨的总量"（0.01 − 0.0001）平摊到热身的每一步上，得到"每步涨多少"。
- `global_step`：一个**全局步数计数器**，跨 epoch 连续累加（每处理一个批次 +1）。它决定我们现在处于热身期还是热身后。从 `-1` 起步，循环里先 `+1`，所以第一次进来是 `0`。
- `if global_step < warmup_steps`：**前 27 步**走这条线 `lr = initial_lr + global_step * lr_increment`——第 0 步是 `initial_lr`，每多一步加一个 `lr_increment`，到第 27 步正好涨到 `peak_lr`。
- `else: lr = peak_lr`：热身结束，固定用最大学习率。
- **`for param_group in optimizer.param_groups: param_group["lr"] = lr`**：这是全章最关键的动作——**手动改优化器的学习率**。算出 `lr` 还不够，得把它写回优化器里，下一步 `optimizer.step()` 才会用新值（见加油站 ②）。

跑完画个图（`plt.plot(range(total_training_steps), track_lrs)`），就能看到学习率从低点起步、爬 20 步到峰值的折线——热身生效。

<a name="加油站1"></a>
> ### 🐍 加油站 ① — 学习率是什么？优化器又是什么？（回顾 ch5）
> 训练就是反复做一件事：**算出"参数该往哪个方向、改多少"（梯度），然后照着改一点点。**
> - **梯度（gradient）**：损失对每个参数的"敏感度"——告诉你这个参数往哪动能让损失变小、以及多陡。`loss.backward()` 负责算它，结果存在每个参数的 `.grad` 里。
> - **学习率（learning rate, lr）**：一个小数，控制"照着梯度改多少"。`新参数 = 旧参数 − lr × 梯度`。lr 太大→步子迈太猛、容易摔（损失震荡/爆炸）；太小→走得太慢、半天到不了谷底。它是最关键的超参数之一。
> - **优化器（optimizer）**：替你执行"照梯度更新参数"的工具。本章用 **`AdamW`**（Adam 的改良版，自带 `weight_decay` 权重衰减防过拟合）。你只跟它打两个交道：`optimizer.zero_grad()`（清空上一步的梯度）和 `optimizer.step()`（用当前梯度+学习率更新参数）。
> 本章三招里有两招（热身、余弦衰减）干的事其实是同一件：**在每一步动态地改这个 `lr`**。

> ### 🐍 加油站 ② — `optimizer.param_groups` 和那个 `["lr"]`
> 优化器内部把参数分成一组组（`param_groups`），每组带自己的设置，**学习率就存在 `param_group["lr"]` 里**。我们的小模型只有一组，所以：
> - **读当前学习率**：`optimizer.param_groups[0]["lr"]`。
> - **改学习率**：遍历所有组，挨个 `param_group["lr"] = 新值`。
> 为什么要遍历而不是直接写一个？因为正经模型可能有多组（比如不同层用不同 lr），遍历能一次全改、不漏。**记住这个套路：想在训练中途调学习率，就是改 `param_groups` 里的 `"lr"`。** 热身和余弦衰减全靠它。

🔧 **亲手改一下**：把 `warmup_steps` 从 `int(0.2 * total_steps)` 改成 `int(0.05 * total_steps)`（只热身 5%），重画 `track_lrs` 的图。预测：爬到峰值的那段折线会变陡还是变缓？为什么？（提示：同样的高度，台阶变少了。）

---

## 2. 余弦衰减：临近谷底就放慢脚步（书 D.2）

### 2.1 要解决什么

热身解决了"开局别摔"，但还有另一头的问题：**训练后期**。如果一直用最大学习率走到底，临近损失最小值时，大步子很容易"一脚迈过头"——冲过谷底、又爬上对面的坡，损失反复横跳、降不下去。

**余弦衰减（cosine decay）** 的办法：**热身结束后，让学习率沿一条余弦曲线（半个余弦周期）平滑地往下降，一直降到接近 0。** 形象地说就是——越接近终点，步子迈得越小，稳稳地"滑"进谷底，而不是"冲"进去。

### 2.2 贴书真实代码：在热身的基础上加衰减

代码和 D.1 几乎一样，只把 `else` 分支（热身之后那段）从"固定用峰值"换成"按余弦曲线往下降"：

```python
import math

min_lr = 0.1 * initial_lr                      # 衰减的最低值（不会真降到 0）
track_lrs = []
lr_increment = (peak_lr - initial_lr) / warmup_steps
global_step = -1

for epoch in range(n_epochs):
    for input_batch, target_batch in train_loader:
        optimizer.zero_grad()
        global_step += 1

        if global_step < warmup_steps:
            # 热身期：和上一节一模一样，线性往上爬
            lr = initial_lr + global_step * lr_increment
        else:
            # 热身后：余弦衰减
            progress = ((global_step - warmup_steps) /
                        (total_training_steps - warmup_steps))
            lr = min_lr + (peak_lr - min_lr) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        track_lrs.append(optimizer.param_groups[0]["lr"])
```

**逐行拆**（只看新增的 `else` 这块，其余和 D.1 同构）：
- `min_lr = 0.1 * initial_lr`：衰减的**地板**——不让学习率真的降到 0，而是降到一个很小的非零值，免得最后几步完全不更新。
- `progress = (global_step - warmup_steps) / (total_training_steps - warmup_steps)`：**衰减进度**，一个从 `0` 到 `1` 的小数。热身刚结束时 `progress=0`，训练快结束时 `progress≈1`。它表示"在'衰减这段路'上走了百分之几"。
- `lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))`：这就是余弦曲线。看那个 `0.5 * (1 + cos(π·progress))`：
  - `progress=0` 时，`cos(0)=1`，整个括号 `= 0.5*(1+1) = 1` → `lr = min_lr + (peak_lr-min_lr)*1 = peak_lr`（衰减起点正好接上峰值）。
  - `progress=1` 时，`cos(π)=−1`，括号 `= 0.5*(1−1) = 0` → `lr = min_lr`（衰减终点落到地板）。
  - 中间则是一条**平滑下凹的余弦曲线**，从 `peak_lr` 顺滑地滑到 `min_lr`。
- 画出来 `track_lrs` 就是：先一段线性爬坡（热身），到顶后一条优雅的余弦下滑线。

> **为什么偏偏用"余弦"这个形状？** 因为它两头平、中间陡：衰减一开始降得慢（让模型在高学习率下多探索一会儿），中段降得快，结尾又变慢（临近谷底精细微调）。这种"慢-快-慢"的节奏，经验上比"直线降"训练得更稳，所以被几乎所有大模型采用。**你不用背公式，记住形状和效果即可。**

<a name="加油站3"></a>
> ### 🐍 加油站 ③ — `math.cos` 和 `math.pi` 在这里干嘛
> `math.cos(x)` 是余弦函数，`math.pi` 就是 π≈3.14159。我们只用到余弦曲线 `[0, π]` 这半段：`cos(0)=1`（最高），`cos(π)=−1`（最低），中间平滑下降。代码里 `math.cos(math.pi * progress)` 让 `progress` 从 0 走到 1 时，自变量从 0 走到 π，于是余弦值从 1 平滑滑到 −1——这正好被我们用来把学习率从 `peak_lr` 拉到 `min_lr`。**只是借用它"平滑下降"的形状，不涉及任何三角函数的深入知识。**

🔧 **亲手改一下**：把 `min_lr = 0.1 * initial_lr` 改成 `min_lr = peak_lr`（地板和峰值一样高），重画 `track_lrs`。预测：曲线还会下降吗？（想想 `(peak_lr - min_lr)` 这一项会变成几。）这能帮你确认：是这个差值在驱动整条衰减曲线。

---

## 3. 梯度裁剪：单步更新别失控（书 D.3）

### 3.1 要解决什么

前两招调的是学习率（步子多大）。但还有个隐患在**梯度本身**：有时某一步算出来的梯度会**异常巨大**（这叫"梯度爆炸"）。哪怕学习率不大，`lr × 巨大梯度` 仍可能是个吓人的更新量，一步就把好不容易学到的参数全毁了，损失瞬间炸成 `NaN`。

**梯度裁剪（gradient clipping）** 的办法：**给梯度设一个长度上限（`max_norm`）。如果这一步所有梯度合起来的"总长度"超过上限，就按比例整体缩小，让它的长度正好等于上限；没超过就不动。** 相当于给每一步更新装了个"限速器"。

### 3.2 贴书真实代码：先制造一次梯度，再裁剪

先正常算一次损失、反向传播，让梯度填进每个参数的 `.grad`：

```python
from chapter05 import calc_loss_batch

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)

loss = calc_loss_batch(input_batch, target_batch, model, device)
loss.backward()
```

为了"看见"梯度有多大，书里写了个小工具，扫一遍所有参数、找出**最大的那个梯度值**：

```python
def find_highest_gradient(model):
    max_grad = None
    for param in model.parameters():
        if param.grad is not None:
            grad_values = param.grad.data.flatten()
            max_grad_param = grad_values.max()
            if max_grad is None or max_grad_param > max_grad:
                max_grad = max_grad_param
    return max_grad

print(find_highest_gradient(model))   # tensor(0.0411)
```

**逐行拆**：
- `for param in model.parameters()`：遍历模型每一个参数张量。
- `if param.grad is not None`：只看那些已经算出梯度的（`backward()` 之后才有 `.grad`）。
- `param.grad.data.flatten()`：把这个参数的梯度张量**摊平成一长串数**，方便取最大值。
- `.max()` + 那个 `if`：在所有参数里滚动比较，留住**全局最大**的梯度值。
- 结果 `tensor(0.0411)`：裁剪**之前**，最大梯度是 0.0411。

现在做裁剪，再看一次：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
print(find_highest_gradient(model))   # tensor(0.0185)
```

**逐行拆**：
- `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`：PyTorch 现成的裁剪函数。它把模型**所有参数的梯度看成一个超长向量**，算它的总长度（L2 范数）；若 > `max_norm`（这里 1.0），就整体按比例缩小到长度恰好 1.0。**注意名字带下划线 `clip_grad_norm_` → 原地修改**，直接改 `.grad`，不返回新东西。
- 结果从 `0.0411` 降到 `0.0185`：所有梯度被等比例压小了，最猛的那个也被拉了回来。**注意**：是整体等比缩放，所以梯度之间的相对大小（方向）没变，只是幅度被压住——更新方向不变、力度收住。

<a name="加油站4"></a>
> ### 🐍 加油站 ④ — `max_norm` 和"范数"是什么
> **范数（norm）** 就是"向量的长度"。最常用的是 **L2 范数**（欧几里得范数）：把每个分量平方、求和、再开根号——就是中学学的"勾股定理"推广到多维。书里的例子：梯度向量 `[1,2,3,4]` 的 L2 范数 = √(1²+2²+3²+4²) = √30 ≈ 5.48；若 `max_norm=1`、当前长度是 5，就乘一个缩放因子 `1/5`，把每个分量都缩到原来的 1/5，新长度正好 1。
> **`max_norm=1.0`** 就是给"所有梯度合起来的总长度"设的上限。超了就按 `max_norm / 当前长度` 这个比例整体缩。`1.0` 是最常用的默认值——既能拦住爆炸，又不会平时没事乱改梯度。

> ### 🐍 加油站 ⑤ — 为什么梯度会"爆炸"？
> 深层网络反向传播时，梯度是从最后一层**一层层连乘**回最前层的。如果中间某些环节的数值偏大，连乘几十层后就可能滚雪球般变得巨大（这就是"梯度爆炸 gradient explosion"）；反过来连乘很多小于 1 的数会趋近 0（"梯度消失"）。LLM 又深又大，偶尔抽风蹦出一个巨大梯度并不稀奇。梯度裁剪不去管它为啥变大，只在**出手更新前**强行把总幅度按住——一个简单粗暴但极其有效的"保险丝"。

🔧 **亲手改一下**：把 `max_norm=1.0` 改成 `max_norm=0.01`（卡得很死），重跑那两行裁剪+打印。预测：裁剪后的最大梯度会比 `0.0185` 更小还是更大？（提示：上限压得越狠，整体缩得越多。）

---

## 4. 三招合一：改进后的训练函数（书 D.4）

### 4.1 要解决什么

三招都验证过了，现在把它们**一起焊进** ch5 的训练循环。书里的升级版叫 `train_model`，对照 ch5 的 `train_model_simple`，**只多了三处**：① 每步动态算 `lr`（热身+余弦衰减）并写回优化器；② `backward` 之后、`step` 之前插一句梯度裁剪；③ 多记一个 `track_lrs` 方便事后画图。**其余——取批次、`zero_grad`、算损失、`backward`、`step`、定期评估、采样——和原版一字不差。**

### 4.2 贴书真实代码：升级版 `train_model`

```python
from chapter05 import evaluate_model, generate_and_print_sample
import math

def train_model(model, train_loader, val_loader, optimizer, device,
                n_epochs, eval_freq, eval_iter, start_context, tokenizer,
                warmup_steps, initial_lr=3e-05, min_lr=1e-6):
    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    tokens_seen, global_step = 0, -1

    # 从优化器里读出"峰值学习率"（构造 optimizer 时设的那个 lr）
    peak_lr = optimizer.param_groups[0]["lr"]
    total_training_steps = len(train_loader) * n_epochs
    lr_increment = (peak_lr - initial_lr) / warmup_steps

    for epoch in range(n_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1

            # ===== 新增①：算这一步的学习率（热身 + 余弦衰减）=====
            if global_step < warmup_steps:
                lr = initial_lr + global_step * lr_increment
            else:
                progress = ((global_step - warmup_steps) /
                            (total_training_steps - warmup_steps))
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (
                     1 + math.cos(math.pi * progress))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            track_lrs.append(lr)
            # ====================================================

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()

            # ===== 新增②：热身结束后才裁剪梯度 =====
            if global_step > warmup_steps:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
            # =====================================

            optimizer.step()
            tokens_seen += input_batch.numel()

            # 下面这段定期评估 + 打印，和 ch5 原版一样
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader,
                    device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Iter {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}")
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen, track_lrs
```

**逐行拆（只看相对 `train_model_simple` 的三处新增/变化，其余照旧）**：
- `peak_lr = optimizer.param_groups[0]["lr"]`：**峰值学习率不再写死，而是从优化器里读**——你构造 `optimizer` 时设的那个 `lr` 就被当成"要爬到的最高点"。这样换实验只改一处。
- **新增① 学习率调度**：`global_step < warmup_steps` 走线性热身，否则走余弦衰减——和 D.1+D.2 拼起来一模一样，算完照例 `param_group["lr"] = lr` 写回优化器。
- **新增② 梯度裁剪**：注意位置——**必须在 `loss.backward()` 之后**（梯度已经算出来了）、**`optimizer.step()` 之前**（还没拿梯度去更新），夹在中间这一句裁剪才有意义。还有个细节：`if global_step > warmup_steps` ——**热身阶段先不裁**，等过了热身期再开启裁剪。
- **`track_lrs.append(lr)`**：顺手记下每步学习率，训练完能画曲线复盘。
- 其余每一行——`zero_grad` / `calc_loss_batch` / `backward` / `step` / `eval_freq` 定期评估 / `generate_and_print_sample` 采样 / 返回那一堆列表——**和 ch5 的 `train_model_simple` 完全相同**。这正是"加补丁"的含义：主干没动，三处微创。

### 4.3 贴书真实代码：跑起来

```python
import tiktoken

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)

peak_lr = 5e-4   # 注意：峰值通过 optimizer 的 lr 传进去
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1)
tokenizer = tiktoken.get_encoding("gpt2")
n_epochs = 15

train_losses, val_losses, tokens_seen, lrs = train_model(
    model, train_loader, val_loader, optimizer, device, n_epochs=n_epochs,
    eval_freq=5, eval_iter=1, start_context="Every effort moves you",
    tokenizer=tokenizer, warmup_steps=warmup_steps,
    initial_lr=1e-5, min_lr=1e-5
)
```

**逐行拆**：
- 重新初始化一个全新模型（固定种子，保证可复现）。
- `optimizer = AdamW(..., weight_decay=0.1)`：构造优化器，它内部的 `lr` 就成了 `train_model` 里读到的 `peak_lr`。
- `tokenizer = tiktoken.get_encoding("gpt2")`：GPT-2 的分词器，采样时把生成的 token 转回文字用。
- 调用 `train_model`，把三招要用的参数（`warmup_steps`、`initial_lr`、`min_lr`）一并传进去。

书里给的运行输出（在 MacBook Air 上约 5 分钟）：

```
Ep 1 (Iter 000000): Train loss 10.934, Val loss 10.939
Ep 1 (Iter 000005): Train loss 9.151, Val loss 9.461
Every effort moves you,,,,,,,,,,,,...
...
Ep 15 (Iter 000130): Train loss 0.041, Val loss 6.915
Every effort moves you?" "Yes--quite insensible to the irony. She wanted him
vindicated--and by me!" He laughed again, ...
```

**怎么读这段输出**：训练损失从 10.9 一路降到 0.041——**函数在工作，损失确实被压下去了**。但验证损失（`Val loss`）降到 6 左右就不动甚至回升了——这是**过拟合**，因为《The Verdict》数据太小、被反复迭代了 15 轮。这跟 ch5 的现象一样，**不是这三招的锅**：三招让训练更稳，但救不了"数据太少"。书里也提醒：换个更大的数据集再来对比，才看得出这个升级版的真正好处。

🔧 **亲手改一下**：把调用里的 `warmup_steps=warmup_steps` 改成 `warmup_steps=0`（完全不热身），其余不变重跑前几个 Iter。预测：开头几步的 `Train loss` 会更平稳还是更容易出现大跳？（联系 D.1：没有热身，开局就是全速学习率。）对照一下两次的前几行输出。

---

## 5. 练习

### 本册自测题（改了会怎样）

1. **关掉余弦衰减**：在 `train_model` 的 `else` 分支里把整段余弦公式换成 `lr = peak_lr`（热身后一直用峰值）。重跑并对比 `lrs` 曲线和最终 `Train loss`。你能说出"少了'临近谷底放慢'会有什么风险"吗？（联系 D.2。）

2. **梯度裁剪的位置**：如果手滑把 `clip_grad_norm_` 那一句放到了 `optimizer.step()` **之后**，它还有用吗？（提示：`step` 已经拿旧梯度更新完参数了，这时再裁剪的是谁？想清楚"裁剪必须夹在 `backward` 和 `step` 之间"。）

3. **三招各管一段**：用一句话分别说出——热身管训练的哪个阶段、余弦衰减管哪个阶段、梯度裁剪管的是哪类突发情况？三者有没有重叠、能不能只留一个？

4. **峰值从哪来**：`train_model` 里 `peak_lr = optimizer.param_groups[0]["lr"]`。如果你把 `optimizer` 构造成 `AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)`，那热身会爬到多高？（答：`1e-3`——峰值就是你给优化器的那个 `lr`。）

---

## 6. 本章小结

1. 三招都是**给 ch5 训练循环打的补丁，不碰模型、不碰"预测下一个词"这个目标**——它们只调"每一步怎么更新参数"。
2. **学习率热身**：开局别用大学习率，前若干步从 `initial_lr` 线性爬到 `peak_lr`，防止冷启动时大步摔跤。
3. **余弦衰减**：热身后让学习率沿余弦曲线平滑降到 `min_lr`，"慢-快-慢"地滑进谷底，防止后期一脚迈过损失最小值。
4. **梯度裁剪**：`torch.nn.utils.clip_grad_norm_(..., max_norm=1.0)` 给每步梯度的总长度设上限，超了就等比缩回，给训练装个"保险丝"防爆炸。
5. **改训练循环的两个机械动作要记牢**：调学习率＝改 `optimizer.param_groups[*]["lr"]`；裁梯度＝在 `loss.backward()` 之后、`optimizer.step()` 之前插一句 `clip_grad_norm_`。
6. 升级版 `train_model` 和原版 `train_model_simple` 逐行对照，**只多三块**——这就是"工程现实"里"主干上加枝"的典型样子。

---

## 7. 能改自检清单（全勾＝过关）

- [ ] 我能说出热身、余弦衰减、梯度裁剪各自解决训练的什么问题。
- [ ] 给我 `train_model`，我能准确指出它比 `train_model_simple` 多了哪三处。
- [ ] 我知道"在训练中途改学习率"就是改 `optimizer.param_groups` 里的 `["lr"]`，并能写出那个 `for` 循环。
- [ ] 我能解释为什么梯度裁剪必须夹在 `backward()` 和 `step()` 之间。
- [ ] 我能看懂 `find_highest_gradient` 输出从 `0.0411` 降到 `0.0185` 说明了什么。
- [ ] 我能解释那段训练输出里"Train loss 降到 0.041 但 Val loss 回升"是过拟合、且不是这三招的锅。

---

## 8. 通往下一章

这三招是概念课第 7 天那句话的活样本：**主干（模型 + 猜下一个词）一动不动，工程师只在训练循环上加几片"枝"，就把训练变稳、变省。** 你以后读任何真实的 LLM 训练代码，几乎都会看到这三招（外加混合精度、梯度累积等更多"枝"）成套出现——现在你能一眼认出它们，并知道每一片枝挂在主干的哪个位置。

> 带走主线：热身、衰减、裁剪让训练**更稳更省**，但模型在每一步学的，**始终还是那件事——把下一个词预测得更准。**
