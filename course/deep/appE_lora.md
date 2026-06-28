# 深化册 · 附录 E　LoRA：给冻住的大模型，旁边挂一小撮新参数

> **对应**：概念课 **第 7 天**（工程现实 · LoRA）｜ 书 **附录 E（使用 LoRA 进行参数高效的微调）**
> **一句话直觉**：把原模型**整个冻住不动**，只在每个线性层旁边**挂一小撮新参数（低秩矩阵 A·B）**来训练——用极少的资源就能微调大模型。这就是「人人都能微调」的底层技术。
> **这一章你将亲手得到**：把[第 6 章](ch06_classify.md)的"分类微调"改造成 **LoRA 微调**，并**对比改造前后的可训练参数量和速度**——亲眼看到 1.24 亿个参数怎么缩到 267 万个。
>
> **前置**（卡住就翻到对应「加油站」）：
> - 分类微调是怎么回事（换输出层、训练循环）→ 本附录建立在 [第 6 章](ch06_classify.md) 之上，建议先读它
> - `nn.Module` / `nn.Linear` / 什么是"参数" → [第 0 章 · Python/PyTorch 加油站](ch00_python_pytorch.md) 与 [ch03 加油站 ④](ch03_attention.md#加油站4)
> - `requires_grad`、矩阵乘 `@` → 本附录 [加油站 ②](#加油站2)、[ch03 加油站 ②](ch03_attention.md#加油站2)

---

[![在 Colab 中打开本章](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rasbt/LLMs-from-scratch/blob/main/appendix-E/01_main-chapter-code/appendix-E.ipynb)

> 配套可运行 notebook（rasbt/LLMs-from-scratch）。**边读边跑、跟着每节的「🔧 亲手改一下」改一改。**

## 0. 三分钟回顾（概念课直觉）

概念课第 7 天讲过一条线：**预训练很贵，但微调可以很便宜。**

把一个大模型从零预训练，要烧几十上百万的算力——这事只有大公司干得起。但你拿到别人训好的模型后，想让它"专门会干某件事"（比如判断垃圾邮件、按你的语气说话），其实**不需要重训全部参数**。

第 6 章的做法是"**全参数微调**"：换上一个新的输出层，然后把**整个模型的 1.24 亿个参数**都拿去训练。能用，但有两个痛点：

1. **贵**：1.24 亿个数都要算梯度、都要更新，显存和时间都吃得多。
2. **存不下**：你给 100 个客户各定制一版，就得存 100 份完整的大模型——每份几百兆甚至几个 G。

LoRA（**Lo**w-**R**ank **A**daptation，低秩适应）换了个思路，正好治这两个痛点：

> **原模型一个字都不改（冻住），只在每个线性层旁边并联一条"小旁路"，训练时只动这条旁路。**

这条旁路小到什么程度？这一章你会亲手数出来：**全参数 1.24 亿 → LoRA 只剩 267 万**，缩了将近 50 倍。存的时候也只存这 267 万个数就行，原模型大家共用一份。

回到概念课第 7 天的比喻：**预训练好的大模型是"主干"，LoRA 就是嫁接在主干上的一根"小枝"。** 主干一旦长成，所有人共享；每个人按自己的需要，往上接一根便宜的小枝。正因为微调被 LoRA 打到这么便宜，开源大模型生态才能这么繁荣——**普通人用一张消费级显卡，就能定制自己的大模型。**

> 这一章的代码完全建立在第 6 章（垃圾邮件分类微调）之上。数据怎么下、模型怎么加载、训练函数 `train_classifier_simple` 怎么写，都是第 6 章的东西，**本章直接 import 复用，不重写**。我们的全部工作，就是在"加载好模型"和"开始训练"之间，**插进 LoRA 这一小段改造**。

---

## 1. LoRA 到底在做什么：W + A·B（书 E.1）

### 1.1 要解决什么

常规微调里，训练在干一件事：给原始权重矩阵 `W` 算出一个**更新量 ΔW**（delta W，"W 的变化量"），然后：

```
W_updated = W + ΔW
```

`ΔW` 和 `W` **一样大**。`W` 有一百万个数，`ΔW` 就有一百万个要学的数——这就是"贵"的根源。

LoRA 的核心洞察是：**这个 ΔW 其实不需要那么"满"。** 微调时模型要学的"调整方向"往往集中在少数几个方向上（这就是"低秩"的数学含义，见加油站 ①）。所以与其老老实实学一个满的 `ΔW`，不如**用两个小矩阵的乘积去近似它**：

```
ΔW ≈ A · B
```

于是权重更新被改写成：

```
W_updated = W + A · B
```

这里 `A` 和 `B` 都比 `W` **小得多**。假设 `W` 是 `[d, d]`（比如 `[768, 768]`），我们让：

- `A` 的形状是 `[d, r]`——把 `d` 维**压**到一个很小的维度 `r`（rank，秩，比如 16）；
- `B` 的形状是 `[r, d]`——再把 `r` 维**升**回 `d` 维。

`A · B` 的结果形状是 `[d, d]`，和 `W` 对得上，能加进去。但 `A` 和 `B` 里的数加起来只有 `d×r + r×d = 2dr` 个，远少于 `W` 的 `d×d` 个。`d=768`、`r=16` 时：满的是 `768×768 ≈ 59 万`，LoRA 的是 `2×768×16 ≈ 2.5 万`——**少了二十多倍**，而且 `r` 越小省得越多。

<a name="加油站1"></a>
> ### 🐍 加油站 ① — 什么是"低秩"？矩阵分解的直觉
> 别被"秩（rank）"这个词吓到，它的直觉很朴素。
> 一个 `[d, d]` 的大矩阵，看起来有 `d×d` 个独立的数；但很多时候，它其实**没那么"满"**——它能被两个瘦长的小矩阵相乘"拼"出来。
> 打个比方：你要描述一张 `1000×1000` 的灰度图（一百万个像素），如果这张图是"竖条纹"，那它其实只由"一行的样子"× "一列的明暗"决定——用一个 `1000×1` 和一个 `1×1000` 相乘就能还原，只要两千个数。这就是 **秩 1**。
> **秩 r**，就是"需要 r 组这样的'行×列'叠加才能拼出来"。`r` 越小，这个矩阵越"简单"、越能被压缩。
> LoRA 赌的是：微调要学的那个 `ΔW`，**本来就是个低秩的、简单的东西**——所以用 `A·B`（中间卡一个小小的 `r`）去近似它，几乎不损失效果，却省掉绝大多数参数。
> `A` 把 `d` 维**压缩**到 `r` 维（像把信息塞进一个窄瓶颈），`B` 再从 `r` 维**还原**回 `d` 维。`r` 是那个瓶颈的宽度，小，所以参数少。

### 1.2 一个关键改写：把旁路和主干"分开算"

如果只是 `W_updated = W + A·B`，那还得把 `A·B` 加回 `W` 里，原模型就被改了。LoRA 真正好用的地方，靠的是一条小学就学过的**乘法分配律**。

设输入数据是 `x`，一层的计算本来是 `x @ W`。微调后应该是 `x @ (W + ΔW)`。把括号拆开：

```
x @ (W + ΔW) = x @ W + x @ ΔW
```

换成 LoRA 的近似：

```
x @ (W + A·B) = x @ W  +  x @ A·B
                └─原层─┘  └─LoRA旁路─┘
```

看出门道了吗？**原来那项 `x @ W` 原封不动**，LoRA 只是在旁边**并联**了一条 `x @ A·B`，最后把两条结果**相加**。这意味着：

- **原模型权重 `W` 全程保持不变**（冻住）——它谁都不影响，可以被所有任务共享。
- LoRA 矩阵 `A`、`B` 是**独立的一小撮**，用的时候动态加上去，不用时摘掉。
- 给 100 个客户定制，就存 100 份小小的 `A`、`B`，**大模型只存一份**。存储和扩展性一下就上去了。

> **一句话记牢**：常规微调是"把 W 改了"；LoRA 是"W 不动，旁边并一条小旁路 A·B，两路相加"。下面所有代码，都是在把这一句话变成 PyTorch。

---

## 2. `LoRALayer`：那条"小旁路"本身（书 E.1 · 代码清单 E.5）

### 2.1 要解决什么

先把"旁路"`x @ A·B`（再乘个缩放因子 alpha）做成一个独立的小层。它的活儿很单纯：**接收输入 `x`，吐出 `alpha × (x @ A @ B)`。**

### 2.2 贴书真实代码

```python
import math
import torch

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))   # [in_dim, r]
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))   # 给 A 一个合理初始值
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))  # [r, out_dim]，全 0 起步
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)   # [.., in_dim] -> [.., out_dim]
        return x
```

### 2.3 逐行拆

- `class LoRALayer(torch.nn.Module)` + `super().__init__()`：老规矩，声明"我是一个网络层"（`nn.Module` 的套路见 [ch03 加油站 ④](ch03_attention.md#加油站4)）。
- `self.A = nn.Parameter(torch.empty(in_dim, rank))`：登记矩阵 **A**，形状 `[in_dim, rank]`——把 `in_dim` 维压到 `rank` 维。`torch.empty` 先开一块没初始化的内存（里面是垃圾值），紧接着下一行就给它填上合理的初始值。
- `nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))`：用 **Kaiming 均匀初始化**给 `A` 填数。这是给线性层权重用的标准初始化方案（`nn.Linear` 内部默认也用它），目的是让初始数值落在一个利于训练的范围里。`a=math.sqrt(5)` 是 PyTorch 对线性层的默认惯例参数，照抄即可，不必深究。
- `self.B = nn.Parameter(torch.zeros(rank, out_dim))`：登记矩阵 **B**，形状 `[rank, out_dim]`——把 `rank` 维升回 `out_dim` 维。**注意它初始化成全 0。** 这一步极其关键，下一段专门讲。
- `self.alpha = alpha`：保存缩放因子 alpha（一个普通数，不是要训练的参数）。
- `forward` 里 `self.alpha * (x @ self.A @ self.B)`：数据流就一行——`x` 先乘 `A`（被压到 `rank` 维），再乘 `B`（升回 `out_dim` 维），最后整体乘上 `alpha` 调一下"音量"。形状从 `[.., in_dim]` 变到 `[.., out_dim]`，和原线性层的输出对得上，待会儿好相加。

> **A 和 B 一压一升，参数就省下来了。** `x` 本来要直接乘一个 `[in_dim, out_dim]` 的大矩阵；现在拆成"先压到 `rank`、再升回去"，中间卡了个细脖子 `rank`。脖子越细（`rank` 越小），要学的数越少。这就是 1.1 节那个 `2dr ≪ d²` 在代码里的样子。

### 2.4 为什么 B 要初始化成全 0？

这是 LoRA 一个特别巧的设计。回想旁路是 `A @ B`：**只要 B 全是 0，那么 `A @ B` 不管 A 是什么，结果都是一个全 0 矩阵。**

于是在**刚挂上 LoRA、还没开始训练的那一刻**：

```
输出 = x @ W  +  alpha × (x @ A @ B)
     = x @ W  +  alpha × 0
     = x @ W                       ← 和原模型一模一样！
```

也就是说，**LoRA 刚装上去时，对模型的输出毫无影响**——模型表现和改造前完全相同。然后训练开始，`B` 才从 0 慢慢被学出非零值，旁路才逐渐"生效"。这保证了改造是**平滑的、不破坏原模型**的：从"完全等于原模型"出发，一点点学出该有的调整。（待会儿 5.4 节你会亲眼看到：挂上 LoRA 后、训练前，准确率和第 6 章一字不差，正是这个原因。）

> ### 🐍 加油站 — `alpha` 和 `rank` 这两个旋钮怎么调？
> - **`rank`（秩 r）**：那个"细脖子"的宽度，**直接决定 LoRA 引入多少新参数**。`rank` 越大，旁路越"有表现力"、能学的东西越多，但参数也越多、越接近全参数微调。它是在"模型适应能力"和"省资源"之间的平衡旋钮。
> - **`alpha`（缩放因子）**：决定旁路输出**对原层输出的影响有多大**（音量旋钮）。`alpha` 越大，LoRA 这条枝说话越响。
> - 经验法则（书里给的）：`rank` 和 `alpha` 都取 **16** 是个不错的默认值；`alpha` 常取成 `rank` 的**一半、相等或两倍**。本章用 `rank=16, alpha=16`。

🔧 **亲手改一下**：把 `LoRALayer` 单独实例化出来试试它的形状变换——
```python
torch.manual_seed(123)
layer = LoRALayer(in_dim=768, out_dim=768, rank=16, alpha=16)
x = torch.randn(2, 5, 768)        # 假装一批数据：[batch=2, 序列=5, 维度=768]
print(layer(x).shape)             # 预测：会打印什么？
print(layer(x).abs().sum())       # 预测：这个和会是多少？（提示：B 是全 0）
```
第一行预测输出 `[2, 5, 768]`（进出维度一样）；第二行预测是 `0.0`——因为 `B` 初始全 0，整条旁路此刻输出全 0。跑一下验证你的预测。

---

## 3. `LinearWithLoRA`：把原 Linear 包起来，并上旁路（书 E.1 · 代码清单 E.6）

### 3.1 要解决什么

`LoRALayer` 只是那条旁路。现在要把它和**一个已经训练好的原始 `nn.Linear`** 拼到一起，做成一个"**主干 + 旁路**"的复合层——它既保留原层的本事，又多了一条可训练的小旁路。这样我们才能**拿它去替换模型里现成的线性层**。

### 3.2 贴书真实代码

```python
class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear                       # 原来那一层，原封不动收进来
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)        # 主干 + 旁路，相加
```

### 3.3 逐行拆

- `def __init__(self, linear, rank, alpha)`：注意第一个参数 `linear`——它不是维度数字，而是**一整个已经存在的 `nn.Linear` 层对象**。我们把现成的层"接管"过来。
- `self.linear = linear`：把传进来的原始线性层**原样存住**。它带着自己训练好的权重 `W`，我们一个数都不动。
- `self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)`：照着原层的进出维度，造一条**尺寸刚好匹配**的旁路。`linear.in_features` / `linear.out_features` 是 `nn.Linear` 自带的属性（它知道自己输入几维、输出几维），直接拿来当旁路的 `in_dim` / `out_dim`——这样两路输出形状必然一致，能相加。
- `forward` 里 `self.linear(x) + self.lora(x)`：**这一行就是 1.2 节那条公式 `x @ W + x @ A·B` 的代码实现。** 左边 `self.linear(x)` 是主干（原层），右边 `self.lora(x)` 是旁路（LoRA），两路结果**逐元素相加**，合成最终输出。

> **它是一个"套子"**。`LinearWithLoRA` 把原 `Linear` 整个套在里面，对外仍然是"输入 x、输出同样形状"的一层——所以它可以**无缝顶替**模型里任何一个 `nn.Linear`，模型其他部分根本察觉不到。而因为 `B` 初始全 0（见 2.4），刚换上去时它的行为**和原层完全一样**。

🔧 **亲手改一下**：自己造一个原始 `Linear`，包成 `LinearWithLoRA`，验证"刚包上时输出不变"——
```python
torch.manual_seed(123)
original = torch.nn.Linear(768, 768)
x = torch.randn(1, 768)

wrapped = LinearWithLoRA(original, rank=16, alpha=16)
print(torch.allclose(original(x), wrapped(x)))   # 预测：True 还是 False？
```
预测是 `True`：因为旁路此刻输出全 0，`linear(x) + 0 == linear(x)`。这正是"平滑改造、不破坏原模型"的体现。

---

## 4. `replace_linear_with_lora`：递归把全模型的 Linear 都换掉（书 E.1）

### 4.1 要解决什么

GPT 模型里藏着**几十个** `nn.Linear`——每个 Transformer 块的 `W_query`、`W_key`、`W_value`、`out_proj`，前馈网络里的两层，还有最后的输出头……（看 5.4 节打印出的模型结构就知道有多密）。我们不可能手动一个个去替换。需要一个函数，**自动钻进模型的每个角落，把碰到的 `nn.Linear` 都换成 `LinearWithLoRA`。**

### 4.2 贴书真实代码

```python
def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # 是线性层：当场换成带 LoRA 旁路的版本
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # 不是线性层：它可能内部还套着别的层，钻进去继续找
            replace_linear_with_lora(module, rank, alpha)
```

### 4.3 逐行拆

- `for name, module in model.named_children()`：`named_children()` 列出这个模块**直接挂着的下一层子模块**，每个给你一对 `(名字, 子模块对象)`。比如对一个 Transformer 块，它会列出 `("att", 注意力模块)`、`("ff", 前馈模块)`、`("norm1", 归一化)`……
- `if isinstance(module, torch.nn.Linear)`：判断这个子模块**是不是**一个原始线性层。
- `setattr(model, name, LinearWithLoRA(module, rank, alpha))`：**是**的话——把这个子模块包成 `LinearWithLoRA`，再用 `setattr` **就地装回原来的位置**（同一个名字 `name`）。等于"把这层抽出来，套上 LoRA，再插回去"。
- `else: replace_linear_with_lora(module, rank, alpha)`：**不是**线性层的话，它内部可能还嵌套着更深的层（比如一个 Transformer 块里又装着注意力模块，注意力模块里才是那几个 Linear）。于是**对它自己再调用一次本函数**，钻进去找——这就是"递归"。一层层往下挖，直到把每个叶子上的 `Linear` 都换掉为止。

> ### 🐍 加油站 — 递归遍历 `named_children()` 替换层
> **模型是一棵"套娃树"**：`GPTModel` 里装着一串 `TransformerBlock`，每个块里装着 `MultiHeadAttention` 和 `FeedForward`，再往里才是一个个 `nn.Linear`。`named_children()` 只看**眼前这一层**的直接孩子，看不到孙子。
> 要把整棵树里所有 `Linear` 都换掉，标准技巧就是**递归**：碰到 `Linear` 就换，碰到"还有孩子的"就走进去对它再来一遍。函数自己调用自己，一层层下钻，直到树的最底层。
> - `isinstance(x, T)`：判断 `x` 是不是类型 `T`（这里是 `nn.Linear`）。
> - `setattr(obj, "name", value)`：等价于 `obj.name = value`，但名字是个字符串变量时只能这么写——替换层时正好需要。

🔧 **亲手改一下**：只想给**注意力层**加 LoRA、不动前馈层，能做到吗？最简单的办法是**先全换，再把不想要的换回来**，或者改判断条件。先想：如果在 `if` 里加一句"名字里带 query/key/value 才换"，函数该怎么改？（提示：`name` 就是子模块的名字字符串，可以 `if isinstance(...) and "W_" in name`。）这正是真实项目里"只给部分层上 LoRA"的常见做法。

---

## 5. 冻结、应用、数参数：亲眼看 1.24 亿缩成 267 万（书 E.1）

工具齐了。现在按顺序做三件事：**① 把原模型整个冻住 → ② 应用 LoRA → ③ 数一数可训练参数量，前后对比。**

> 前面的步骤（下数据、加载 GPT-2、换上二分类输出头 `model.out_head`）都和第 6 章一模一样，本章直接复用（书里是 E.2、E.3 节，照搬 ch06 的 `download_and_unzip_spam_data`、`SpamDataset`、`load_weights_into_gpt`、`calc_accuracy_loader` 等）。**这里只讲 LoRA 独有的改造部分。**（数据/加载代码见书 E.2–E.3，本节不重复。）

### 5.1 先数一遍：冻结前有多少可训练参数

```python
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters before: {total_params:,}")
# Total trainable parameters before: 124,441,346
```

**逐行拆**：
- `model.parameters()`：把模型里**所有**参数张量挨个吐出来。
- `p.numel()`：`number of elements`，一个参数张量里有**多少个数**（比如 `[768, 768]` 的就是 589824 个）。
- `if p.requires_grad`：只数那些**需要梯度、即可训练**的（`requires_grad=True`）。
- `sum(...)`：全加起来。`{total_params:,}` 里的 `:,` 是给大数字加千位逗号，好读。
- 结果 **124,441,346**——一亿两千多万，整个 GPT-2 small（含新的输出头）此刻全都是可训练的。这就是"全参数微调"要动的量。

### 5.2 把原模型整个冻住

```python
for param in model.parameters():
    param.requires_grad = False

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters after: {total_params:,}")
# Total trainable parameters after: 0
```

**逐行拆**：
- `for param in model.parameters(): param.requires_grad = False`：遍历每一个参数，把它的 `requires_grad` 关成 `False`——告诉 PyTorch"**这个数别算梯度、别更新它**"。一圈下来，整个模型被**冻住**。
- 再数一次可训练参数：**0**。此刻模型里没有任何东西会被训练——它彻底变成了一块"只读的主干"。

<a name="加油站2"></a>
> ### 🐍 加油站 ② — `requires_grad=False` 是怎么"冻住"参数的
> 每个参数张量身上都有个开关 `requires_grad`：
> - `True`：训练时 PyTorch 会**为它计算梯度**，优化器据此更新它的值——它会"学习、变化"。
> - `False`：**不算梯度、不更新**，这个数从此固定不动——它被"冻住"了。
> LoRA 的整套玩法就建立在这个开关上：先把**原模型全部冻住**（`requires_grad=False`），让它一个数都不变；下一步装上的 LoRA 参数默认是 `True`（`nn.Parameter` 新建出来就可训练），于是训练时**只有那一小撮 LoRA 参数在动**。冻住的部分前向传播照常参与计算（提供 `x @ W`），只是反向传播时被跳过——省下的就是大头。

### 5.3 应用 LoRA，再数一次

```python
replace_linear_with_lora(model, rank=16, alpha=16)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable LoRA parameters: {total_params:,}")
# Total trainable LoRA parameters: 2,666,528
```

**逐行拆**：
- `replace_linear_with_lora(model, rank=16, alpha=16)`：第 4 节那个递归函数出场，把模型里**每一个** `nn.Linear` 都换成 `LinearWithLoRA`。被换进来的原 `Linear` 仍然是冻住的（5.2 关过开关了），但**新挂上的 `LoRALayer` 里的 A、B 是 `nn.Parameter`，默认 `requires_grad=True`**——它们是新鲜的、可训练的。
- 再数可训练参数：从 0 涨到 **2,666,528**（267 万）。这 267 万**全是 LoRA 旁路**的参数。

**对比一下这两个数**：

| | 可训练参数量 | 占原模型 |
|---|---|---|
| 全参数微调（第 6 章） | 124,441,346（1.24 亿） | 100% |
| **冻住后** | 0 | 0% |
| **LoRA（rank=16）** | **2,666,528（267 万）** | **约 2.1%** |

也就是说，**LoRA 只训练原模型 2% 出头的参数量，把可训练参数砍掉了将近 50 倍。** 想多调一点？把 `rank` 调大，可训练参数随之增多——这是你手里的旋钮。

### 5.4 验证：结构真的改了，但行为暂时没变

打印一下模型，确认 `Linear` 都被换成了 `LinearWithLoRA`：

```python
print(model)
```

会看到每个注意力的 `W_query/W_key/W_value/out_proj`、前馈的两层、输出头 `out_head`，都变成了这样的嵌套结构（节选）：

```
(W_query): LinearWithLoRA(
  (linear): Linear(in_features=768, out_features=768, bias=True)   ← 原层，冻住
  (lora): LoRALayer()                                              ← 新旁路，可训练
)
```

每个 `LinearWithLoRA` 里都清清楚楚装着两件东西：**冻住的原 `linear` + 可训练的 `lora`**——和我们 3.2 节写的类一一对应。

再算一次"训练前"的分类准确率：

```python
# train/val/test accuracy（用 ch06 的 calc_accuracy_loader）
# Training accuracy: 46.25%
# Validation accuracy: 45.00%
# Test accuracy: 48.75%
```

**这三个数和第 6 章（还没微调时）一模一样。** 为什么挂了 LoRA 还是没变？**因为 B 初始化成全 0**（2.4 节那个巧思）——`A·B` 此刻是全 0 矩阵，旁路一点没生效，模型行为完全等同于改造前。改造是"无损接入"的，真正的学习还没开始。

---

## 6. 训练：复用第 6 章的训练函数（书 E.1 · 代码清单 E.7）

### 6.1 要解决什么

模型已经改造完毕：主干冻住、旁路待训。**接下来的训练，和第 6 章一字不差**——因为对训练循环来说，它根本不在乎参数是"全部"还是"只有 LoRA"，它只管"把所有 `requires_grad=True` 的参数往降损失的方向推"。而现在 `requires_grad=True` 的，恰好只剩那 267 万个 LoRA 参数。

### 6.2 贴书真实代码

```python
import time
from chapter06 import train_classifier_simple    # ← 直接复用第 6 章的训练函数

start_time = time.time()
torch.manual_seed(123)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 5
train_losses, val_losses, train_accs, val_accs, examples_seen = \
    train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50, eval_iter=5,
        tokenizer=tokenizer
    )

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")
```

### 6.3 逐行拆

- `from chapter06 import train_classifier_simple`：**整个训练循环原样借用第 6 章的**（前向、算损失、反向、更新、定期评估，那一整套）。我们没有为 LoRA 改训练逻辑——这正是 LoRA 优雅的地方：**它只改"模型长什么样"，不改"模型怎么训"。**
- `optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)`：注意这里仍然传 `model.parameters()`（全部参数）。看似把冻住的也交给了优化器，但**冻住的参数 `requires_grad=False`、不产生梯度，优化器自然不会动它们**——所以实际只更新 LoRA 那 267 万个。`lr=5e-5` 是学习率，`weight_decay` 是正则化，都沿用第 6 章。
- `train_classifier_simple(...)`：跑 `num_epochs=5` 轮；`eval_freq=50` 每 50 步评估一次；`eval_iter=5` 评估时只抽 5 个批次估个大概（快）。返回训练/验证的损失和准确率曲线数据。
- `time.time()` 前后一夹：测出总训练时长。

### 6.4 结果与速度

训练日志（节选）：

```
Ep 1 (Step 000000): Train loss 3.820, Val loss 3.462
Ep 1 (Step 000100): Train loss 0.111, Val loss 0.229
Training accuracy: 97.50% | Validation accuracy: 95.00%
...
Ep 5 (Step 000600): Train loss 0.000, Val loss 0.056
Training accuracy: 100.00% | Validation accuracy: 97.50%
Training completed in 12.10 minutes.
```

跑完在完整数据集上评估：

```
Training accuracy:   100.00%
Validation accuracy:  96.64%
Test accuracy:        98.00%
```

**读这组结果**：

- **效果几乎追平全参数微调**。第 6 章全量微调的准确率也在这个量级（测试集约 95–98%）。我们只训练了 **2% 出头的参数**，却拿到了和动全部参数相当的结果——这就是 LoRA"参数高效"的实证。训练集 100%、验证/测试略低（96.64% / 98.00%），说明有一点点过拟合，但泛化得相当好。
- **关于速度的诚实话**：在这个**小模型**上，LoRA 训练**反而比第 6 章慢一点**（书里在 M3 MacBook Air 上约 12–15 分钟）——因为前向传播多了一条旁路 `x @ A @ B` 的计算。但 LoRA 真正省的是**反向传播**：模型越大，"给全部参数算梯度并更新"越贵，而 LoRA 只给那一小撮算。所以**模型一大，LoRA 通常就比全参数微调更快、更省显存**——它本来就是为大模型设计的。

> **回到主线**：你现在做的事，本质还是第 6 章那件事——**调模型，让它把"下一步该输出 spam 还是 ham"预测得更准**。LoRA 没有改变这个目标，它只是把"为达到这个目标要动多少参数"从 1.24 亿压到了 267 万。**预测下一个词（或下一个类别）的主线没变，变的是微调的"性价比"。**

---

## 7. 练习

### 本册自测题（改了会怎样）

1. **把 rank 调大**：把 `replace_linear_with_lora(model, rank=16, alpha=16)` 改成 `rank=32`、再试 `rank=4`。先**预测**可训练参数量会怎么变（变多还是变少？大约几倍？），再重新数一遍验证。
   <details><summary>参考（点开）</summary>

   LoRA 参数量大致和 `rank` **成正比**（每层是 `in×r + r×out`）。`rank` 从 16 翻到 32，可训练参数约**翻倍**（≈ 530 万）；降到 4，约**降到四分之一**（≈ 67 万）。`rank` 就是"花多少参数去适应"的旋钮——大则更能学、更接近全参数微调，小则更省。
   </details>

2. **如果忘了冻结会怎样**：把 5.2 节那段"冻结所有参数"的循环**删掉**，直接 `replace_linear_with_lora` 再数参数。预测：可训练参数量会是多少？这还算"参数高效"吗？
   <details><summary>参考（点开）</summary>

   不冻结的话，原模型的 1.24 亿参数**全都还是可训练的**，再加上新挂的 267 万 LoRA 参数——可训练量变成 **1.27 亿**，比全参数微调还多！LoRA 省钱的**前提**就是先把原模型冻住（`requires_grad=False`）。冻结这一步不是可选项，是 LoRA 的命根子。
   </details>

3. **只给注意力层上 LoRA**：照第 4 节「亲手改」的提示，改 `replace_linear_with_lora` 的判断条件，让它**只替换注意力里的 `W_query/W_key/W_value`**、不动前馈层。预测参数量会比"全换"少多少？跑出来对得上吗？
   <details><summary>参考（点开）</summary>

   前馈层（`[768,3072]` 和 `[3072,768]`）维度比注意力层大得多，是参数大户。只给注意力上 LoRA，可训练参数会**明显减少**。这也是真实项目的常见取舍：在"省多少"和"效果掉多少"之间挑层来上 LoRA。改法：`if isinstance(module, torch.nn.Linear) and "W_" in name`（再把 else 分支保留以便继续递归）。
   </details>

---

## 8. 本章小结

1. **LoRA 的核心一句话**：原模型冻住不动，在每个线性层**旁边并联一条小旁路 `A·B`**，训练只动旁路。靠的是分配律 `x @ (W + A·B) = x @ W + x @ A·B`，让主干和旁路**分开算、最后相加**。
2. **"低秩"省在哪**：`A` 把维度压到一个很小的 `rank`，`B` 再升回去。中间这个细脖子让参数量从 `d²` 降到 `2dr`——`rank` 越小省得越多。
3. **三个类，一层套一层**：`LoRALayer`（旁路本身）→ `LinearWithLoRA`（原层 + 旁路，`forward` 里相加）→ `replace_linear_with_lora`（递归把全模型的 `Linear` 都换成前者）。
4. **B 初始全 0 是点睛之笔**：保证刚挂上 LoRA 时 `A·B=0`、模型行为和改造前完全一样（准确率一字不差），训练从"等于原模型"平滑出发。
5. **冻结是前提**：`requires_grad=False` 把原模型 1.24 亿参数全关掉，新 LoRA 参数默认可训练，于是只有那 267 万在学——可训练参数砍掉将近 50 倍。
6. **训练完全复用第 6 章**：LoRA 只改"模型结构"，不改"训练循环"。效果几乎追平全参数微调，而真正的省钱（省反向传播）在大模型上才充分显现。

---

## 9. 能改自检清单（全勾＝过关）

- [ ] 我能用一句话说清 LoRA 在做什么，并写出 `x @ (W + A·B) = x @ W + x @ A·B` 这条改写。
- [ ] 我能解释"低秩"为什么省参数（`A` 压、`B` 升、中间卡个小 `rank`）。
- [ ] 我能说出 `LoRALayer` → `LinearWithLoRA` → `replace_linear_with_lora` 三者各干什么、怎么套在一起。
- [ ] 我能解释"B 初始化成全 0"带来什么效果，以及为什么这很重要。
- [ ] 我能说清为什么必须先 `requires_grad=False` 冻结，以及不冻会怎样。
- [ ] 给我一个模型，我能数出它的可训练参数量，并预测加 LoRA / 改 rank 后这个数怎么变。
- [ ] 我能独立完成三道自测题，先预测再跑验证。

---

## 10. 通往下一章

LoRA 是概念课第 7 天那棵树上最重要的一根"枝"：**主干（预训练大模型）一旦长成，所有人共享；每个人按需接一根便宜的小枝，就能定制出自己的模型。** 正因为微调被打到这么便宜，开源生态才会这么繁荣——一张消费级显卡 + 几百兆的 LoRA 文件，普通人就能拥有"专属的大模型"。

> 带走主线：从注意力到 GPT、到预训练、到微调、再到 LoRA，所有这些零件和技巧，**最终都只服务于同一件事——把下一个词预测得更准，并把这份能力，用尽可能低的成本，交到更多人手里。**
