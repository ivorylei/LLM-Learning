# 深化册 · 第 0 章　Python / PyTorch 加油站：把"造模型的工具"先摸熟

> **对应**：概念课 **第 1 天**（动手部分）｜ 书 **附录 A（PyTorch 入门）**
> **一句话直觉**：后面每一章都在玩同三样东西——**张量**（装数字的多维表格）、**autograd**（自动算梯度的引擎）、**nn.Module**（搭网络的乐高底座）。这一章把它们一次性摸熟，后面就不慌。
> **这一章你将亲手得到**：一个**完整跑通的训练循环**——从造数据、搭网络、算损失、反向传播、更新参数，到评估、存取模型、用 GPU。这套循环是第 5 章预训练、第 6/7 章微调的**母版**。
>
> **怎么用这一章**：它是"工具手册"，不必一口气背完。**建议先通读一遍建立印象，后面各章的「🐍 加油站」侧栏卡住时再回来查对应小节。**

---

[![在 Colab 中打开 · 附录A 第一部分](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rasbt/LLMs-from-scratch/blob/main/appendix-A/01_main-chapter-code/code-part1.ipynb) [![第二部分](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rasbt/LLMs-from-scratch/blob/main/appendix-A/01_main-chapter-code/code-part2.ipynb)

> 配套可运行 notebook（rasbt/LLMs-from-scratch 附录 A）。**边读边跑。**

## 0. 三件套和一条主线

概念课第 1 天给过三个数学直觉：**向量=空间里的点、矩阵乘=加权求和、梯度=往哪挪更对**。这一章就是把这三句话，落到能跑的 PyTorch 代码上：

| 数学直觉 | PyTorch 里是什么 | 在哪一节 |
|---|---|---|
| 一串数 / 一张表 | **张量 tensor** | §1 |
| 往哪挪更对（梯度）| **autograd**（`.backward()`）| §2 |
| 加权求和的网络 | **nn.Module / nn.Linear** | §3 |
| 滚下山（训练）| **训练循环**（前向→损失→反向→更新）| §5 |

> **环境**：零基础就用 **Google Colab**（浏览器打开 colab.research.google.com，新建笔记本即可，自带 PyTorch）。本机也行：`pip install torch`。检查：`import torch; print(torch.__version__)`。本书示例基于 PyTorch 2.x，版本差一点点不影响理解。

---

## 1. 张量：装数字的多维表格（书 A.2）

### 1.1 创建与"阶"

```python
import torch
tensor0d = torch.tensor(1)                       # 0 阶：标量（一个数）
tensor1d = torch.tensor([1, 2, 3])               # 1 阶：向量（一串数）
tensor2d = torch.tensor([[1, 2], [3, 4]])        # 2 阶：矩阵（一张表）
tensor3d = torch.tensor([[[1, 2], [3, 4]],
                         [[5, 6], [7, 8]]])      # 3 阶：表叠起来
```

**逐行拆**：张量就是"能装任意维数字的容器"。**阶（rank）**＝维数：标量 0 阶、向量 1 阶、矩阵 2 阶，再高就直接叫 3D、4D 张量。LLM 里的数据基本都是 3D（`[批, 词数, 维度]`）——你在第 3 章已经见过 `[2, 6, 3]` 这种形状了。

### 1.2 数据类型（dtype）

```python
print(torch.tensor([1, 2, 3]).dtype)        # torch.int64   （整数默认 64 位）
print(torch.tensor([1.0, 2.0]).dtype)       # torch.float32 （小数默认 32 位）
floatvec = torch.tensor([1, 2, 3]).to(torch.float32)   # 用 .to() 改类型
```

**逐行拆**：整数默认 `int64`，小数默认 `float32`。深度学习几乎全用 `float32`——精度够、又比 64 位省内存、还对 GPU 友好。`.to(...)` 既能改类型，后面（§8）也用它把张量搬到 GPU。

### 1.3 形状、变形、转置、矩阵乘

```python
t = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(t.shape)            # torch.Size([2, 3])  两行三列
print(t.view(3, 2))       # 变形成 3×2（不改数据，只换怎么切）
print(t.T)                # 转置：行列对调 -> 3×2
print(t @ t.T)            # 矩阵乘：[2,3]@[3,2] -> [2,2]
```

**逐行拆**：`.shape` 是你最该常按的"体检键"。`.view(...)` 换"怎么划分"、`.T` 换"行列方向"、`@`（或 `.matmul`）做矩阵乘——**这三样你在第 3 章注意力里已经反复用过了**，这里只是把它们的"出身"讲清楚。

<a name="g-tensor-vs-list"></a>
> ### 🐍 加油站 — 张量 vs Python 列表
> 都能装一串数，区别在：① 张量强制**同一类型、规则形状**（不能一行 2 个一行 3 个）；② 张量能整体做数学运算（`a + b`、`a @ b`）而不用写循环；③ 张量能搬上 **GPU** 并行算、能被 **autograd** 自动求梯度。这三点正是深度学习需要的，所以全程用张量、几乎不用裸列表。

---

## 2. autograd：自动求梯度的引擎（书 A.3–A.4）

概念课说"训练＝滚下山，顺着梯度往损失更小的方向挪"。**梯度谁来算？autograd。** 你只要搭好计算、调一句 `.backward()`，PyTorch 自动把每个参数的梯度算好。

```python
import torch.nn.functional as F
y  = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2], requires_grad=True)   # 关注它的梯度
b  = torch.tensor([0.0], requires_grad=True)

z = x1 * w1 + b          # 一次"加权求和"
a = torch.sigmoid(z)     # 压到 0~1
loss = F.binary_cross_entropy(a, y)   # 衡量预测和真值差多少

loss.backward()          # 自动反向传播
print(w1.grad)           # tensor([-0.0898])  损失对 w1 的梯度
print(b.grad)            # tensor([-0.0817])
```

**逐行拆**：
- `requires_grad=True`：告诉 PyTorch"这个参数我要算梯度"。一旦标了，凡是用它算出来的东西，PyTorch 都在背后悄悄记一张**计算图**（谁是谁算出来的）。
- `loss.backward()`：顺着计算图从 `loss` 倒推回去，把每个 `requires_grad=True` 参数的梯度算出来，存进它的 `.grad`。
- **你不用手算任何导数**——这正是 autograd 的全部意义。梯度 `w1.grad` 就是"w1 往哪个方向、挪多少，能让 loss 下降最快"。

> **微积分不会也没关系**：你只需要知道"`.backward()` 把梯度算好了"。第 1 天的"滚下山"直觉，对应的就是这一步。

---

## 3. 搭一个网络：nn.Module（书 A.5）

PyTorch 里搭网络的套路永远是：**继承 `nn.Module` → `__init__` 里备零件 → `forward` 里定义数据怎么流过零件**。（你在第 3 章写 `SelfAttention_v2` 时已经用过这个套路了。）

```python
class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 30),   # 第 1 隐藏层
            torch.nn.ReLU(),                   # 非线性
            torch.nn.Linear(30, 20),           # 第 2 隐藏层
            torch.nn.ReLU(),
            torch.nn.Linear(20, num_outputs),  # 输出层
        )
    def forward(self, x):
        return self.layers(x)                  # 数据顺着层流过去

model = NeuralNetwork(50, 3)
print(model)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("可训练参数总数:", num_params)           # 2213
```

**逐行拆**：
- `torch.nn.Linear(a, b)`：一个全连接层，做的就是 `x @ W + 偏置`——又是"加权求和"。`W` 和偏置是它**自动备好的可训练参数**。
- `torch.nn.Sequential(...)`：把几层按顺序串起来，`forward` 里一句 `self.layers(x)` 就让数据依次流过，不用一层层手写。
- `ReLU`：非线性激活。**没有它，再多层叠起来等价于一层**（概念课第 4 天讲过这个道理，那里用的是 GELU，同理）。
- `sum(p.numel() ...)`：把每个参数张量的元素个数加起来＝"参数量"。概念课第 4 天那个"参数就是一大堆可调数字"，这行代码就能数出来。

> **推理时关掉梯度**：只做预测、不训练时，用 `with torch.no_grad():` 包起来，省内存、提速：
> ```python
> with torch.no_grad():
>     out = model(torch.rand(1, 50))
> probas = torch.softmax(out, dim=1)   # 想要"概率"再显式 softmax
> ```
> PyTorch 习惯：**模型最后一层输出原始分数（logits），不自带 softmax**，因为损失函数会更高效地代劳（见 §5）。

---

## 4. 喂数据：Dataset 与 DataLoader（书 A.6）

模型要一批一批（batch）吃数据。PyTorch 的标准做法：写一个 `Dataset` 说"单条数据怎么取"，再用 `DataLoader` 自动打包成批、洗牌。

```python
from torch.utils.data import Dataset, DataLoader

X_train = torch.tensor([[-1.2, 3.1], [-0.9, 2.9], [-0.5, 2.6],
                        [2.3, -1.1], [2.7, -1.5]])
y_train = torch.tensor([0, 0, 0, 1, 1])

class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y
    def __getitem__(self, index):           # 取第 index 条
        return self.features[index], self.labels[index]
    def __len__(self):                       # 一共多少条
        return self.labels.shape[0]

train_ds = ToyDataset(X_train, y_train)
torch.manual_seed(123)
train_loader = DataLoader(dataset=train_ds, batch_size=2,
                          shuffle=True, drop_last=True, num_workers=0)

for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx+1}:", x, y)
```

**逐行拆**：
- `Dataset` 三件套：`__init__`（备数据）、`__getitem__`（取一条）、`__len__`（共几条）。**第 2 章的 `GPTDatasetV1` 就是照这个模板写的**，你会发现它一模一样。
- `DataLoader` 参数：`batch_size=2` 每批 2 条；`shuffle=True` 每轮打乱（防止模型记住顺序）；`drop_last=True` 丢掉凑不满一批的尾巴（否则最后一小批会干扰训练稳定）；`num_workers=0` 用主进程加载（小数据/Jupyter 里设 0 最稳，大数据可设 4 并行加速）。
- 遍历一遍 `train_loader` ＝ 训练一个 **epoch**（把数据完整过一遍）。

---

## 5. 训练循环：全书的"母版"（书 A.7）

**这是本章最重要的一段。** 第 5 章训练 LLM、第 6/7 章微调，训练循环都和这里**结构一模一样**，只是模型更大、数据是文本。

```python
import torch.nn.functional as F
torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)   # 优化器，学习率 0.5
num_epochs = 3

for epoch in range(num_epochs):
    model.train()                                  # 进入训练模式
    for features, labels in train_loader:
        logits = model(features)                   # ① 前向：算预测
        loss = F.cross_entropy(logits, labels)     # ② 算损失：差多少
        optimizer.zero_grad()                      # ③ 清空上一轮的梯度
        loss.backward()                            # ④ 反向：算梯度
        optimizer.step()                           # ⑤ 更新：挪一小步
    model.eval()                                   # 进入评估模式
```

**逐行拆（把这五步背下来，后面全是它）**：
1. **前向** `model(features)`：数据流过网络，得到预测分数 logits。
2. **算损失** `F.cross_entropy(logits, labels)`：交叉熵，衡量"预测"和"真值"差多少。注意直接喂 logits——它内部自带 softmax，更稳更快。（概念课第 5 天说的"交叉熵损失"就是它。）
3. **清梯度** `optimizer.zero_grad()`：PyTorch 的梯度是**累加**的，不清零会把上一轮的也算进来。**新手最常忘这行**，忘了训练就崩。
4. **反向** `loss.backward()`：autograd 把所有参数的梯度算好（§2 那一步，只是现在是整个网络）。
5. **更新** `optimizer.step()`：优化器按梯度把每个参数挪一小步（SGD：参数 −= 学习率 × 梯度）。这就是"滚下山"挪的那一步。

> 跑完会看到损失从 0.75 一路掉到 0.00——模型在这个玩具集上**收敛**了。这正是概念课第 5 天"损失滚下山"的真实样子。
>
> **`model.train()` / `model.eval()` 是干嘛的？** 切换训练/评估模式。有些零件（如 **dropout**——你在第 3 章见过——和 BatchNorm）在训练时和推理时行为不同。这个玩具网络没这些零件，加不加无所谓，但**养成习惯总写上**，换大模型时就不会踩坑。

---

## 6. 评估与预测：argmax 和准确率（书 A.7 / A.10）

```python
model.eval()
with torch.no_grad():
    logits = model(X_train)
predictions = torch.argmax(logits, dim=1)   # 每行取最大分数的下标 = 预测类别
print(predictions)                           # tensor([0, 0, 0, 1, 1])
print((predictions == y_train).sum())        # 数对了几个

def compute_accuracy(model, dataloader):
    model = model.eval()
    correct, total = 0.0, 0
    for features, labels in dataloader:
        with torch.no_grad():
            logits = model(features)
        preds = torch.argmax(logits, dim=1)
        correct += (labels == preds).sum()
        total += len(labels)
    return (correct / total).item()
```

**逐行拆**：`argmax(logits, dim=1)` 沿"类别"那一维取最大分数的下标，就是预测的类别——**不必先 softmax**，因为最大分数的位置不变。`compute_accuracy` 遍历整个 loader 数对的比例，返回准确率。这个"前向→argmax→比对"的套路，第 6 章判断垃圾邮件时会原样用上。

---

## 7. 存取模型：state_dict（书 A.8）

```python
torch.save(model.state_dict(), "model.pth")     # 存：只存参数字典

model = NeuralNetwork(2, 2)                       # 取：先建同样结构的空壳
model.load_state_dict(torch.load("model.pth"))   # 再灌入参数
```

**逐行拆**：`state_dict()` 是"层→参数"的字典，**只存参数、不存结构**。所以加载时要先 `NeuralNetwork(2, 2)` 建一个**结构完全一致**的空模型，再把参数灌进去。第 5 天"直接下载 GPT-2 的权重来用"，技术上就是这个 `load_state_dict`——别人训好的参数，你建好同样结构就能装上跑。

---

## 8. 用 GPU 加速（书 A.9）

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Mac（M 系列芯片）改用：
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = model.to(device)                          # 模型搬上去
for features, labels in train_loader:
    features, labels = features.to(device), labels.to(device)  # 数据也搬
    ...
```

**逐行拆**：PyTorch 里"设备"指算在哪、数据放哪。把**模型和数据搬到同一设备**（`.to(device)`），其余照旧。**最常见的报错**："Expected all tensors to be on the same device"——就是模型在 GPU、数据还在 CPU（或反过来），把两边都 `.to(device)` 就好。前几章在 Colab/CPU 上跑都够用；GPU 的威力要到训练大模型时才显出来。

> 第 3 章那句 `register_buffer` 的好处现在能完全理解了：它让掩码这种"非参数张量"也跟着 `model.to(device)` 自动搬，省得你手动 `.to`，避免上面这个设备不匹配的坑。

---

## 9. 练习

> 📓 **对照官方答案**：卡住了别硬磕——[在 Colab 打开本章练习解答 notebook ↗](https://colab.research.google.com/github/rasbt/LLMs-from-scratch/blob/main/appendix-A/01_main-chapter-code/exercise-solutions.ipynb)

### 书上原题
- **练习 A.1 / A.2**：在你的环境（Colab 最省事）装好 PyTorch，跑通官方校验代码，确认 `torch.__version__` 能打印。
- **练习 A.3**：数一数 §5 那个 `NeuralNetwork(2, 2)` 有多少参数。
  <details><summary>思路</summary>用 §3 的 `sum(p.numel() for p in model.parameters() if p.requires_grad)` 直接打印。手算：输入2→30（2×30+30）、30→20（30×20+20）、20→2（20×2+2），加起来。代码跑一遍对答案最快。</details>
- **练习 A.4**：比较 CPU vs GPU 上矩阵乘的耗时。Jupyter 里对大矩阵 `a`、`b` 用 `%timeit a @ b`，看多大尺寸时 GPU 开始更快。

### 本册自测题
1. 把 §5 训练循环里的 `optimizer.zero_grad()` **删掉**，重跑，观察损失还降不降、变得多怪。（体会"为什么必须清梯度"。）
2. 把学习率 `lr=0.5` 改成 `lr=0.001`，3 个 epoch 还能收敛到 0 吗？再改成 `lr=5.0` 呢？（体会"学习率太小学不动、太大会震荡"。）
3. 把 §3 网络里两个 `ReLU` 都删掉，参数量变了吗？这个"纯线性"网络表达能力会怎样？（联系"没有非线性，多层=一层"。）

---

## 10. 本章小结

1. **张量**是装数字的多维表格；`.shape` / `.view` / `.T` / `@` 是你天天用的四个动作。
2. **autograd**＝调一句 `.backward()` 就自动算好所有梯度，你永远不用手算导数。
3. 搭网络永远三步：**继承 `nn.Module` → `__init__` 备零件 → `forward` 定数据流**。
4. **训练循环五步**：前向 → 算损失 → 清梯度 → 反向 → 更新。**这是全书所有训练的母版**，记死它。
5. `Dataset/DataLoader` 喂批数据、`state_dict` 存取参数、`.to(device)` 上 GPU——这些"周边"后面每章都会复用。

---

## 11. 能改自检清单（全勾＝过关）

- [ ] 我能默写出训练循环的五步，并说出每步在干什么。
- [ ] 我能解释 `optimizer.zero_grad()` 为什么不能省。
- [ ] 给我一个 `nn.Module` 子类，我能指出哪是"备零件"、哪是"定数据流"。
- [ ] 我能用 `.shape` 调试任意张量，并解释 `.view`/`.T`/`@` 各自做了什么。
- [ ] 我能说清 `model.train()/eval()`、`torch.no_grad()`、`.to(device)` 各在什么时候用。

---

## 12. 通往下一章

工具摸熟了。**第 1 章**会拉高一层，给你一张"我们到底要造什么、分几步造"的代码地图——把你刚学的这些零件，安放进"造一个 GPT"的整体蓝图里。

> 带走主线：所有这些工具，最终都服务于一件事——训练一个**预测下一个词**的模型。
