# Day7 — Ownership、Tensor 与构建复盘
日期：2026-09-13

## 当天学习内容
复盘 shared_ptr / weak_ptr / lock、生命周期、容器指针失效、拷贝构造与赋值、资源泄漏和重复释放；初步接触 RVO / NRVO；Tensor 规约维度的语义；CMake 依赖。范围截止原请求按天整理日志时，不混入其后新增的 stride / view / reshape 课程。

## 典型代码 / 实验
引用计数题：
```cpp
auto p1 = std::make_shared<Request>(101);
queue.push_back(p1);
std::weak_ptr<Request> observer = p1;
auto p2 = observer.lock();
```
在没有其他持有者且 lock 成功时，同一控制块有 p1、queue 中元素和 p2 三个强引用；observer 仍然存在。

危险 Debug 题（仅分析，不运行）：
```cpp
auto p = std::make_shared<Request>(101);
Request* raw = p.get();
std::shared_ptr<Request> p2(raw); // 错误：建立独立控制块
```
实际 Request 只有 1 个、控制块有 2 个。该操作破坏所有权并导致未定义行为；make_shared 创建的对象不能由这个独立默认删除器安全释放，不能把后续析构描述为保证发生的正常执行过程。正确共享方式是 auto p2 = p。

Tensor 题：
```python
# x.shape == (8, 128, 768)，依次为 batch、sequence、hidden
mean = x.mean(dim=1, keepdim=True)       # (8,1,768)
var = ((x - mean) ** 2).mean(dim=-1, keepdim=True) # (8,128,1)
y = (x - mean) / torch.sqrt(var + 1e-5) # (8,128,768)
```
可广播不代表目标计算正确。对每个 token 的 hidden 值标准化时，均值和平方偏差均应围绕 dim=-1 计算。该题在原聊天明确为纯分析。

## 用户提出过的关键问题
- p.get() 是什么？取得原始指针，不新建 Request，也不增加强引用数。
- lock 是否转换或释放 weak_ptr？不会；成功时返回一个新的强引用，weak_ptr 本身保留。
- 复制 shared_ptr 是否创建新控制块？不会，共享原控制块。

## 犯过的错误及纠正
- 曾回答危险题有 3 个 Request、2 个控制块：控制块数量对，Request 数量应为 1；指针变量不等于被指向对象。
- 在 (32,128) 对 dim=0 求均值且 keepdim=True 的题中，把结果写成 (128,)；应为 (1,128)。
- 已找出均值与平方偏差规约方向混用的问题，但需把“跨 sequence”说清：固定 batch 和 hidden，跨 128 个 token。
- shared_ptr 复制增加同一控制块强引用数，不新建控制块。
- vector 重新分配后的旧指针是悬空指针，不保证变空或崩溃。

## 当前掌握情况
原始回答证明：能正确推导 (8,1,768)、(8,128,1)、(8,128,768)，并能解释 softmax 在 dim=-1 与 dim=1 上分别跨 hidden 与 sequence 归一化。Softmax 当时仅作维度语义讨论，不记为正式系统学习完成。
已能识别核心所有权风险，仍需巩固对象数、指针数与控制块数的区分。RVO / NRVO 只属初步认识。
本次截图确认 cpp/day7_debug_review.cpp 未跟踪，但未读取其内容、未编译运行；不据文件存在断言实验完成。

## Day7 补充：Tensor 内存布局与 GDB（2026-09-13）

本节接续当天前半段的所有权、Tensor 规约维度和 CMake 复盘。原日志中的“范围截止原请求”仅描述前半段记录；本节补齐之后实际开展的学习。以下实验结果按当天对话记录整理，本次日志编辑没有重新运行 PyTorch 或 GDB。

### 当天学习内容与典型实验

以 `x = torch.arange(24).reshape(4, 6)` 为例：storage 是底层存储；shape 描述各维长度；stride 表示某一维索引增加 1 时跨过的元素数；storage_offset 是首个逻辑元素相对 storage 起点的元素偏移。二维索引位置为 `storage_offset + i * stride[0] + j * stride[1]`，换成字节地址还需乘以 `element_size()`。

```python
import torch

x = torch.arange(24).reshape(4, 6)
a = x.T
s = x[1:3, 1::2]
c = a.contiguous()
r = a.reshape(24)

for name, t in [("x", x), ("a", a), ("s", s), ("c", c), ("r", r)]:
    print(name, t.shape, t.stride(), t.storage_offset(),
          t.is_contiguous(), t.data_ptr(), t.untyped_storage().data_ptr())
```

| Tensor | shape | stride | storage_offset | 默认布局连续性 | 与 x 的存储关系 |
|---|---|---|---|---|---|
| x | (4, 6) | (6, 1) | 0 | 连续 | 基准 |
| a = x.T | (6, 4) | (1, 6) | 0 | 不连续 | 共享 storage，首元素地址相同 |
| s = x[1:3, 1::2] | (2, 3) | (6, 2) | 7 | 不连续 | 共享 storage，但首元素地址不同 |
| c = a.contiguous() | (6, 4) | (4, 1) | 0 | 连续 | 复制到新 storage |
| r = a.reshape(24) | (24,) | (1,) | 0 | 连续 | 此例需要复制 |

slice 的值为 `[[7, 9, 11], [13, 15, 17]]`。沿行仍跨 6 个元素；列步长为原列 stride 1 乘切片步长 2；offset 为 `1*6 + 1*1 = 7`。

`data_ptr()` 指向 tensor 的首个逻辑元素，不能把它直接等同于 storage 起点。这里 s 与 x 的 data_ptr 不同，但 `untyped_storage().data_ptr()` 相同；地址差为 `7 * x.element_size()` 字节。因此，“data_ptr 不同”本身不能证明 storage 不共享。

`transpose(0, 1)` 与本例二维 `x.T` 交换两个维度的 shape 和 stride，不搬动原 storage 中的元素。`contiguous()` 按当前逻辑顺序生成连续布局；对本例不连续的 a 需要复制，而对已经满足目标连续布局的 tensor 可以直接返回自身。

`view` 要求现有 stride 与目标形状兼容，不能兼容时会报错；`reshape` 可以在兼容时返回 view，不能兼容时通过复制完成重排。不能概括为“所有 non-contiguous tensor 的 reshape 都必须复制”。

本例 `a.view(24)` 不能成立；`a.reshape(24)` 的顺序为：

```text
0, 6, 12, 18, 1, 7, 13, 19, 2, 8, 14, 20,
3, 9, 15, 21, 4, 10, 16, 22, 5, 11, 17, 23
```

这不同于 x 的原 storage 顺序 `0, 1, 2, ..., 23`，也不能用一个一维固定 stride 表示，因此此次 reshape 不能共享原地址。相比之下，`x.reshape(24)` 可共享原 storage；作为复习推导，s 的逻辑偏移为 `7, 9, 11, 13, 15, 17`，可用 stride 2 表示，说明不连续布局也可能支持无复制 reshape。

### 今天实际犯过的错误与纠正

1. **x.T 的 stride**：曾答成 `(4, 1)`，正确是 `(1, 6)`。`(4, 1)` 是把转置结果另存为连续布局后的 stride，转置本身只交换维度信息。
2. **contiguous 后的地址**：曾认为 non-contiguous 的 a 调用 `contiguous()` 后 data_ptr 仍相同。本例必须分配新连续 storage 并复制，因此 c 与 a 的地址不同。
3. **transpose 后 reshape(24)**：低估了转置后的逻辑遍历顺序与原 storage 顺序的差异，误判可以共享原地址。纠正方式是列出前几个值 `0, 6, 12, 18, 1, ...`，再判断能否用固定 stride 表示；本例需要复制。
4. **slice 的 stride**：对 `x[1:3, 1::2]` 曾答成 `(3, 2)` / `(6, 1)`，正确是 `(6, 2)`。shape 变成 `(2, 3)` 不会把行 stride 自动压成 3，列步长 2 也不能忽略。
5. **GDB finish 后打印 result**：`finish` 显示返回 100，立即 `print result` 却得到垃圾值（当天记录为 `-1550054406`）。当时调用行 `int result = sum_vector(data);` 的初始化尚未完成；执行 `next` 后再打印，result 才是 100。这是当天停点的实际现象，不表示每次 finish 都必然停在赋值之前。

### GDB 入门实操

使用 `-g` 编译，加入供 GDB 使用的调试信息。以下为复习命令示例：

```bash
g++ -g cpp/day7_gdb.cpp -o day7_gdb
gdb ./day7_gdb
```

| 命令 | 当天学习的用途 |
|---|---|
| break / b | 在函数或源码行设置断点，例如 `break sum_vector` |
| run / r | 启动程序，运行到断点 |
| next / n | 执行当前源码行，通常不进入被调用函数 |
| step / s | 单步执行，可进入有调试信息的被调用函数 |
| print / p | 查看变量或表达式，例如 `print nums`、`print result` |
| continue / c | 继续运行至下一断点或结束 |
| finish | 执行完当前函数，返回调用者，并可显示返回值 |
| info locals | 查看当前栈帧的局部变量 |
| bt | 查看调用栈 |

当天在 `sum_vector(data)` 中观察参数 `{10, 20, 30, 40}`、返回值 100，并通过 `finish → print result → next → print result` 区分“函数已返回”和“调用者初始化已完成”。在函数内部使用 bt，看到 `#0 sum_vector(...)`、`#1 main()`，理解当前函数及其调用来源。源码一行可能对应多个执行步骤，调试时要结合停点判断变量是否已初始化。

### 用户提出的重要问题

- **“.T 能否视为 reshape？”** 不能。本例 .T 是交换维度的转置，改变逻辑索引与底层元素的对应关系；reshape 保持输入 tensor 的逻辑元素顺序再组织形状。例如 `x.T` 与 `x.reshape(6, 4)` 虽然 shape 相同，元素排列却不同。
- **“为什么有些 reshape 复制，有些不复制？”** 关键是目标形状是否与输入的 size/stride 兼容；能仅调整元数据就返回 view，否则需要复制。应结合逻辑顺序、stride 和 storage 关系判断。
- **“为什么计划没显式安排 C++ 继承/多态，是用不到吗？原本准备什么时候学？”** 原计划没有单独列明，属于安排表达上的疏漏；原意是在 CMake/GDB 工程化之后、进入并发之前补类设计、继承与多态。当天澄清安排，不等于已经完成课程。
- **“还有什么遗漏或隐含安排，包括 Python？”** 随后把后续内容显式列出，见下节。

### 后续安排（尚未完成，不计为已掌握）

- C++：继承与动态多态（virtual、override、纯虚函数、虚析构、对象切片、基类指针/引用）；模板基础；lambda 捕获、iterator/algorithm；const 深化；编译/链接、静态库/动态库；随后进入 thread、mutex、锁、condition_variable、atomic、生产者消费者和线程池。其他待补内容包括错误处理、内存布局、函数指针/回调及性能意识。
- Python：OOP 深化、继承和 super；decorator、generator、typing、dataclass、模块/package/import、常见魔术方法，以及函数参数和作用域的进一步巩固。
- PyTorch：在 Tensor 布局基础上继续 broadcast/reduction、matmul、autograd、计算图、forward/backward、nn.Module/Parameter，以及后续 Softmax/LayerNorm 等算子。当天对这些内容的计划讨论不能视为正式课程完成。
- CUDA/Triton 和两个月后的能力描述属于未来目标与估计，不记入今日已掌握能力。

### 当前掌握情况与待巩固点

今天已经练习 Tensor 布局推导和地址实验，跑通 GDB 基础调试流程；仍需巩固“shape 不决定 stride”“共享 storage 不等于首元素地址相同”“reshape 是否复制取决于布局兼容性”。上述五个错误作为后续复习检查点保留，不能仅因得到纠正就标记为熟练掌握。
