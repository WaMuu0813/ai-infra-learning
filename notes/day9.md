# Day 9｜PyTorch Autograd、Python 装饰器与 C++ 并发

## 当日学习内容

### PyTorch：从 Tensor 到自动求导

- 计算图、forward、backward、gradient，以及 `requires_grad` 的传播。
- leaf / non-leaf Tensor 与 `grad_fn`：通过运算来源理解反向传播，不必背具体节点名称。
- 链式法则、分支计算图中梯度贡献的相加。
- 梯度默认累积、`w.grad.zero_()` 清零，以及 API 名称末尾 `_` 通常表示原地操作。

### Python：Decorator、Generator 与闭包状态

- 装饰器的应用时机，以及装饰器函数与 wrapper 的执行区别。
- `yield`、`next()`、暂停与恢复、`StopIteration`。
- Iterable、Iterator、Generator 的关系，以及生成器耗尽后不能自动重启。
- 普通 wrapper 装饰生成器函数时，创建生成器与执行生成器函数体是两个阶段。
- 区分 wrapper 共享的闭包状态与各生成器对象独立的执行状态。

### C++：线程、同步与阻塞队列

- 结合已有 OS 基础复习 process / thread：进程可理解为资源容器，线程可理解为执行流。
- 同一进程内线程共享地址空间、heap、global data；各自拥有调用栈和寄存器上下文等执行状态。
- `std::thread`、线程入口、执行顺序、`join()`。
- Data race、未定义行为与 lost update 的交错模型。
- `std::mutex`、`std::lock_guard`、临界区与锁粒度。
- `std::condition_variable`、`std::unique_lock`、带谓词的 `wait()`、`notify_one()`。
- 用 `std::queue<std::shared_ptr<Request>>`、互斥锁和条件变量组合 Blocking Queue，串联所有权、move、RAII 与 Request Manager。
- `std::ref(queue)`：在线程参数传递中表达引用传递。

## 典型代码 / 实验

### 1. Autograd：沿计算图求导

```python
w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)

a = w * x
b = a + 2
loss = b ** 2
loss.backward()
```

| 观察项 | 结果或含义 |
|---|---|
| `w.is_leaf` | `True` |
| `a.is_leaf` | `False`，由运算产生 |
| `a.grad_fn` / `b.grad_fn` / `loss.grad_fn` | 观察到类似 `MulBackward0`、`AddBackward0`、`PowBackward0` 的节点 |
| 前向计算 | `a = 6`、`b = 8`、`loss = 64` |
| 链式法则 | `dL/dw = dL/db × db/da × da/dw = 16 × 1 × 3 = 48` |
| `w.grad` | 实际得到 `48` |

当天还观察到第二次反向传播后梯度从 `48` 变为 `96`，验证了**梯度默认累积**。需要清零时使用：

```python
w.grad.zero_()
```

复现时注意：再次反向传播需要重新前向构图，或在前一次反向传播时按需保留计算图；不能直接将上述同一个 `loss.backward()` 无条件重复执行。原记录未注明当天采用哪一种方式。

分支计算图：

```python
a = w ** 2
b = 3 * w
loss = a + b
```

数学导数为 `dL/dw = 2w + 3`。同一个变量沿多条路径影响 loss 时，各路径的梯度贡献相加；这与多次 backward 对 `.grad` 的累积要分开理解。

### 2. Decorator：定义时应用，调用时执行 wrapper

```python
@deco
def f():
    ...
```

可理解为先定义函数，再执行 `f = deco(f)`。

当天输出题的关键顺序：

```text
定义原函数 gen
→ 应用 deco(gen)，依次输出 A、D
→ 返回 wrapper，名字 gen 指向 wrapper
→ 程序继续执行 print("H")
```

因此开头应为 `A → D → H`；wrapper 的内容要到之后调用被装饰函数时才执行。

### 3. Generator：创建、执行与耗尽

| 操作或对象 | 行为 |
|---|---|
| `list` | 可迭代对象（Iterable） |
| `iter(list对象)` | 得到迭代器（Iterator） |
| 调用未装饰的生成器函数 | 得到生成器对象，此时不执行函数体 |
| Generator | 也是 Iterator |
| `next(g)` | 开始或恢复执行，运行至下一次 `yield` 或结束 |
| `yield` | 产出一个值并暂停，保留执行状态 |
| 生成器结束 | 迭代时通过 `StopIteration` 表示结束 |
| 再次遍历已耗尽的 `g` | 不会从头开始；需重新创建生成器 |

`for` 可粗略理解为：先 `iter()`，不断 `next()`，遇到 `StopIteration` 结束。

### 4. Decorator × Generator：日志结束不等于消费结束

对当天使用的普通 wrapper：

```text
g = generate()
→ 进入 wrapper，输出 [start]
→ wrapper 调用原生成器函数，得到 generator object
→ wrapper 输出 [end]，返回该对象

next(g)
→ 才开始执行原生成器函数体，运行至 yield
```

这里 `[end]` 只表示 wrapper 创建并返回生成器的过程结束，**不表示生成器已消费完**。这一顺序针对当天的普通 wrapper，不能套用于所有装饰器实现。

### 5. Closure × 两个 Generator：共享计数，独立暂停

```python
g1 = gen(10)
g2 = gen(20)
```

当天题目中，两个调用使用同一个 wrapper，其闭包内维护同一个 `count`：

- 两次调用依次输出 `call: 1`、`call: 2`。
- `g1`、`g2` 是两个生成器对象，各自保存局部变量 `x` 和暂停位置。
- 按题目的交错调用顺序，产出为 `10 → 11 → 20 → 12 → 21`；这一部分当天推导正确。
- `next(g1)` 只恢复 `g1`，不会重新调用 wrapper，因此不增加闭包中的调用计数。

### 6. Thread 与 join：区分两个执行流

```cpp
std::thread t1(worker);
std::thread t2(worker);
t1.join();
t2.join();
```

- 创建线程前的 `main start` 先执行；两个 worker 谁先输出不确定。
- 两次 `join()` 都返回后，才执行后面的 `main end`。
- 新线程从 `worker()` 开始执行；入口函数返回，线程结束。
- `std::thread` 不会自动反复调用 worker。持续处理任务来自自己编写的循环。
- worker 在条件变量上等待时，函数尚未返回，但线程可以阻塞等待，不必持续占用 CPU 执行。

| 阶段 | main 线程 | t1 线程 |
|---|---|---|
| 创建线程后 | 继续执行 `main()` | 从 `worker()` 入口开始执行 |
| main 调用 `t1.join()` | 若 t1 尚未结束，则等待 | 继续执行自己的 worker |
| worker 返回 | 等待目标线程结束 | 线程结束 |
| join 返回 | 继续执行 main 中 join 后面的代码 | 已结束，不会收到新任务 |

### 7. Data Race：实验现象与语义边界

两个线程各执行一百万次 `counter++`，数学预期为 `2000000`。未加同步时，当天观察到 `1264047`、`1310403`、`1314095` 等不同结果。

用于解释 lost update 的明确交错：

```text
初始 counter = 5

t1 读取 5
t2 读取 5
t1 写入 6
t2 写入 6
```

**该交错模型的最终值是 6。** 但真实 C++ 程序中，对普通共享变量的这种未同步读写构成 data race，属于未定义行为，不能保证它只是“少加几次”。

### 8. Mutex：正确性与锁粒度

| 加锁方式 | 特点 |
|---|---|
| 整个循环持锁 | 能保护计数；两个线程的这段工作基本串行执行 |
| 每次 `counter++` 单独持锁 | 同样能保护计数，但频繁加锁、解锁有开销，不保证更快 |

当天正确判断：**锁粒度越小不等于性能越好**。应综合临界区大小、锁竞争、同步次数与实际 workload，通过 benchmark 判断。

### 9. Condition Variable 与 Blocking Queue

核心接口：

```cpp
cv.wait(lock, predicate);
cv.notify_one();
```

带谓词的等待要按以下过程理解：

```text
持锁检查条件
→ 不满足则释放锁并等待
→ 被唤醒后重新获得 mutex
→ 再次检查条件
→ 条件满足才继续，否则继续等待
```

- `notify_one()` 不代表等待线程立刻开始处理任务；它仍需获得锁并检查条件。
- 即使醒来，也可能因其他 worker 先取走任务而发现队列为空；还要考虑 spurious wakeup（虚假唤醒）。
- 因而应理解为 `while (条件不满足) wait`，不能只检查一次 `if`。

阻塞队列的处理流程：

```text
producer 入队 Request
→ notify
→ worker 被唤醒、重新获得锁并检查队列
→ 取出 Request
→ process
```

```cpp
queue_.push(std::move(request));
```

这里转移的是 `shared_ptr` 的所有权句柄，不是搬动 `Request` 对象本体。

## 我提出的问题

- **创建 thread 后，它会一直执行 worker 吗？** 新线程以 worker 为入口执行一次；是否持续工作取决于函数内部是否有循环。入口返回，线程结束；等待条件变量时可以阻塞。
- **能把 `join()` 看作线程的“断点”吗？join 后目标线程就终止了吗？** 不能。`join()` 不负责终止目标线程，而是让调用者等待目标线程结束。它返回时，目标线程已经结束。
- **worker 已经结束，为什么还说“当前线程继续执行 join 后的代码”？线程不是执行创建时的入口函数吗？** 这里继续执行的是调用 join 的 main 线程。t1 执行 worker，main 执行 main；两者各有自己的入口和执行流。

## 犯过的错误及纠正

1. **局部导数算错**：曾算错 `w²` 的导数，应为 `2w`。计算图的整体理解不能替代每一步局部导数的核对。
2. **装饰器执行时机混淆**：最初预测输出以 `H → A → B → C → D` 开始；应先在定义阶段应用装饰器，开头为 `A → D → H`，之后调用时才执行 wrapper。
3. **生成器创建与执行混淆**：普通 wrapper 的 `[start]`、`[end]` 可以在创建生成器对象时就输出；真正的生成器函数体要等迭代时执行。
4. **闭包状态与生成器状态混淆**：曾将两次调用预测为 `call: 1`、`call: 1`；同一个 wrapper 共享一份 count，应为 `call: 1`、`call: 2`。两个生成器的暂停位置和局部变量仍各自独立。
5. **交错模型与真实程序混淆**：对题目明确给出的“读 5、读 5、写 6、写 6”回答“不确定”；该模型中结果确定为 6。真实存在 data race 的 C++ 程序则是未定义行为，不能承诺输出。
6. **join 的调用者与等待目标混淆**：曾把 join 理解为目标线程的断点或终止动作。实际是 main 调用 `t1.join()`，main 等待，t1 执行至结束；随后 main 继续。

## 重难点与待巩固内容

- **Autograd**：手算链式法则与分支梯度，区分图内多路径相加和多次 backward 的梯度累积。
- **Python 执行时机**：按“定义并应用装饰器 → 调用 wrapper → 创建生成器 → next 恢复执行”逐步追踪。
- **状态归属**：一份 wrapper 闭包计数，与多个生成器各自的局部变量、暂停位置分开记录。
- **线程生命周期**：每个线程执行自己的入口函数；入口返回即结束，join 等待的是另一个线程。
- **并发正确性**：data race 是未定义行为；条件变量醒来后必须重新持锁、检查条件。
- **性能意识**：小锁粒度不保证更快，结合 workload 做 benchmark。
- **所有权衔接**：队列中 move 的是 `shared_ptr` 句柄；将线程同步与之前的 RAII、Request Manager 联系起来理解。

优先复习：**Decorator × Generator × Closure 的状态追踪，以及 Thread / Worker / join 的执行主体区分**。
