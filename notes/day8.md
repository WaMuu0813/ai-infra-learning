# Day 8｜C++ 抽象能力与 Python 闭包

## 当日学习内容

### C++：继承、多态与行为传递

- **继承与运行时多态**：静态类型、动态类型、`virtual`、`override`、纯虚函数、抽象类、虚析构函数和 object slicing（对象切片）。
- **多态的常见实现**：初步接触 `vtable` / `vptr`，知道它们是编译器实现运行时多态的常见机制，未深入 ABI。
- **模板基础**：通过 `template <typename T>` 将类型参数化，由编译器根据实际类型实例化对应代码；未进入模板元编程。
- **Lambda 与 STL**：Lambda 的基本形式，以及与 algorithm、iterator 的组合。
- **Callback（回调）**：函数指针、Lambda、`std::function`。核心思想是函数不仅能接收数据，也能接收行为；后续线程池 task、异步回调和 scheduler 都会用到。

### Python：对象行为、参数与作用域

- **继承**：`super()`、方法覆盖。
- **特殊方法（dunder method）**：`__init__`、`__call__`、`__len__`、`__getitem__`；理解常见对象操作如何对应到特殊方法。
- **参数与作用域**：keyword-only 参数、局部作用域（local）、`global`、`nonlocal`。
- **Closure（闭包）**：外层函数返回后，内层函数仍能访问所捕获的外层变量；函数本身也是对象，可以被返回、保存和调用。

## 典型代码 / 示例

### 1. 静态类型、动态类型与虚析构

```cpp
class Base {
public:
    virtual void run() = 0;
    virtual ~Base() = default;
};

class Derived : public Base {
public:
    void run() override {}
};

Base* p = new Derived();
p->run();
delete p;
```

- `p` 的静态类型是 `Base*`，实际对象的动态类型是 `Derived`。
- `run()` 是虚函数，因此这里调用 `Derived::run()`；`override` 用于让编译器检查是否正确覆盖了基类虚函数。
- 纯虚函数使这里的 `Base` 成为抽象类，不能直接实例化。
- 本例通过基类指针删除派生对象，基类需要虚析构函数；若没有，执行这种 `delete` 会产生未定义行为。
- **对象切片**：将派生对象按值复制到可实例化的基类对象时，派生部分不会保留。注意区分按值复制与通过基类指针、引用访问派生对象。

### 2. 类型参数化与行为作为参数

模板声明的基本形式：

```cpp
template <typename T>
```

Lambda 示例：

```cpp
[](int x) {
    return x * 2;
}
```

模板把“使用什么类型”交给实例化时确定；回调把“执行什么行为”交给调用方提供。函数指针、Lambda 和 `std::function` 是当天接触的相关工具。

### 3. Python 特殊方法与关键字专用参数

| 特殊方法 | 对应行为 |
|---|---|
| `__init__` | 初始化实例 |
| `__call__` | 让实例能够像函数一样被调用：`obj()` |
| `__len__` | 响应 `len(obj)` |
| `__getitem__` | 响应 `obj[key]` 等索引操作 |

```python
def f(x, *, name):
    ...
```

`name` 是 keyword-only 参数，必须以关键字传入，例如 `f(1, name="demo")`。

### 4. 闭包与 nonlocal

```python
def outer():
    counter = 0

    def inner():
        nonlocal counter
        counter += 1
        return counter

    return inner
```

- `outer()` 返回的是函数对象 `inner`，没有在返回时调用它。
- 返回的 `inner` 保留了对外层变量 `counter` 的访问，因此外层函数返回后，计数状态仍可继续使用。
- `nonlocal counter` 让赋值作用于外层函数的变量；`global` 则用于指向模块级变量。

## 我提出的问题

- **`return counter` 和 `return counter()` 有什么区别？** 前者返回 `counter` 当前指向的对象或值；后者先把 `counter` 当作可调用对象调用，再返回调用结果。是否能加括号，要看它指向的对象是否可调用。

## 犯过的错误及纠正

- **曾混淆“返回对象”和“调用对象后返回结果”**：在闭包示例中，`counter` 是整数，应该写 `return counter`；写成 `return counter()` 会尝试调用整数并引发 `TypeError`。
- **对应到函数对象**：`return inner` 返回函数本身，`return inner()` 则立即执行函数并返回结果。后续理解 decorator 时，需要继续区分“函数对象”和“函数调用”。

## 重难点与待巩固内容

1. **静态类型 vs 动态类型**：通过基类指针调用虚函数时，能判断实际调用哪个实现。
2. **`virtual` / `override` / 虚析构**：分别理解动态分派、覆盖检查和多态删除。
3. **Object slicing**：区分按值复制与通过指针、引用使用多态。
4. **模板**：抓住“类型参数化”，暂不深入模板元编程。
5. **Lambda / Callback**：抓住“行为作为参数”，理解与 STL 算法及后续任务调度的联系。
6. **Closure**：解释外层函数返回后，内层函数为什么仍能访问外层变量；区分返回函数和调用函数。

优先复习：**多态、Callback、Closure**。

## 尚未完成的内容

- **编译 → 链接 → symbol → 静态库 / 动态库**：当天仅提及，因 C++ 内容较多而主动暂停，后续仍需补课。
- **Decorator**：这里只建立函数对象与闭包的前置认识，留待 Day9，不计为当天已学内容。
