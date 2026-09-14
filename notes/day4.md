# Day4 — shared_ptr / weak_ptr 与 Python 工程化
日期：2026-09-10

## 当天学习内容
shared_ptr、make_shared、control block、strong count、use_count、get；weak_ptr、lock、expired、循环引用。Python 学习异常、文件读写、with open、context manager、venv 和 requirements.txt。

## 典型代码 / 实验
```cpp
auto p1 = std::make_shared<Request>(101);
auto p2 = p1;                       // 共用同一个控制块
std::weak_ptr<Request> observer = p1; // 不增加强引用数
auto p3 = observer.lock();          // 成功时增加一个强引用
```
Request 是课堂类型，片段需放入已有类型上下文。互相持有 shared_ptr 的对象可能形成循环所有权；将适当的观察方向改为 weak_ptr 可以打破该循环。

危险反例（不运行）：
```cpp
Request* raw = new Request(101);
std::shared_ptr<Request> p1(raw);
std::shared_ptr<Request> p2(raw); // 两个控制块管理同一对象，导致未定义行为
```

## 用户提出过的关键问题
- 复制 shared_ptr 和由 raw 构造 shared_ptr 有何区别？复制共享控制块；从裸指针构造会建立新的所有权管理。
- use_count 统计什么？共享该控制块的强引用数。
- weak_ptr 为什么不增加引用计数？它不增加强引用数，不拥有对象；它仍参与控制块的弱引用管理。
- lock 做什么？对象仍存在时取得 shared_ptr，否则得到空 shared_ptr。
- raw pointer 能类比符号链接吗？只能作有限直觉类比。
- get 和 release 是否学过？回顾确认 get；release 的掌握不据此推定，且 shared_ptr 没有 release 成员。

## 犯过的错误及纠正
不要把硬链接数、文件描述符数和 shared_ptr 引用数完全等同。多个 shared_ptr 变量也不必然意味着多个控制块；这点在 Day7 再次暴露为易混点。

## 当前掌握情况
已学习共享所有权和弱观察关系；后续复盘显示控制块数量仍需练习。Python 工程化内容来自聊天回顾，未复验具体文件。
