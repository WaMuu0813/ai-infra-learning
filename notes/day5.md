# Day5 — 容器、Request Manager 与 CLI
日期：2026-09-11

## 当天学习内容
vector 的 size、capacity、reserve、data、重新分配与失效规则；move / noexcept；unordered_map 的 hash、bucket、collision、load_factor、rehash、find、operator[]。设计 Request Manager，学习 argparse 和 logging。

## 典型代码 / 实验
聊天中的 vector 示例：
```cpp
std::vector<int> v;
v.reserve(2);
v.push_back(10);
v.push_back(20);
int* p = v.data();
v.push_back(30);
```
需观察实际 capacity：reserve(2) 保证至少容纳 2 个元素，不保证恰好为 2。如果第三次插入触发重新分配，p 就失效。为确定触发条件，可在复习实验中先填满实际 capacity，再追加一个元素；不解引用旧 p。

课堂设计：
```cpp
std::vector<std::shared_ptr<Request>> queue;
std::unordered_map<int, std::weak_ptr<Request>> request_table;
```
queue 持有资源；表用于观察和查找，不延长 Request 生命周期。lock 返回的临时强引用需纳入计数。

CLI 示例：
```python
parser.add_argument("--model", required=True)
parser.add_argument("--batch-size", type=int, default=1)
logging.basicConfig(level=logging.WARNING)
```
片段依赖 argparse parser 和 logging 导入；WARNING 阈值下 INFO 不输出，WARNING 输出。

## 用户提出过的关键问题
- key 要连续吗，1 和 666 可以同时作 key 吗？无需连续。
- rehash 后元素地址都失效吗？rehash 使迭代器失效，但不使元素的指针和引用失效；删除元素另当别论。
- iterator 如何访问 key / value？用 it->first / it->second。
- required、default 是什么？分别表示参数必须提供、缺省取值。
- --batch-size 为什么变成 args.batch_size？argparse 默认把选项名中的连字符转为下划线。

## 犯过的错误及纠正
曾把失效指针说成 null pointer，并认为解引用一定 crash。正确说法是 dangling pointer；解引用是未定义行为，不能依赖任何输出。
vector 增长不保证翻倍。hash(key) % bucket_count 只是桶分配的简化模型，不能当标准要求的实现公式。

## 当前掌握情况
已能将容器与所有权设计联系起来。需要继续巩固失效条件、强引用计数和查询操作副作用；unordered_map 的 operator[] 在 key 不存在时会插入元素。
