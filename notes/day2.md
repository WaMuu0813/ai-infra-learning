# Day2 — Ubuntu 与 Python 基础
日期：2026-09-08

## 当天学习内容
使用 Ubuntu 22.04、VS Code Remote SSH；回顾记录涉及 build-essential、git、cmake、vim、tree、openssh-server。复习 list、dict、循环、类、self、列表推导、*args / **kwargs、模块与 import、__name__。

## 典型代码 / 实验
```python
nums = [1, 2, 3]
squares = [x * x for x in nums]

def f(*args):
    print(args)

f(1, "hello", [1, 2])

class Student:
    def __init__(self, name):
        self.name = name

    def show(self):
        print(self.name)

if __name__ == "__main__":
    Student("demo").show()
```
以上按聊天用例整理，未在远端重新执行。直接运行模块时 __name__ 为 "__main__"；首次正常导入时模块顶层代码会执行，主入口保护中的代码不会因此执行。

## 用户提出过的关键问题
- self 必须是第一个参数吗？普通实例方法的第一个形参接收实例；self 是惯用名，可类比 C++ 隐式的 this。
- *args 只能传数字吗？不是，它接收多个位置参数，不限制为数字。
- if __name__ == "__main__" 是什么？用于区分直接运行和作为模块导入。

## 犯过的错误及纠正
聊天回顾记录了对 *args 类型限制的疑问；纠正为参数可以是字符串、列表等。导入模块不等于只加载函数定义，顶层语句也可能运行。

## 当前掌握情况
已复习基础语法和实例方法，开始理解模块执行行为。记录未包含本日独立综合测验，不将“讲过”视为完全熟练。
