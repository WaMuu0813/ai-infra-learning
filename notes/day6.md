# Day6 — Tensor 基础与 CMake 多文件工程
日期：2026-09-12

## 当天学习内容
NumPy shape、ndim、索引、切片、reshape、reduction、axis、keepdims、broadcasting、逐元素计算、矩阵乘法；均值、方差、标准化。使用 CPU PyTorch 学习 shape、dtype、device、numel、reshape、flatten。CMake configure / generate / build、target、增量构建和头文件依赖；多文件目录 cpp/day6_request_manager/（名称来自聊天回顾，远端未逐项检查）。

## 典型代码 / 实验
shape 练习：
- (2,3,4) 对 axis=1 求和：结果 (2,4)，keepdims=True 时 (2,1,4)。
- (2,3) 的 x[:,1] 是 (2,)，x[:,1:2] 是 (2,1)。
- (8,1,6,1) 与 (7,1,5) 广播为 (8,7,6,5)。

标准化复习片段：
```python
mean = x.mean(dim=-1, keepdim=True)
var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
y = (x - mean) / torch.sqrt(var + 1e-5)
```
x 为浮点 Tensor。这里采用总体方差定义；输出均值约为 0，方差为 var/(var+eps)，仅当 var 相比 eps 足够大时接近 1，常量输入输出方差为 0。

课堂构建流程：
```bash
mkdir build
cd build
cmake ..
cmake --build .
```
回顾记录了修改 request_manager.cpp 后主要重编译对应目标文件再链接的观察；本次未重新运行。

## 用户提出过的关键问题
- axis 如何理解？指定规约哪一个维度，比只记行列更适合高维数组。
- var、sqrt 是什么？先解释方差和平方根，再使用 API。
- CMake 是编译器吗？不是，它配置并生成构建信息，build 再调用实际构建工具。
- target 是什么？构建目标，例如一个可执行程序。
- 为什么改 cpp 后通常只需 build？已有配置和依赖信息可驱动增量构建。
- 头文件不列入 add_executable 为何能工作？源文件通过 include 使用它，构建工具记录头文件依赖。

## 犯过的错误及纠正
曾把标准化结果均值与方差说反：应为均值约 0、方差通常接近 1，且受 epsilon 影响。
(2,3) 不等于 (2,3,1)，长度为 1 的维度仍然存在。
曾认为重编译时先展开头文件才发现变化；实际构建工具先用已记录依赖判断目标过期，再调用编译器。
用户指出新 API 不应未经解释就放入练习；这属于教学方式问题，不归为用户错误。

## 当前掌握情况
已学习 shape 推导、广播和基本构建流程。聊天记录当时使用 CPU 环境；未把版本号当作当前环境验证结果。本日未正式学习 autograd、stride、CUDA、Triton。
