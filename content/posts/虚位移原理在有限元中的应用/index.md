---
title: 虚位移原理在有限元中的应用
date created: 星期二, 二月 17日 2026, 6:55:22 晚上
date modified: 星期二, 二月 17日 2026, 6:55:36 晚上
---
tags: #有限元 #虚位移原理
# 1.虚位移原理
## 1.1 什么是虚位移
虚位移通常使用 $\delta u$来进行表示

>[!info] **虚位移原理的基本特征**
> + 微小变动且为假想
> + 满足几何约束条件
> + 发生时不经过时间

## 2.1 虚位移原理的基本表述

**如果一个变形体处于静力平衡状态，那么对于任意满足位移边界条件的微小虚位移 $\delta u$，作用在物体上的真实外力所做的==“外虚功”==，必等于物体内部真实应力所做的==“内虚功”==。**
$$\delta u W_{ext}=\delta u W_{in t}$$
采用一个基本的线性杆虚位移变形过程

**外力功:**
$$\delta W_{ext} = \int_{0}^{L} b(x) \cdot \delta u(x) \, dx + P \cdot \delta u(L)$$

>其中$b(x)$为体力关于$x$的函数,P为合外力

**内力功:**

杆件内部的真实应力$\sigma$在虚应变$\epsilon$上做的功（本质为虚应变能）
真实应力 $\sigma=E\epsilon=E \frac{du}{dx}$ 虚应变 $\epsilon= \frac{d(\delta u)}{dx}=$ 于是有内虚功为:
$$\delta W_{int} = \int_{0}^{L} \left( E \frac{du}{dx} \right) \cdot \left( \frac{d(\delta u)}{dx} \right) \cdot A(x) \, dx$$

## 2.2 虚位移原理的平衡方程

由虚位移原理 $\delta W_{ext} = \delta W_{int}$，得到杆件的控制方程：

$$\int_{0}^{L} b(x) \cdot \delta u(x) \, dx + P \cdot \delta u(L) = \int_{0}^{L} EA(x) \frac{du}{dx} \cdot \frac{d(\delta u)}{dx} \, dx$$

>[!note] 关键思想
> 虚位移原理将微分形式的平衡方程转化为**积分弱形式**，降低了对位移函数光滑性的要求，这是有限元方法的重要理论基础。

---

# 2. 有限元离散化

## 2.1 位移插值与形函数

将连续体离散为有限个单元，单元内任意点的位移用**形函数**和**节点位移**表示：

$$u(x) = \sum_{i=1}^{n} N_i(x) \cdot d_i = \mathbf{N}(x) \cdot \mathbf{d}$$

其中：
- $N_i(x)$ —— 形函数（插值函数）
- $d_i$ —— 节点位移
- $n$ —— 单元节点数

## 2.2 虚位移的离散化

虚位移同样采用相同的形函数插值：

$$\delta u(x) = \mathbf{N}(x) \cdot \delta \mathbf{d}$$

虚应变为：

$$\frac{d(\delta u)}{dx} = \frac{d\mathbf{N}}{dx} \cdot \delta \mathbf{d} = \mathbf{B} \cdot \delta \mathbf{d}$$

其中 **$\mathbf{B}$ 为应变-位移矩阵**：

$$\mathbf{B} = \frac{d\mathbf{N}}{dx}$$

---

# 3. 单元刚度矩阵推导

## 3.1 代入虚位移原理

将离散化表达式代入虚功方程，先看**内虚功**：

$$\delta W_{int} = \int_{V} \sigma \cdot \delta \epsilon \, dV = \int_{V} (\mathbf{E} \cdot \mathbf{B} \cdot \mathbf{d})^T \cdot (\mathbf{B} \cdot \delta \mathbf{d}) \, dV$$

对于线弹性材料：

$$\delta W_{int} = \delta \mathbf{d}^T \cdot \left( \int_{V} \mathbf{B}^T \mathbf{E} \mathbf{B} \, dV \right) \cdot \mathbf{d}$$

## 3.2 外虚功的离散化

$$\delta W_{ext} = \int_{V} \mathbf{N}^T \mathbf{b} \, dV \cdot \delta \mathbf{d} + \mathbf{N}^T \mathbf{P} \cdot \delta \mathbf{d}$$

即：

$$\delta W_{ext} = \delta \mathbf{d}^T \cdot \mathbf{f}$$

其中 **等效节点力**：

$$\mathbf{f} = \int_{V} \mathbf{N}^T \mathbf{b} \, dV + \mathbf{N}^T \mathbf{P}$$

## 3.3 单元刚度矩阵

由 $\delta W_{int} = \delta W_{ext}$，且 $\delta \mathbf{d}$ 为任意虚位移：

$$\boxed{\mathbf{K}^e \mathbf{d} = \mathbf{f}}$$

其中 **单元刚度矩阵** 为：

$$\mathbf{K}^e = \int_{V} \mathbf{B}^T \mathbf{E} \mathbf{B} \, dV$$

>[!tip] 刚度矩阵的物理意义
> - $\mathbf{K}^e$ 是 $n \times n$ 对称矩阵（$n$ 为单元自由度）
> - 元素 $K_{ij}$ 表示：第 $j$ 个自由度产生单位位移时，在第 $i$ 个自由度上产生的力
> - 反映了单元的力-位移关系

---

# 4. 整体有限元方程

## 4.1 组装过程

将所有单元的贡献按自由度编号组装到整体系统中：

$$\mathbf{K} = \sum_{e} \mathbf{K}^e, \quad \mathbf{F} = \sum_{e} \mathbf{f}^e$$

得到 **整体平衡方程**：

$$\mathbf{K} \mathbf{D} = \mathbf{F}$$

## 4.2 求解流程

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   离散化网格    │────▶│  计算 K^e 和 f  │────▶│   组装 K 和 F   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
┌─────────────────┐     ┌─────────────────┐              ▼
│   后处理结果    │◀────│ 求解 KD = F    │◀────┐   引入边界条件
└─────────────────┘     └─────────────────┘     │   (约束处理)
                                                │
                                         ┌──────────────┘
                                         ▼
                                  ┌─────────────────┐
                                  │  得到节点位移 D │
                                  └─────────────────┘
```

---

# 5. 小结

| 概念 | 表达式 | 说明 |
|:---:|:---:|:---|
| 虚位移原理 | $\delta W_{ext} = \delta W_{int}$ | 等效积分弱形式 |
| 位移插值 | $u = \mathbf{N}\mathbf{d}$ | 形函数逼近真实位移 |
| 应变-位移 | $\epsilon = \mathbf{B}\mathbf{d}$ | 几何方程的离散形式 |
| 单元刚度 | $\mathbf{K}^e = \int \mathbf{B}^T \mathbf{E} \mathbf{B} dV$ | 虚位移原理的核心结果 |
| 整体方程 | $\mathbf{K}\mathbf{D} = \mathbf{F}$ | 有限元最终求解形式 |

>[!info] 核心要点
> 虚位移原理为有限元方法提供了严格的数学基础：
> 1. 通过**加权余量法**将微分方程转化为积分形式
> 2. 以**形函数**为权函数，自然导出刚度矩阵
> 3. 单元刚度矩阵具有**对称、正定、稀疏**的特性
> 4. 适用于复杂几何和非均质材料的数值求解