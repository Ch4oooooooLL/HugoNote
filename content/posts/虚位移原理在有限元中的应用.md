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
