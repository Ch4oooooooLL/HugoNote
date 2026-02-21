---
title: FEM 复习：1D 变截面杆件分析 (Python 实现)
date created: 星期二, 二月 17日 2026, 12:57:51 下午
date modified: 星期二, 二月 17日 2026, 6:22:06 晚上
tags: [FEM, Python, NumericalAnalysis, Structure, Review]
---


## 1. 问题定义与物理模型

**目标**：求解一根长度为 $L$、弹性模量为 $E$、横截面积 $A(x)$ 随位置变化的杆件，在右端受到拉力 $P$ 作用下的位移场 $u(x)$。

**控制方程 (Strong Form)**：
$$
\frac{d}{dx}\left( E A(x) \frac{du}{dx} \right) = 0, \quad x \in [0, L]
$$
**边界条件**：
1.  Dirichlet (本质边界): $u(0) = 0$ (左端固定)
2.  Neumann (自然边界): $E A(L) \frac{du}{dx}|_{x=L} = P$ (右端受力)

**有限元离散 (Weak Form)**：
对于线性杆单元，单元刚度矩阵 $k_e$ 为：
$$
k_e = \frac{EA_{avg}}{L_e} \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}
$$
其中 $A_{avg}$ 取单元中心处的截面面积近似。



## 2. 代码实现架构

程序整体分为五个模块：
1.  **前处理 (Pre-processing)**: 离散化几何域，生成节点与单元。
2.  **单元计算 (Element Calculation)**: 计算局部刚度矩阵。
3.  **组装 (Assembly)**: 构建全局刚度矩阵 $K$ 和载荷向量 $F$。
4.  **边界处理 (Boundary Conditions)**: 修正矩阵以引入约束。
5.  **求解与可视化 (Solver & Post-processing)**: 解方程组并绘图。

### 2.1 初始化与参数定义

```python
import numpy as np
import matplotlib.pyplot as plt

# === 物理参数 ===
L_total = 10.0      # 杆件总长度 [m]
P_load  = 1000.0    # 末端拉力 [N]
E_mod   = 2.1e11    # 弹性模量 [Pa]

# === 几何参数 (变截面模拟) ===
def get_area(x):
    """
    定义横截面积 A 随坐标 x 的变化函数
    此处模拟锥形杆: 根部 0.01 m^2 -> 末端 0.005 m^2
    """
    return 0.01 - (0.005 * x / L_total)

# === 网格控制 ===
NUM_ELEMENTS = 10                  # 单元数量
NUM_NODES    = NUM_ELEMENTS + 1    # 节点数量 (1D线性单元)
```

### 2.2 网格生成 (Meshing)

> [!info] **原理：拓扑关系**
> *   **Nodes (Coordinates)**: 存储物理坐标。
> *   **Connectivity (Topology)**: 存储节点间的连接关系。对于 1D 线性单元，第 $i$ 个单元连接节点 $i$ 和 $i+1$。

```python
# 1. 生成节点坐标数组
# linspace: 生成等间距分布的节点
nodes = np.linspace(0, L_total, NUM_NODES)

# 2. 生成单元连接表 (Connectivity Matrix)
# elements[i] = [左节点索引, 右节点索引]
elements = []
for i in range(NUM_ELEMENTS):
    elements.append([i, i + 1])

# 转为 array 方便后续索引
elements = np.array(elements)
```

> [!NOTE] **Python Syntax: `np.linspace`**
> `np.linspace(start, stop, num)` 生成包含 `num` 个点的等差数列（闭区间）。这是 FEM 生成结构化网格最常用的函数。

### 2.3 单元刚度矩阵与全局组装

此部分是 FEM 算法核心。

> [!info] **原理：直接刚度法 (Direct Stiffness Method)**
> 1.  **局部近似**: 假定单元内部属性均匀，取中点面积 $A_{mid}$ 计算 $k_e$。
> 2.  **自由度映射 (DOF Mapping)**:
>     * 局部索引 `0` (左) $\rightarrow$ 全局索引 `node_i`
>     * 局部索引 `1` (右) $\rightarrow$ 全局索引 `node_j`
> 3.  **累加**: 全局矩阵 $K_{global}$ 的对应位置累加局部矩阵的分量。

```python
# 初始化全局矩阵 (使用稠密矩阵，大规模问题需改用 scipy.sparse)
K_global = np.zeros((NUM_NODES, NUM_NODES))
F_global = np.zeros(NUM_NODES)

print("开始组装刚度矩阵...")

for e in range(NUM_ELEMENTS):
    # --- Step 1: 获取拓扑信息 ---
    # 获取当前单元连接的全局节点索引
    idx_L = elements[e, 0]  # 左节点 Global ID
    idx_R = elements[e, 1]  # 右节点 Global ID
    
    # 获取节点物理坐标
    x_L = nodes[idx_L]
    x_R = nodes[idx_R]
    
    # --- Step 2: 计算单元刚度矩阵 k_e ---
    # 单元长度
    Le = x_R - x_L
    # 单元中点坐标 -> 计算中点面积 (变截面近似处理)
    x_mid = (x_L + x_R) / 2.0
    A_avg = get_area(x_mid)
    
    # 刚度系数 k = EA/L
    k_val = E_mod * A_avg / Le
    
    # 局部刚度矩阵 (2x2)
    # [ k, -k]
    # [-k,  k]
    k_element = k_val * np.array([[1, -1], [-1, 1]])
    
    # --- Step 3: 组装到全局矩阵 (Scatter) ---
    # 映射关系:
    # local [0, 0] -> global [idx_L, idx_L]
    # local [0, 1] -> global [idx_L, idx_R] ...
    
    K_global[idx_L, idx_L] += k_element[0, 0]
    K_global[idx_L, idx_R] += k_element[0, 1]
    K_global[idx_R, idx_L] += k_element[1, 0]
    K_global[idx_R, idx_R] += k_element[1, 1]

print("组装完成。")
```

### 2.4 施加边界条件 (Boundary Conditions)

> [!warning] **关键点：奇异性消除**
> 原始 $K_{global}$ 是奇异矩阵（行列式为 0），代表结构存在刚体位移模式。必须引入 Dirichlet 边界条件才能求解。

> [!info] **算法：置 1 法 (Penalty/Replacement)**
> 目标：强制 $u_{node} = \bar{u}$。
> 操作：
> 1.  将 $K$ 中对应 **行** 全部置零。
> 2.  将对角线元素 $K_{ii}$ 置为 1。
> 3.  将 $F$ 中对应元素 $F_i$ 置为 $\bar{u}$。
> *注：为了保持对称性，通常也可将对应 **列** 置零，但这步对于非对称求解器不是必须的。*

```python
# 复制矩阵以保留原始数据（良好的编程习惯）
K_final = K_global.copy()
F_final = F_global.copy()

# 1. 施加 Neumann 边界 (力边界)
# 在最右端节点 (索引 -1) 施加拉力 P
F_final[-1] += P_load

# 2. 施加 Dirichlet 边界 (位移边界)
# 约束: 左端节点 (索引 0) 位移为 0
fixed_node = 0
fixed_val  = 0.0

# (1) 行清零
K_final[fixed_node, :] = 0.0
# (2) 对角线置1
K_final[fixed_node, fixed_node] = 1.0
# (3) 修正载荷项
F_final[fixed_node] = fixed_val

# (可选) 对称化处理：列清零
# 注意：若执行列清零，需修正 RHS 向量中受该节点影响的其他项
# F_final -= K_global[:, fixed_node] * fixed_val 
# 但由于此处 fixed_val=0，此步可省略。
```

> [!NOTE] **Python Syntax: Slicing**
> `array[i, :] = 0` 表示将第 `i` 行的所有列元素赋值为 0。这是 Numpy 进行矩阵操作的核心语法。

### 2.5 求解与后处理

```python
# === 求解线性方程组 Ax = b ===
# U 为位移向量
U = np.linalg.solve(K_final, F_final)

# === 理论解对比 (Exact Solution) ===
# 对于变截面杆，理论解需积分: u(x) = Integral(P / (E*A(x)) dx)
# 简单的数值积分用于对比
x_fine = np.linspace(0, L_total, 100)
u_exact = np.zeros_like(x_fine)
for i in range(1, len(x_fine)):
    # 简单的梯形积分
    x_seg = x_fine[i]
    # 近似计算 u = Integral(P/EA)
    # 这里仅做粗略对比示意
    dx = x_fine[i] - x_fine[i-1]
    mid_A = get_area((x_fine[i] + x_fine[i-1])/2)
    du = (P_load / (E_mod * mid_A)) * dx
    u_exact[i] = u_exact[i-1] + du

# === 绘图 ===
plt.figure(figsize=(10, 6))

# 绘制 FEM 结果
plt.plot(nodes, U, 'o-', label='FEM (Numerical)', linewidth=2, markersize=8)

# 绘制理论近似
plt.plot(x_fine, u_exact, 'r--', label='Theoretical (Approx)', alpha=0.7)

plt.title(f'1D Variable Cross-section Bar Analysis\nElements: {NUM_ELEMENTS}')
plt.xlabel('Position x [m]')
plt.ylabel('Displacement u [m]')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

print(f"最大位移 (FEM): {U[-1]:.6e} m")
```

---
