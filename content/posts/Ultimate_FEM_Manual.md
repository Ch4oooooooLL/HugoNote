# 有限元方法 (FEM) 究极技术手册：从理论推导到 Python 实现

这份手册旨在为学习者提供“原子级”的细节拆解。我们将不仅仅给出公式，而是解释公式背后的每一个积分项和逻辑判断，并直接映射到 `PyFEM-Dynamics` 的源码。

---

## 第一章：数学之魂——有限元的核心原理

### 1.1 弱形式 (Weak Form) 的推导
原本的平衡微分方程（强形式）要求在每一点都成立，这在数值上很难直接求解。
通过将平衡方程乘以一个**权重函数**（即虚位移 $\delta u$）并在全域积分，我们得到了虚功原理的数学表达：
$$ \int_0^L (EA u')' \delta u dx + f \delta u = 0 \implies \int_0^L EA u' (\delta u)' dx = f \delta u |_0^L + \int_0^L q(x) \delta u dx $$
这就是**弱形式**。注意 $u'$ 和 $(\delta u)'$ 的出现，这意味着我们只需要位移的一阶导数（应变）可积即可，降低了对解的平滑度要求。

### 1.2 形函数（Interpolation Functions）的本质
在单元内部，我们无法获得真实位移 $u(x)$，只能通过节点位移 $d_i$ 进行插值。
- **线性插值 (Truss)**: $u(x) = a_0 + a_1 x$。通过边界条件 $u(0)=d_1, u(L)=d_2$ 联立方程解出 $a_i$，得到：
  $$ N_1(x) = 1 - \frac{x}{L}, \quad N_2(x) = \frac{x}{L} $$
- **Hermite 三次插值 (Beam)**: 因为梁方程涉及四阶导数，弱形式涉及二阶导数，所以要求位移及其导数（转角）在节点处都连续。
  $$ v(x) = a_0 + a_1 x + a_2 x^2 + a_3 x^3 $$
  通过 4 个边界条件（左右两端的 $v$ 和 $dv/dx$）解出系数，得到手册中提到的 $N_1 \sim N_4$。

---

## 第二章：单元矩阵的“原子”构成

### 2.1 刚度矩阵 $K_e$ 的积分过程
$$ K_{ij} = \int_V B_i^\top D B_j dV $$
其中 $B$ 是应变-位移矩阵（即形函数的导数）。
- **桁架单元**: $B = \frac{d}{dx}[N_1, N_2] = [-\frac{1}{L}, \frac{1}{L}]$。
  $$ K_e = A \int_0^L B^\top E B dx = \frac{EA}{L} \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix} $$
- **代码对应**: [core/element.py](file:///d:/CODE/FEM/PyFEM_Dynamics/core/element.py) 中的 [get_local_stiffness](file:///d:/CODE/FEM/PyFEM_Dynamics/core/element.py#28-32)。

### 2.2 质量矩阵的物理含义
- **一致质量矩阵 (Consistent)**: 考虑了单元内部质量分布的连续性。$M = \int \rho A N^\top N dx$。
- **集中质量矩阵 (Lumped)**: 假定质量分布在节点上。
  > **为什么用 Lumped?**
  > 在显式动力学或大型系统中，Lumped 矩阵是对角阵。在解 $Ma=F$ 时，对角阵意味着**不需要求逆**（只需除以对角线元素），极大提高了动态响应的计算效率。

---

## 第三章：坐标变换与系统组装

### 3.1 旋转矩阵 $T$ 的几何推算
二维空间中，局部坐标系 $x'$ 与全局坐标系 $x$ 的关系为：
$$ \begin{bmatrix} u' \\ v' \end{bmatrix} = \begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} $$
对于 4 自由度的桁架，变换矩阵 $\mathbf{T}$ 是一个分块对角阵。
**代码逻辑**: [get_transformation_matrix](file:///d:/CODE/FEM/PyFEM_Dynamics/core/element.py#91-103) 生成此对角块，然后通过 `T.T @ k_local @ T` 旋转。

### 3.2 组装算法的“零索引”陷阱
在 Python/NumPy 中，索引从 0 开始。
1. `node.dofs` 存储的是全局自由度编号（例如：0, 1, 2...）。
2. `np.ix_(dofs, dofs)` 会选取矩阵的子块进行叠加。
3. **注意**: 必须是**累加** (`+=`) 而不是替换，因为一个节点可能连接多个单元。

---

## 第四章：数值求解器深度剖析

### 4.1 边界条件的数学处理
- **划零划一法**: 本质是删除相关的代数方程。在 Python 中，我们通常保留维度，但修正方程。
- **乘大数法 (Penalty)**: 实际上是给节点接了一个**刚度极大的弹簧**。如果 $k_{spring} \gg k_{structure}$，则节点的位移将被迫接近 0。

### 4.2 Newmark-Beta 动力学积分的稳定性
$$ \mathbf{M}\ddot{\mathbf{u}} + \mathbf{C}\dot{\mathbf{u}} + \mathbf{K}\mathbf{u} = \mathbf{F} $$
- **无条件稳定条件**: $2\beta \ge \gamma \ge 0.5$。程序默认采用 $\beta=0.25, \gamma=0.5$，这意味着无论时间步长 $\Delta t$ 取多大，算法都不会发散。
- **等效荷载 $\hat{F}$ 的推导**:
  由于 $\ddot{u}_{t+\Delta t}$ 和 $\dot{u}_{t+\Delta t}$ 都可以表示为 $u_{t+\Delta t}$ 的函数，带入动力学方程并移项，所有已知的 $t$ 时刻量都移到右边，形成 $\hat{F}$。

---

## 第五章：Python 实现的高级策略

### 5.1 稀疏矩阵 (Sparse Matrix) 的选择
- `scipy.sparse.lil_matrix`: 适合**构建阶段**，插入新元素快。
- `scipy.sparse.csc_matrix`: 适合**计算阶段**，在执行 `spsolve` 或矩阵乘法时效率最高。
- **手册建议**: 永远在组装后调用 `.tocsc()` 转换格式。

### 5.2 线性方程组求解器的性能
- 对于较小模型：直接使用 `spla.spsolve`。
- 对于动力学（重复求解）: 使用 `spla.splu` 进行 LU 分解。
  > **LU 分解原理**: 将矩阵分解为上下三角阵。分解一次耗时较长，但后续的解方程过程从 $O(n^3)$ 降级为 $O(n^2)$。

---

## 结语：如何继续进阶？
1. **非线性**: 考虑几何大变形（$K$ 随位移变化）或材料非线性（$E$ 随应力变化）。
2. **三维空间**: 涉及绕 $X, Y, Z$ 三轴的旋转，变换矩阵更为复杂。
3. **振型分析**: 通过求解特征值问题 $det(K - \omega^2 M) = 0$ 获取结构的固有频率。

这份手册是您的起点。祝您在有限元的海洋中探索愉快！
