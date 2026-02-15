---
title: Push test
date: 2026-01-20
draft: false
math: true
tags:
  - Mechanics
  - Python
  - FEM
---
## 一、常见数理方程的建立

### 1.1 三类典型的方程

+ 双曲方程 (以波动方程为代表):$\frac{\partial^2 u}{\partial t^2}-a^2\nabla^2u=f(x,y,z,t)$
+ 抛物型方程 (热传导方程为代表):$\frac{\partial u}{\partial t}-a^2\nabla^2u=f(x,y,z,t)$
+ 椭圆型方程 (泊松方程为代表):$\nabla^2u=f(x,y,z,t)$
### 1.2 波动微分方程的建立
#### 1.2.1 基本假设
+ 弦是理想的柔软的,即只可以承受拉力不可以承受压力
+ 弦是均匀的,即 $\rho$ 是常数
+ 微小振动原理:$\sin\theta \approx tan \theta$
#### 1.2.2 方程建立过程
+ 选取 $(x,x+\Delta x)$ 微元,使用 $u(x,t)$ 表示在 t 时刻,x 位置的位移
+ 对微元竖直方向进行分析:$F_u=T_2sin\beta-T_2sin\alpha$
+ 在几何上,$tan \alpha=\frac{\partial u}{\partial x}\Big|_{x}$,$tan\beta=\frac{\partial u}{\partial x}\Big|_{x+\Delta x}$
+ 垂直合力可以写成:$F_u=T(\frac{\partial u}{\partial x}(x+\Delta x,t)-\frac{\partial u}{\partial x}(x,t))$
+ 微元质量为 $\Delta m=\rho \Delta x$
+ 微元的垂直加速度 $a=\frac{\partial^2u}{\partial t^2}$
+ 由 $F=ma$ 可以得到:$$T\left[\frac{\partial u}{\partial x}(x+\Delta x,t)-\frac{\partial u}{\partial x}(x,t) \right]=\frac{\partial^2u}{\partial t^2}(\rho \Delta x)$$
+ 对左侧除以 $\Delta x$,当其向 0 逼近时,有 $$\lim_{ \Delta x \to 0 }\frac{T\left[ \frac{\partial u}{\partial x}(x+\Delta x,t)-\frac{\partial u}{\partial x}(x,t) \right]}{\Delta x}=\frac{\partial^2u}{\partial x^2} $$
+ 进而,方程转变为:$$T\frac{\partial^2 u}{\partial x^2}=\rho\frac{\partial ^2 u}{\partial t^2}$$
+ 令 $a=\frac{\rho}{T}$,整理得到一元振动微分方程:$$\frac{\partial^2u}{\partial x ^2}=a^2\frac{\partial^2u}{\partial t^2}$$
#### 1.2.3 各类变体

常见的对一维波动方程有影响的因素包括: 重力影响、阻尼、弹性支承、垂直悬挂等。
+ 受到重力影响时，需要引入重力项，即 $F_{u}=T[\tan \beta -\tan \alpha]-\rho \Delta x=\rho \Delta x\frac{\partial^2u}{\partial t^2}$
+ 收到阻尼影响时，引入一个和 $u_{t}$ 方向相反的变量 $F_{f}=-k\Delta x\frac{\partial u}{\partial t}$，对外力的描述进行修正
### 1.3 热传导微分方程的建立
#### 1.3.1 基本设定
+ 考虑一个各边长为 $\Delta x,\Delta y,\Delta z$,则在三个方向上存在温度差，满足 Fourier 热传导定律
+ Fourier 热传导定律：$$
  \begin{aligned}
   q_x=-k \frac{\partial u}{\partial x} \\
   q_{y}=-k\frac{\partial u}{\partial y} \\
   q_{z}=-k \frac{\partial u}{\partial z} \\
   q=-k \nabla u
  \end{aligned}

  $$
#### 1.3.2 方程的建立
+ 则在 $\Delta t$ 时间内沿着 $x$ 方向流入六面体的热量：
$$
  
  \begin{gathered}
   \left[q_{x}\Big|_{x} - q_{x} \Big|_{x+\Delta x}\right] =\left[  k \frac{\partial u}{\partial x}_{x+\Delta x}-k  \frac{\partial u}{\partial x}_{x}  \right]\Delta x\Delta z\Delta t=k \frac{\partial^2 u}{\partial x^2}\Delta x\Delta y\Delta z\Delta t \\
  \left[q_{y}\Big|_{y}-q_{y}\Big|_{y+\Delta y}\right] = k \frac{\partial^2 u}{\partial y^2}\Delta x\Delta y\Delta z\Delta t \\
  \left[q_{z}\Big|_{z}-q_{z}\Big|_{z+\Delta z}\right]=k \frac{\partial^2u}{\partial z^2}\Delta x\Delta y\Delta z\Delta t
  

\end{gathered}

$$
+ 在没有其他热量来源和热量消耗的情况下，净流入的热量应当等于介质在 $\Delta t$ 内升高对应温度所需要的热量：$$
  k \left(\frac{\partial^2u}{\partial x^2}+ \frac{\partial^2u}{\partial y^2}+ \frac{\partial^2u}{\partial z^2}\right)\Delta x\Delta y\Delta z\Delta t=\rho \Delta x\Delta y\Delta z\cdot c \cdot \Delta u
$$
+ 整理后得到:$$
  \frac{\partial u}{\partial t}-\frac{k}{\rho c} \nabla ^2u =0
  $$
#### 1.3.3 方程的变体
+ 当介质不均匀时，即 k 与 $(x,y,z)$ 相关，热传导方程变为：$$
  \rho c \frac{\partial u}{\partial t}-\nabla(k\nabla u)=F(x,y,z,t)
  $$
+ 对含有热源的方程的推导：

	那么，在 $\Delta t$ 时间内，体积元 $\Delta x \Delta y \Delta z$ 内部**产生**的热量为：
	$$ Q_{source} = F \cdot \Delta x \Delta y \Delta z \cdot \Delta t $$
	
	
	**新的能量守恒逻辑为：**
	$$ (\text{净流入的热量}) + (\text{内部产生的热量}) = (\text{介质升温所需的内能增量}) $$
	
	**对应的数学表达式为：**
	
	$$ \underbrace{k \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2} \right) \Delta x \Delta y \Delta z \Delta t}_{\text{净流入的热量}} + \underbrace{F \cdot \Delta x \Delta y \Delta z \cdot \Delta t}_{\text{内部产生的热量}} = \underbrace{\rho \Delta x \Delta y \Delta z \cdot c \cdot \Delta u}_{\text{内能增量}} $$
	
	
	两边同时除以 $\Delta x \Delta y \Delta z \cdot \Delta t$：
	
	$$ k \nabla^2 u + F = \rho c \frac{\Delta u}{\Delta t} $$
	
	当 $\Delta t \to 0$ 时，$\frac{\Delta u}{\Delta t}$ 变为 $\frac{\partial u}{\partial t}$：
	
	$$ k \nabla^2 u + F = \rho c \frac{\partial u}{\partial t} $$

	非齐次热传导方程形式：
$$ \frac{\partial u}{\partial t} = a^2 \nabla^2 u + f $$

## 二、定解问题
+ **边界条件**和**初始条件**统称为*定解条件*，方程附加上定解条件构成定解问题
### 2.1 初始条件与边界条件的提出

#### 步骤 1：构建几何与时间框架 (Setup)
在确定条件之前，必须先明确“舞台”在哪里。
*   **画草图**：画出物理对象的几何形状（杆、膜、体）。
*   **建坐标**：确定原点、正方向、定义域范围（如 $0 \leq x \leq L$）。
*   **定变量**：明确未知函数 $u(x,t)$ 代表什么物理量（温度、位移、电势等）。

#### 步骤 2：确定初始条件 (Initial Conditions - $t=0$)
**核心逻辑**：描述过程开始那一瞬间的状态。
*   **判别阶数**：
    * 如果是**热传导/扩散方程**（对时间 $t$ 是一阶导数），只需要**1 个**初始条件（初始分布）。
    * 如果是**波动方程**（对时间 $t$ 是二阶导数），需要**2 个**初始条件（初始位置 + 初始速度）。
*   **操作提问**：
    1.  $t=0$ 时，物体各处的 $u$ 值是多少？ $\rightarrow u|_{t=0} = \varphi(x)$
    2.  $t=0$ 时，物体各处的变化率（速度）是多少？ $\rightarrow \frac{\partial u}{\partial t}|_{t=0} = \psi(x)$

#### 步骤 3：确定边界条件 (Boundary Conditions - Space Limits)
**核心逻辑**：描述物体边界与外界环境的相互作用。
*   **定位边界**：找出几何区域的边缘（如 $x=0$ 和 $x=L$）。
*   **分类对号入座**：
    *   **第一类（Dirichlet）**：直接控制边界上的**值**（如固定温度、固定位移）。
    *   **第二类（Neumann）**：控制边界上的**通量/梯度**（如绝热、受力、流速）。
    *   **第三类（Robin）**：值与梯度的**线性组合**（如牛顿冷却定律、弹性支撑）。

#### 步骤 4：一致性与适定性检查 (Check)
*   **相容性**：检查初始条件在边界处的值是否与边界条件冲突（例如：$t=0$ 时端点温度是否符合边界设定的温度）。

---

翻译”对照表 (核心工具)

这是操作中最实用的部分，通过物理描述直接映射到数学表达式。

#### 1. 初始条件映射 ($t=0$)

| 物理描述 | 数学表达式 | 适用方程 |
| :--- | :--- | :--- |
| **初始温度分布**为 $f(x)$ | $u(x,0) = f(x)$ | 热传导/波动 |
| **初始形状/位移**为 $f(x)$ | $u(x,0) = f(x)$ | 波动 |
| **静止开始** (Starting from rest) | $u_t(x,0) = 0$ | 波动 |
| **初始给予速度** $g(x)$ | $u_t(x,0) = g(x)$ | 波动 |
| **受冲击/敲击** (Impulse) | $u_t(x,0) = A\delta(x-x_0)$ (狄拉克函数) | 波动 |

#### 2. 边界条件映射 ($x=0$ 或 $x=L$)

| 边界类型           | 物理场景 (热学)               | 物理场景 (力学/波动)           | 数学表达式 (设边界在 $x_0$)                                                          |
| :------------- | :---------------------- | :--------------------- | :-------------------------------------------------------------------------- |
| **第一类 (固定值)**  | 恒温、保持在 $T_0$ 度          | 端点固定、铰支                | $u(x_0, t) = \text{Constant}$                                               |
| **第二类 (导数)**   | **绝热** (Insulated)、热流为 0 | **自由端** (Free end)、无张力 | $\frac{\partial u}{\partial n} = 0$ (法向导数为 0)                                |
| **第二类 (非零通量)** | 注入恒定热流 $q$              | 端点受已知外力 $F(t)$         | $K\frac{\partial u}{\partial x} = q$ 或 $T\frac{\partial u}{\partial x} = F$ |
| **第三类 (混合)**   | 表面散热 (牛顿冷却定律)           | 弹性支撑 (弹簧连接)            | $\frac{\partial u}{\partial n} + h(u - u_{env}) = 0$                        |

## 3 达朗贝尔公式与行波法求解一维波动方程
### 3.1 公式适用场景与数学模型
+ 达朗贝尔公式用于求解**一维、无界弦的自由振动问题**，方程标准形式为：$$
  \begin{cases}
  \frac{\partial^2u}{\partial t^2}=a^2 \frac{\partial^2u}{\partial x^2} & (-\infty < x<+ \infty,t>0 )  \\
  u(x,y)=f(x)  &  \text{初始位移} \\
  \frac{\partial u}{\partial t}(x,0)=g(x)  &  \text{初始速度} 
  \end{cases}
  $$
### 3.2 行波法主要思想
+ 通过变量代换的方式，将波动方程改写成两部分函数的相加：$$
  u(x,t)=F(x-at)+G(x+at)
  $$
+ 在这其中，F 表示向右传播的波，G 表示向左传播的波
### 3.3 达朗贝尔公式
+ 达朗贝尔公式的主要形式如下：$$
  u(x,t)=\frac{1}{2} \left[ f(x-at)+f(x+at)\right]+ \frac{1}{2a} \int_{x-at}^{x+at}g(\tau)d\tau
  $$
+ 公式各部分理解，前半部分表示位移贡献项，后半部分表示速度贡献项
### 3.4 达朗贝尔公式解题步骤
+ 求解以下定解问题： $$ \begin{cases} u_{tt} = 4 u_{xx} & (-\infty < x < +\infty, t > 0) \\ u(x, 0) = \sin x \\ u_t(x, 0) = \cos x \end{cases} $$ **解答过程：** 
+ 
1.  **识别参数**： * 对比标准方程 $u_{tt} = a^2 u_{xx}$，可知 $a^2 = 4$，即 **$a = 2$**。 * 初始位移 **$f(x) = \sin x$**。 * 初始速度 **$g(x) = \cos x$**。 2. **代入公式**： $$ u(x, t) = \frac{1}{2}[\sin(x - 2t) + \sin(x + 2t)] + \frac{1}{2 \times 2} \int_{x - 2t}^{x + 2t} \cos \tau \, d\tau $$
2.  **分步计算**：
+ **第一部分（位移项）**： 利用积化和差公式 $\sin A + \sin B = 2 \sin \frac{A+B}{2} \cos \frac{A-B}{2}$： $$ \frac{1}{2}[\sin(x - 2t) + \sin(x + 2t)] = \frac{1}{2} \cdot [2 \sin(x) \cos(2t)] = \sin(x)\cos(2t) $$ 
+ **第二部分（速度项）**： $$ \begin{aligned} \frac{1}{4} \int_{x - 2t}^{x + 2t} \cos \tau \, d\tau &= \frac{1}{4} [\sin \tau]_{x - 2t}^{x + 2t} \\ &= \frac{1}{4} [\sin(x + 2t) - \sin(x - 2t)] \end{aligned} $$ 利用积化和差公式 $\sin A - \sin B = 2 \cos \frac{A+B}{2} \sin \frac{A-B}{2}$： $$ \sin(x + 2t) - \sin(x - 2t) = 2 \cos(x) \sin(2t) $$ 所以第二部分为： $$ \frac{1}{4} [2 \cos(x) \sin(2t)] = \frac{1}{2} \cos(x) \sin(2t) $$
1. **合并结果**： $$ u(x, t) = \sin(x)\cos(2t) + \frac{1}{2} \cos(x) \sin(2t) $$   **答案：** $u(x, t) = \sin x \cos 2t + \frac{1}{2} \cos x \sin 2t$
