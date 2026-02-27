# AGENT.md - FEM Study Obsidian Vault

## 项目概述

这是一个关于**有限元方法（FEM）**学习的Obsidian笔记仓库，包含Python实现的有限元程序和相关理论知识。

- **仓库类型**: Obsidian笔记仓库
- **主题**: 有限元方法（Finite Element Method）
- **编程语言**: Python
- **笔记格式**: Markdown + Obsidian扩展语法

## 目录结构

```
/
├── .obsidian/              # Obsidian配置目录
│   ├── plugins/            # 插件配置
│   │   ├── obsidian-latex-suite/    # LaTeX公式支持
│   │   ├── obsidian-linter/         # 格式化工具
│   │   ├── obsidian-git/            # Git集成
│   │   └── folder-notes/            # 文件夹笔记
│   └── ...
├── posts/                  # 笔记内容目录
│   ├── 1D有限元Python程序复建与注解/
│   ├── 2D桁架PythonFEM程序/
│   ├── PyFEM-Dynamics 工作记录/
│   ├── 虚位移原理在有限元中的应用/
│   └── ...
├── search.md               # 搜索页面
└── archive.md              # 归档页面
```

## 笔记规范

### 1. Frontmatter格式

所有笔记使用YAML frontmatter:

```yaml
---
title: "笔记标题"
date created: 日期时间
date modified: 日期时间
tags: [标签1, 标签2, 标签3]
---
```

### 2. 标签系统

常用标签:
- `FEM` - 有限元方法相关
- `Python` - Python实现
- `NumericalAnalysis` - 数值分析
- `Structure` - 结构力学
- `Review` - 复习总结
- `Dynamics` - 动力学

### 3. 特殊语法

**数学公式（LaTeX）**:
- 行内公式: `$E = mc^2$`
- 块级公式:
  ```latex
  $$
  \frac{d}{dx}\left( E A(x) \frac{du}{dx} \right) = 0
  $$
  ```

**代码块**:
- Python代码使用 ```python 标记
- 包含完整可运行的代码示例

**Obsidian Callouts**:
- `> [!info]` - 信息提示
- `> [!warning]` - 警告提示
- `> [!note]` - 笔记提示

## 常用操作

### 创建新笔记

1. 在 `posts/` 目录下创建新的文件夹（如"新主题"）
2. 在文件夹内创建 `index.md` 作为主笔记
3. 添加标准frontmatter
4. 图片存放在文件夹内的 `images/` 子目录

### 编辑笔记

- 保持代码块的完整性和可运行性
- 数学公式使用LaTeX语法
- 添加适当的标签便于搜索
- 使用Obsidian Callouts增强可读性

### 图片处理

- 图片存放在对应笔记文件夹的 `images/` 子目录
- 使用相对路径引用: `./images/xxx.png`
- 建议使用描述性文件名

## 注意事项

1. **不要修改** `.obsidian/` 目录下的配置文件
2. 笔记使用 **UTF-8** 编码
3. 日期格式: `星期X, 月X日 年, 时:分:秒 上/下午`
4. 保持笔记的原子性，一个主题一个笔记
5. 定期使用obsidian-git插件同步到Git仓库

## 相关资源

- Obsidian: https://obsidian.md
- LaTeX语法参考: https://en.wikibooks.org/wiki/LaTeX/Mathematics
- NumPy文档: https://numpy.org/doc/
