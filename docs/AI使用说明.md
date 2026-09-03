# WorkflowPilot AI 使用说明

## 1. AI 辅助范围

本项目开发过程中使用 AI 工具辅助：

- 代码阅读与问题定位
- 架构方案讨论
- 测试用例设计
- 文档整理
- 部分代码实现辅助

## 2. 人工完成的核心设计

项目核心架构由人工设计和确认，包括：

- 自然语言编辑到 Workflow IR 的整体链路
- Graph Context / Desired State / Obligation 设计
- Constraint Planner 约束执行机制
- Executor 安全修改机制
- Validation 验证闭环
- Scope grounding 设计
- Mutation authorization 设计

## 3. AI 输出的验证流程

所有 AI 辅助生成内容均经过：

1. 人工架构审查
2. 代码修改确认
3. 单元测试验证
4. Regression 测试验证
5. Evidence 汇总验证

## 4. AI 方案调整案例

开发过程中曾对 AI 建议进行修改，例如：

### 示例1：避免 operation type 膨胀

AI 初期可能倾向：

```text
rename_operation
redirect_operation
condition_operation
```

等专用操作。

最终调整为：

统一 Workflow IR + constraints + deterministic executor。

### 示例2：避免直接生成图修改

调整为：

```text
Intent
↓
Workflow IR
↓
Constraints
↓
Planner
↓
Executor
↓
Validation
```

### 示例3：condition 编辑限制

没有采用开放式条件生成，而限定：

explicit condition expression replacement

避免模型错误生成业务规则。

## 5. 当前 AI 使用原则

AI 用于：

- 提升开发效率
- 辅助分析
- 提供候选方案

最终设计决策、代码审查、测试验证由人工完成。
