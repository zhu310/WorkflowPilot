# WorkflowPilot：智能流程画布自然语言编辑系统

WorkflowPilot 是一个面向复杂业务流程画布的自然语言编辑原型。项目关注审批、数据处理、运维等场景中的流程调整问题：传统流程画布需要人工拖拽节点和连线，而直接让大模型输出最终图结构又容易产生错连、漏边和非目标区域误修改。

本项目采用“大模型语义理解 + 图结构对象绑定 + 约束规划 + 确定性执行”的分层架构。大模型不直接修改画布数据，而是表达用户的语义目标；程序侧负责对象绑定、IR 转换、约束规划、操作授权、确定性执行和结果校验。

## 核心链路

```text
用户自然语言指令
-> Semantic Goal 理解
-> Grounding 对象绑定
-> Lowering
-> Workflow IR
-> Compiler
-> Constraint Planner
-> Deterministic Executor
-> Transaction / Versioned Graph
```

这条链路的核心目标是把“语义理解”和“图结构修改”解耦：

- LLM 负责理解用户想要的编辑目标；
- Grounding 将“这个节点”“失败分支”“刚才新增的节点”等表达绑定到具体节点或边；
- Lowering 将语义目标转换为局部 Workflow IR；
- Compiler / Planner 将 IR 转换为节点、连线、属性和约束变更；
- Executor 只执行经过授权的确定性操作；
- Transaction / Version 记录执行过程和图快照，支持连续编辑与追溯。

## 主要能力

### 1. 自然语言流程编辑

系统支持对流程画布进行受控修改，包括：

- 新增、删除或修改节点；
- 调整节点属性，例如名称、坐标、汇聚属性；
- 创建、删除或修改连线；
- 修改条件分支的标签和条件表达式；
- 将简单串行结构调整为确定性并行结构；
- 将某条分支路由到指定节点。

### 2. Semantic Goal 与 Grounding

系统通过 Semantic Goal 表达用户目标，包括对象引用、目标状态、关系意图和语义约束等结构化信息。

Grounding 会结合以下上下文完成对象绑定：

- 当前图结构；
- 前端 Selection；
- 最近成功事务的 diff；
- 候选节点和候选边；
- 多轮交互中的 pending clarification 状态。

当引用无法唯一确定时，系统不会擅自执行，而是返回澄清问题和候选对象。

### 3. Workflow IR 与约束规划

Workflow IR 表示局部的目标图结构，而不是完整替换整张图。Compiler 会把 IR 转换为机器可执行的约束，包括：

- required relations；
- forbidden relations；
- node property changes；
- edge property changes；
- absent entities；
- preservation-related constraints。

Planner 根据这些约束生成操作计划，并为每一步操作附带 mutation authorization。Executor 会拒绝未授权或越界的操作。

### 4. 事务、版本与连续编辑

系统使用 Transaction 记录：

- 用户指令；
- 前端选区；
- 对象绑定结果；
- 规格与执行计划；
- 执行状态；
- 结果版本。

VersionState 维护会话级图快照，支持 undo、redo 和 rollback。最近成功事务的 diff 会被用于后续 Grounding，例如支持“刚才新增的节点”“刚才修改的那个节点”等上下文指代。

## 校验边界

系统实现的是机器可检查的结构与约束校验，包括：

- 对象引用是否缺失或歧义；
- Workflow IR schema 是否有效；
- 结构边界是否足够确定；
- required / forbidden relations 是否冲突；
- mutation authorization 是否覆盖所有操作；
- 执行后的图是否存在非法节点、非法边端点或重复 ID。

系统不声称能够证明任意自然语言需求已经被完整语义等价地满足。对于抽象优化类请求，系统只在存在确定性规则和可执行目标时执行；否则会进入澄清、部分执行或 unsupported 状态。

## 示例指令

可以尝试的指令包括：

- 把知识检索节点放到解析任务右侧。
- 在发布结果前增加事实校验。
- 把失败分支改名为人工审核。
- 失败流程转人工复核。
- 增加通知节点，同时保留异常处理。
- 把刚才新增的节点放到风控后面。

## 技术栈

- Python
- TypeScript
- LogicFlow
- LLM API
- Workflow IR
- Constraint Planner
- Pytest

## 当前边界

当前项目仍是原型系统，主要边界包括：

- 不包含长期记忆或持久化别名库；
- 不包含向量检索或行业知识库；
- 不保证任意真实 LLM 输出都稳定，需要结合回归测试验证；
- 不支持任意复杂并行边界推理；
- 不支持完整的自动化“抽象优化方案生成 -> 用户确认 -> 执行”闭环；
- LogicFlow 真实 viewport-based `visible_area` grounding 尚未完整实现。

## 项目价值

WorkflowPilot 的重点不是让大模型直接画流程图，而是设计一套可控的 AI 应用编辑架构：

> 让大模型负责语义理解，让程序负责对象绑定、约束规划和确定性图操作。

这种设计降低了复杂流程编辑中的错连、漏边和非目标区域误修改风险，也使每次修改具备事务记录、版本快照和可追溯的执行计划。
