# Agentic RAG 与 ReAct 检索自纠偏

## 1. ReAct 是什么

ReAct = Reason + Act。它不是固定流水线一次跑到底，而是每一步都记录：

```text
Thought: 当前为什么要这么做
Action: 调用哪个工具，带什么参数
Observation: 工具返回了什么，质量够不够
Decision: 继续、纠偏、降级，还是生成最终回答
```

固定 RAG 一般是：

```text
query expansion -> retrieve -> rerank -> generate
```

当前项目改造后是：

```text
query planning
  -> query expansion
  -> ReAct retrieval step 1
  -> observe retrieval quality
  -> if weak: repair query + enlarge top_k
  -> ReAct retrieval step 2
  -> select best attempt
  -> generate grounded answer
```

## 2. 查询规划层

新增 `agents/rag_agent/query_planner.py`，参考 Ragent 的 `rewriteWithSplit(question, history)`：

- 医学术语归一：如 `covid19 -> COVID-19`、`xray -> X-ray`、`hba1c -> HbA1c`
- 多轮追问补全：如上一轮在聊 diabetes，用户问“这个需要做什么检查”，会补成带 diabetes/glucose 主题的查询
- 复杂问题拆分：按 `and / 以及 / 并且 / 同时` 等拆成最多 3 个子问题
- 主动澄清：没有上下文又问“这个严重吗 / 怎么办 / 要吃药吗”时，不硬答，要求补充疾病、症状、持续时间和目标问题

## 3. 复杂问题示例

用户问：

```text
Compare diabetes symptoms and HbA1c testing and treatment basics
```

Query plan：

```text
Thought:
问题同时问症状、检查和基础治疗，需要拆成多个子问题分别检索。

Action:
split query into:
1. Compare diabetes symptoms?
2. diabetes HbA1c testing?
3. diabetes treatment basics?

Observation:
得到 3 个子问题，每个子问题分别进入 ReAct 检索。
```

检索过程：

```text
Thought 1:
先用扩展后的 diabetes 查询检索本地知识库。

Action 1:
tool = knowledge_base
query = "diabetes symptoms HbA1c testing treatment basics ..."
top_k = 5

Observation 1:
document_count = 0
confidence = 0.0
sufficient = false

Thought 2:
第一次检索质量不足，去掉格式噪声，扩大召回范围。

Action 2:
tool = knowledge_base
query = "diabetes symptoms HbA1c testing treatment"
top_k = 10

Observation 2:
document_count = 3
confidence = 0.82
sufficient = true
```

## 4. Agent 怎么判断检索、调工具还是反问

第一层是 `IntentRouter`：

- 医学知识、疾病解释、文献问答：`RAG_AGENT` + `knowledge_base`
- 最新指南、近期研究、当前疫情：`WEB_SEARCH_PROCESSOR_AGENT` + `web_search`
- 胸片、脑 MRI、皮肤病灶图片：对应 Vision Agent + vision tool
- 剂量、处方、急症、自伤、prompt injection：安全拦截
- 意图低置信或上下文不足：主动澄清

第二层是 RAG 内部：

- 默认先走本地知识库
- 若检索质量不足，ReAct 修正 query 并扩大 top_k
- 若最终 confidence 仍低，外层图会降级到 Web Search

## 5. 如何防止死循环

ReAct 循环有硬边界：

- `react_max_steps = 2`
- `react_timeout_seconds = 8.0`
- `min_retrieval_confidence = 0.40`

达到置信度阈值就提前停止；一直失败也最多跑两步，避免无限“再想想、再搜搜”。

## 6. 工具调用失败怎么降级

检索 channel 失败时：

- 单个 channel 返回空结果和 `error`
- 其他 channel 继续执行
- ReAct observation 记录 `channel_errors`
- 第一轮失败后会用更保守 query 和更大 top_k 重试
- 若仍低置信，外层 `confidence_based_routing` 转到 Web Search Agent

MCP-like 工具注册表在 `agents/tools`：

| tool_id | kind | agent |
| --- | --- | --- |
| `knowledge_base` | KB | `RAG_AGENT` |
| `web_search` | EXTERNAL_SEARCH | `WEB_SEARCH_PROCESSOR_AGENT` |
| `chest_xray_classifier` | VISION_MODEL | `CHEST_XRAY_AGENT` |
| `skin_lesion_segmenter` | VISION_MODEL | `SKIN_LESION_AGENT` |
| `brain_mri_analyzer` | VISION_MODEL | `BRAIN_TUMOR_AGENT` |
| `medical_safety_policy` | SAFETY | `INPUT_GUARDRAILS` |

路由阶段会做 tool registry lookup，并写入 `agent_trace` 和 `route_decision.tool_registry`。

## 7. KB 和 Tool 结果如何区分

当前本地知识库上下文在 prompt 中以文档块形式提供：

```text
===DOCUMENT SECTION===
...
```

Trace 中保留来源：

- `retrieval_channel = faq_keyword_search`
- `retrieval_channel = intent_directed_medical_search`
- `retrieval_channel = qdrant_global_search`

工具层通过 `tool_id` 区分：

- KB 证据：`knowledge_base`
- 外部实时信息：`web_search`
- 影像模型：`chest_xray_classifier` / `skin_lesion_segmenter` / `brain_mri_analyzer`
- 安全策略：`medical_safety_policy`

后续接标准 MCP 时，可以把 prompt context 进一步显式分为：

```text
[KB:knowledge_base]
...

[TOOL:web_search]
...
```

## 8. 面试回答模板

**问：ReAct 和固定流水线的区别是什么？**

> 固定 RAG 是 query expansion、retrieve、rerank、generate 一次跑完。ReAct 是每一步先 thought，再 action 调工具，再 observation 看结果。我的项目里 ReAct 主要用于检索自纠偏：第一次检索 chunk 太少或 confidence 太低时，不直接生成，而是修正 query、扩大 top_k 再检索一次，最后选择质量最高的一轮结果生成回答。

**问：用户问了系统回答不了的问题怎么办？**

> 分两类。第一类是信息不足，比如“这个严重吗”，没有上下文时系统会主动澄清，让用户补充疾病、症状、持续时间和想问的目标。第二类是本地知识库不足，如果 RAG confidence 低或回答提示 insufficient information，外层 Agent Graph 会降级到 Web Search；如果仍没有可靠信息，会明确说明信息不足，而不是编造答案。

**问：ReAct 怎么防止死循环？**

> 我设置了最大步数和超时，当前 `react_max_steps=2`、`react_timeout_seconds=8s`。每轮观察检索结果数量、confidence 和 channel error，达到阈值就停止；不达标就最多纠偏一次。这样保留了 Agentic 自纠偏能力，但不会无限循环。
