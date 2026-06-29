# 基于 Ragent 的改造对比

## 一句话结论

`nageoffer/ragent` 是企业级 Agentic RAG 平台，重点是知识库平台化、意图树、MCP 工具、后台治理、模型路由和全链路 Trace。

当前医疗助手不是照搬 Ragent 的 Java/Spring 平台，而是把它的核心工程思想落到 Python/FastAPI + LangGraph 项目里：意图路由、查询规划、多路检索、ReAct 自纠偏、会话记忆、MCP-like 工具注册、模型可靠性和评测闭环。

## 已经对标补齐的能力

| 能力 | Ragent 做法 | 当前项目改造 |
| --- | --- | --- |
| 意图识别 | 树形 IntentNode + 分数阈值 + 澄清引导 | `agents/router` 中 15 类医疗意图，低置信或上下文不足时主动澄清 |
| 查询重写与拆分 | `QueryRewriteService.rewriteWithSplit(question, history)` | `QueryPlanner` 做术语归一、追问补全、复杂问题拆分，最多 3 个子问题 |
| 多路检索 | 意图定向、全局检索、多通道召回 | FAQ keyword、intent-directed、Qdrant global 三通道 |
| 检索后处理 | 去重、过滤、重排、融合 | Dedup、FAQ priority boost、CrossEncoder rerank、confidence scoring |
| ReAct 自纠偏 | 检索不足时调整策略 | `RAGReActController` 记录 thought/action/observation，低质量时扩大 top_k 并修正 query |
| 会话记忆 | 摘要 + 最近 N 条历史 | `ConversationMemoryService` 本地 JSON 存储，滑动窗口 + 摘要压缩 |
| MCP 工具 | `McpToolRegistry` + `McpToolExecutor` | `agents/tools` 中 MCP-like registry，统一登记 KB、Web、影像和安全工具 |
| Trace | RAG 链路节点追踪 | `route_decision`、`agent_trace`、`retrieval_trace`、`react_trace`、`ingestion_trace` |
| 模型可靠性 | 模型路由、健康状态、首包探测 | `ModelGateway` + retry + 三态熔断器 + FIFO 限流；首包探测 helper 已预留 |
| 评测 | 多层评测闭环 | intent/tool/FAQ retrieval/model gateway/prompt/rate limiter/react/query planner 等本地检查 |

## 当前 RAG 链路

```text
用户问题
  -> 输入安全校验
  -> 医疗意图路由
  -> QueryPlanner: 重写 / 追问补全 / 子问题拆分 / 澄清判断
  -> QueryExpander: 医学术语扩展
  -> 每个子问题走 ReAct 检索
  -> 多路检索: FAQ keyword + intent-directed + Qdrant global
  -> 去重 / FAQ 优先级 boost / rerank / confidence
  -> 合并子问题证据
  -> Grounded response generation
  -> 输出安全校验
```

这已经不是 Naive RAG 的单次 `retrieve -> generate`，而是一个可观测、可纠偏、可评测的 Agentic RAG 链路。

## 当前知识库规模

当前可复现 inventory：

- 已入库 PDF：4 篇
- 候选未入库 PDF：12 篇
- FAQ markdown：7 个领域文件
- FAQ 条目：150 条
- Qdrant points：131
- Docstore files：131
- Parsed doc files：194

FAQ 覆盖领域：

- brain tumor
- chest X-ray / COVID-19
- diabetes
- cardiovascular
- medication safety
- skin lesion
- general medical safety and triage

运行 inventory：

```bash
python scripts/knowledge_inventory.py --json-output data/knowledge_inventory.json
```

## 评测结果

当前 seed baseline：

| 评测项 | 当前结果 |
| --- | ---: |
| Intent routing | 50/50 = 100% |
| Agent routing | 50/50 = 100% |
| Tool routing | 50/50 = 100% |
| Tool-call eval | 25/25 = 100% |
| FAQ keyword Recall@1 / Recall@5 | 81.00% / 96.00% |
| FAQ keyword MRR | 0.933 |
| Full RAG Recall@1 / Recall@5 | 80.00% / 96.00% |
| Full RAG MRR | 0.941 |

新增本轮检查：

- `evals/test_query_planner.py`
- `evals/test_conversation_memory.py`
- `evals/test_tool_registry.py`

## 和 Ragent 的主要差距

当前项目已经补齐核心思想，但仍不是完整企业级平台：

1. 知识库管理仍是本地目录 + 脚本，没有 Ragent 那种后台管理、任务表和知识库生命周期管理。
2. MCP 是 MCP-like in-process registry，还不是标准 MCP server/client 协议栈。
3. 意图树是确定性医疗意图分类，不是可后台编辑的多级 IntentNode 树。
4. 会话记忆是本地 JSON，适合 demo 和单机部署，不是数据库持久化和异步摘要任务。
5. 模型路由已有 retry、限流、熔断和首包探测 helper，但多模型候选优先级调度还可以继续增强。
6. 评测集仍是 seed baseline，面向真实用户问题还需要扩展更多难例。

## 面试推荐说法

可以这样讲：

> 我参考了 Ragent 的企业级 Agentic RAG 架构，但没有照搬它的 Java/Spring 平台。我的项目是医疗垂直场景，所以重点把 Ragent 的核心思想落到 Python 医疗助手里：先做意图路由和安全拦截，再做查询重写、追问补全和复杂问题拆分，然后通过 FAQ keyword、意图定向和全局向量三路检索召回证据。检索结果会经过去重、优先级 boost、重排和置信度评估，如果质量不够，ReAct controller 会扩大 top_k 并修正 query 再检索一次。

> 另外我补了会话记忆，使用滑动窗口 + 摘要压缩控制 token；补了 MCP-like 工具注册表，把 knowledge_base、web_search、胸片分类、皮肤分割、脑 MRI 分析和安全策略统一成 tool_id；同时每次请求会输出 route_decision、agent_trace、retrieval_trace、react_trace。这个版本还不是 Ragent 那种完整企业平台，但已经从一个 demo RAG 改造成有意图、有工具、有记忆、有评测、有可靠性治理的 Agentic RAG 项目。
