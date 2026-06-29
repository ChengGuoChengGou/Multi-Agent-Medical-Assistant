# 评测体系设计

本项目的评测目标不是只看最终回答“像不像”，而是分层评估 Agentic RAG 的关键环节：

```text
用户问题
  -> Intent Router
  -> Tool / Agent Selection
  -> Retrieval
  -> Rerank / Confidence
  -> Generation
  -> Safety / Fallback
```

## 1. 当前评测集规模

当前是 seed evaluation，用于建立可运行、可解释、可扩展的评测框架。

| 数据集 | 条数 | 作用 |
| --- | ---: | --- |
| `intent_routing_eval.jsonl` | 50 | 评估意图识别、Agent 路由、工具选择 |
| `tool_call_eval.jsonl` | 25 | 评估工具调用真值和 Agent 选择 |
| `faq_retrieval_eval.jsonl` | 50 | 评估 FAQ 检索 Recall@K、MRR、nDCG@5、domain/priority/source 命中 |

覆盖场景：

- 普通对话、能力询问、越界请求
- 本地知识库问答、文献型问题、最新医学进展
- 症状咨询、处方/剂量安全拒答、prompt injection
- 胸片、皮肤病灶、脑 MRI 等医学影像路由
- FAQ 检索，包括脑肿瘤、胸片/COVID、糖尿病、通用医疗、皮肤病变、心血管、用药安全 7 个领域

难度分布：

- intent routing：easy 17、medium 21、hard 12
- tool call：easy 8、medium 11、hard 6
- FAQ retrieval：easy 19、medium 21、hard 10

## 2. 分层指标

### Intent / Routing 层

指标：

- `intent_accuracy`
- `agent_accuracy`
- `tool_accuracy`
- 按 `difficulty` 分桶准确率

运行：

```bash
python evals/run_eval.py --task intent
python evals/run_eval.py --task tool
python evals/dataset_report.py --dataset evals/datasets/intent_routing_eval.jsonl
python evals/dataset_report.py --dataset evals/datasets/tool_call_eval.jsonl
```

### Tool Call 层

工具调用评测集人工标注真值：

```json
{
  "query": "Analyze this chest x-ray image.",
  "has_image": true,
  "image_type": "CHEST X-RAY",
  "expected_agent": "CHEST_XRAY_AGENT",
  "expected_tool": "chest_xray_classifier"
}
```

评估逻辑：

- router 输出的 `agent_name` 是否等于 `expected_agent`
- router 输出的 `tool_name` 是否等于 `expected_tool`
- 对不应该调用工具的安全请求，`expected_tool` 标注为 `null`

### Retrieval 层

FAQ retrieval 评测集标注 expected FAQ chunk id：

```json
{
  "id": "faq-rag-001",
  "query": "头痛就是脑肿瘤吗？",
  "expected_faq_ids": ["FAQ-BT-001"],
  "expected_domain": "brain_tumor",
  "expected_priority": "P0",
  "expected_source_orgs": ["MedlinePlus"],
  "difficulty": "easy"
}
```

指标：

- `Recall@K`：Top K 是否覆盖真值 FAQ
- `MRR`：第一个相关 FAQ 出现得多靠前
- `nDCG@5`：相关 FAQ 是否排在更靠前的位置
- `domain_top1_accuracy`：Top-1 FAQ 的医学领域是否正确
- `priority_top1_accuracy`：Top-1 FAQ 的 P0/P1/P2 是否正确
- `source_top1_accuracy`：Top-1 FAQ 的来源机构是否正确
- `channel_hit_count`：最终召回由哪些 retrieval channel 贡献

运行：

```bash
python evals/run_faq_retrieval_eval.py --mode keyword
DISABLE_CROSS_ENCODER_RERANKER=true python evals/run_faq_retrieval_eval.py --mode rag
```

当前 seed 结果：

| 模式 | 条数 | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FAQ keyword channel | 50 | 81.00% | 95.00% | 96.00% | 0.933 | 0.926 |
| Full RAG retrieval | 50 | 80.00% | 93.00% | 96.00% | 0.941 | 0.922 |

Full RAG 模式的 channel 命中：

- `faq_keyword_search`：50/50
- `qdrant_global_search`：50/50
- `intent_directed_medical_search`：21/50

### Generation 层

建议继续引入 RAGAS 或人工标注评估：

- `faithfulness`
- `answer_correctness`
- `answer_relevancy`
- refusal correctness
- citation correctness

## 3. RAGAS 指标解释

### faithfulness

Faithfulness 衡量回答是否忠实于给定上下文，重点是有没有编造上下文里没有的内容。

例子：

- Context 只说 “X-ray may show pneumonia-like findings.”
- Answer 说 “X-ray can 100% confirm COVID.”
- 这就是低 faithfulness，因为回答引入了上下文不支持的结论。

### answer_correctness

Answer correctness 衡量回答相对标准答案是否正确。它更关注最终答案是否答对，而不只是是否来自上下文。

区别：

- Faithfulness：答案是否被 context 支撑
- Answer correctness：答案是否和 ground truth 一致

一个回答可能 faithful 但不 correct：如果检索到的 context 本身不完整，模型忠实引用了它，但没答全标准答案。

一个回答也可能 correct 但不 faithful：模型靠自身知识答对了，但答案没有被检索 context 支撑，在 RAG 里仍然是不可追溯风险。

## 4. 基于评测做过的优化

本次 FAQ 检索优化前后：

| 版本 | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 初版 keyword channel | 48.00% | 74.00% | 80.00% | 0.648 | 0.679 |
| 加入 IDF、缩写 boost、停用词过滤后 | 86.00% | 100.00% | 100.00% | 0.953 | 0.965 |
| 扩展到 150 条 FAQ / 50 条更难 seed eval 后 keyword channel | 81.00% | 95.00% | 96.00% | 0.933 | 0.926 |
| 扩展到 150 条 FAQ / 50 条更难 seed eval 后 Full RAG | 80.00% | 93.00% | 96.00% | 0.941 | 0.922 |

优化动作：

- 增加 `faq_keyword_search`，给 FAQ 层一个明确的 patient-question 召回通道
- 用 IDF-style lexical scoring 降低泛领域词影响，提高稀有医学关键词权重
- 对 `A1C`、`MRI`、`CT`、`AI`、`COVID` 等缩写/英文医学词做强匹配 boost
- 增加中文停用词过滤，减少“是什么/什么意思/需要/可以”等泛问法干扰
- 在 postprocessor 里加入 `faq_priority_boost`，让 P0 安全 FAQ 在 rerank 前获得轻量加权

## 5. 面试回答模板

**问题：你的评估集有多少条？意图分布和难度分布怎样？**

> 当前是 seed evaluation：intent routing 50 条、tool call 25 条、FAQ retrieval 50 条，共 125 条。intent 覆盖普通对话、能力询问、越界请求、知识库问答、最新医学进展、症状咨询、安全拒答、prompt injection 和三类医学影像任务。难度上 intent 是 easy 17、medium 21、hard 12；tool call 是 easy 8、medium 11、hard 6；FAQ retrieval 是 easy 19、medium 21、hard 10。

**问题：工具调用准确率怎么评？真值怎么标注？**

> 我在 JSONL 里标注 `expected_agent` 和 `expected_tool`。比如胸片图片问题真值是 `CHEST_XRAY_AGENT + chest_xray_classifier`，最新研究问题真值是 `WEB_SEARCH_PROCESSOR_AGENT + web_search`，安全拒答问题的 `expected_tool` 是 `null`。评测时比较 router 输出的 agent/tool 和真值是否一致。

**问题：Retrieval 怎么评？**

> 我给 FAQ 检索集标注 `expected_faq_ids`、domain、priority 和 source_org，用 Recall@K、MRR、nDCG@5、Top-1 domain/priority/source accuracy 来评估。目前 50 条 FAQ seed 集上，Full RAG retrieval 的 Recall@1 是 80.00%，Recall@5 是 96.00%，MRR 0.941，nDCG@5 0.922。

**问题：RAGAS 的 faithfulness 和 answer_correctness 区别？**

> Faithfulness 看回答是否被检索上下文支撑，主要防幻觉；answer_correctness 看回答和标准答案是否一致，主要看答没答对。RAG 里两者都重要，因为只答对但不基于上下文，仍然不可追溯。
