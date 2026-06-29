# 文档入库链路与失败处理

本项目的知识库不是把 PDF 直接丢给向量数据库，而是拆成可追踪的入库流水线。这样做的目的有两个：

1. 面向业务：医学文档通常包含正文、表格、图像和检查报告截图，需要先结构化解析，再进入检索。
2. 面向工程：当用户反馈“上传成功但搜不到”时，可以定位到底是解析、图片摘要、切分还是索引失败。

## 1. 上传到可检索的完整步骤

当前 `MedicalRAG.ingest_file()` 将单篇文档拆成 5 个节点：

| 节点 | 代码枚举 | 作用 | 输出摘要 |
| --- | --- | --- | --- |
| 文档解析 | `parse_document` | 使用 Docling 解析 PDF，抽取页面、正文结构、表格和图片 | `images_extracted` |
| 图片摘要 | `summarize_images` | 用多模态模型为医学图片、图表、检查截图生成简短描述 | `image_summaries_generated` |
| 文档格式化 | `format_document` | 将图片占位符替换成图片摘要，形成统一 Markdown 文本 | `formatted_characters` |
| 语义切分 | `chunk_document` | 先按标题粗切，再用 LLM 判断语义边界 | `chunks_created` |
| 向量索引 | `index_vectorstore` | 写入 Qdrant hybrid vector store，同时将原 chunk 写入本地 docstore | `chunks_indexed` |

目录入库 `MedicalRAG.ingest_directory()` 会逐个文件调用单文件入库，并保留每个文件的 `ingestion_trace`，最终汇总：

- `documents_ingested`
- `failed_documents`
- `failed_files`
- `chunks_processed`
- `file_results`

## 2. Trace 结构

每次入库都会生成一个 `IngestionPipelineTrace`：

```json
{
  "task_id": "uuid",
  "source": "path/to/document.pdf",
  "status": "success",
  "duration_seconds": 12.34,
  "error": null,
  "nodes": [
    {
      "node_id": "01-parse_document",
      "node_type": "parse_document",
      "status": "success",
      "duration_seconds": 2.1,
      "error": null,
      "output_summary": {
        "images_extracted": 4
      }
    }
  ]
}
```

这份 trace 可以直接返回给后台管理端，也可以写入数据库作为任务审计日志。

## 3. 失败处理策略

当前实现是“单文件内快速失败，目录内隔离失败”：

- 单文件：任意关键节点失败，整篇文档标记为 `failed`，返回失败节点和错误信息。
- 目录：某篇文档失败不会阻断后续文档，失败文件进入 `failed_files`。
- 图片摘要：单张图片失败时已有降级逻辑，填充 `"no image summary"`，不阻断整篇文档。
- 向量索引：索引失败时该文档不计入成功入库，避免出现“文档未完整写入但显示成功”的状态。

后续如果要更接近生产系统，可以把 trace 写入持久化任务表，并增加重试队列：

- 解析失败：提示用户文件损坏、加密或格式不支持。
- 图片摘要失败：允许跳过图片摘要，仅用文本入库。
- 切分失败：降级为规则切分，例如按标题或固定 token 长度切分。
- 索引失败：保留已解析结果，后续只重试向量写入，不重复解析原 PDF。

## 4. 面试回答模板

**问题：文档从上传到可检索经过哪些步骤？某个步骤失败了怎么处理？**

可以这样回答：

> 我把知识库入库做成了一个可观测的 pipeline，而不是简单地 `parse -> embed`。一篇医学文档进入系统后，先用 Docling 做结构化解析，抽取正文、表格和图片；然后对医学图片做摘要，把图片信息补回 Markdown 文本；接着做语义切分，生成适合检索的 chunk；最后写入 Qdrant hybrid vector store，并把原文 chunk 写入本地 docstore。每个节点都有独立的状态、耗时、输出摘要和错误信息。
>
> 失败处理上，单文件内部采用快速失败，避免半成功数据污染知识库；目录批量入库时采用文件级隔离，一篇失败不会影响其他文件。图片摘要属于非强依赖，单张图失败会降级成占位摘要继续入库；索引失败则整篇文档标记失败。这样用户反馈“搜不到文档”时，我可以根据 trace 判断是解析失败、切分失败还是向量库写入失败。

## 5. 和 Ragent 的参考关系

Ragent 的完整性体现在它把 RAG 当成工程流水线来做：多阶段检索、模型网关、健康状态、链路追踪和可评测闭环。这个项目参考了这种思路，把原来的单函数入库改成了节点化 pipeline。虽然当前还没有引入数据库任务表和后台任务队列，但已经具备后续扩展所需的结构：

- 节点状态枚举
- 任务级 trace
- 文件级失败隔离
- 可序列化输出
- 可落库的任务审计字段
