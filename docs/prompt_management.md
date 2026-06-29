# Prompt 模板管理与效果回退

原项目里 Prompt 主要散落在 Python 字符串中，能跑，但不方便回答生产场景里的几个问题：

- 当前线上用的是哪一版 Prompt？
- 某次评测下降，是模型变了还是 Prompt 变了？
- 新 Prompt 效果不好时，怎么快速回退？

因此本项目增加了一个轻量 Prompt Registry。

## 1. 目录结构

```text
prompts/
  manifest.json
  rag/
    query_expansion.v1.txt
    response_generation.v1.txt
core/
  prompt_registry.py
```

`manifest.json` 负责声明每个 Prompt 的 active version：

```json
{
  "prompts": {
    "rag.response_generation": {
      "active_version": "v1",
      "versions": {
        "v1": {
          "path": "rag/response_generation.v1.txt"
        }
      }
    }
  }
}
```

业务代码只依赖 prompt id，例如：

```python
prompt_template = self.prompt_registry.get("rag.response_generation")
prompt = prompt_template.render(query=query, context=context, chat_history=chat_history)
```

## 2. 当前已纳管的 Prompt

| Prompt ID | 使用位置 | 作用 |
| --- | --- | --- |
| `rag.query_expansion` | `QueryExpander` | 给医学问题补充同义词、医学术语和检索关键词 |
| `rag.response_generation` | `ResponseGenerator` | 基于检索上下文生成有依据的医学回答 |

每次模型调用都会通过 `ModelGateway` 携带 Prompt 元数据：

- `prompt_id`
- `prompt_version`
- `prompt_path`
- `stage`

这样评测日志或线上日志可以按 Prompt 版本聚合，比如比较 `rag.response_generation:v1` 和 `v2` 的 faithfulness、answer correctness、拒答率、平均 token 成本。

## 3. 版本控制与回退方式

新增 Prompt 版本时，不覆盖旧文件，而是新增文件：

```text
prompts/rag/response_generation.v2.txt
```

然后在 `manifest.json` 中加入版本，并把 `active_version` 从 `v1` 改为 `v2`。

如果新版本评测变差，回退只需要把 `active_version` 改回 `v1`。由于 Prompt 文件和 manifest 都在 Git 中，代码评审时能清楚看到：

- Prompt 文本改了什么
- 哪个版本被设置为 active
- 这次改动对应哪组评测结果

## 4. 面试回答模板

**问题：Prompt 模板怎么管理？怎么做版本控制和效果回退？**

可以这样回答：

> 我没有把核心 Prompt 直接写死在业务代码里，而是抽成了 `prompts/manifest.json + prompt txt 文件` 的形式。业务代码只通过 `prompt_id` 获取 active version。每次调用模型时，`ModelGateway` 会把 `prompt_id`、`prompt_version`、`prompt_path` 写进 metadata，后续评测或线上日志都能按 Prompt 版本聚合。
>
> 版本控制上，每次优化 Prompt 都新增一个版本文件，例如 `response_generation.v2.txt`，旧版本不覆盖。灰度或评测通过后，只改 manifest 的 `active_version`。如果发现 faithfulness、answer_correctness 或拒答率变差，就把 active version 改回上一版即可。这样 Prompt 的上线和回退都可以通过 Git diff 审核，不依赖口头记录。

## 5. 和 Ragent 的参考关系

Ragent 项目里 Prompt 是独立资源文件，并通过工程配置组合到模型调用链路中。本项目参考这一点，把 Prompt 从 Python 代码中剥离出来，先用最小实现建立：

- 统一入口
- 独立文件
- active version
- 调用 metadata
- Git 可回滚

后续可以继续扩展成 Prompt A/B Test、Prompt 评测报表和按场景动态选择 Prompt。
