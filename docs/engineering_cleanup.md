# 工程收敛说明

这一轮主要解决一个工程问题：`agents.rag_agent.__init__` 原本直接承载完整 `MedicalRAG` 类，因此任何子模块导入都会先加载 `doc_parser`、Docling、Qdrant 等重依赖。

这会导致两个问题：

- 纯逻辑测试如 `RAGReActController` 也需要安装 Docling；
- 离线 demo 为了绕开重依赖，只能用动态文件导入，代码不够正式。

## 1. 调整方式

现在结构改为：

```text
agents/rag_agent/
  __init__.py          # 轻量懒加载入口
  medical_rag.py       # 完整 MedicalRAG 编排类
  react_controller.py  # 轻量 ReAct 控制器
  retrieval_channels.py
  postprocessors.py
```

`__init__.py` 只保留兼容代理：

```python
class MedicalRAG:
    def __new__(cls, *args, **kwargs):
        from .medical_rag import MedicalRAG as MedicalRAGImpl
        return MedicalRAGImpl(*args, **kwargs)
```

这样保留了旧用法：

```python
from agents.rag_agent import MedicalRAG
```

导入 `MedicalRAG` 名称本身不会加载 Docling；只有真正实例化 `MedicalRAG(config)` 时才会加载完整实现。同时也支持轻量导入：

```python
from agents.rag_agent.react_controller import RAGReActController
```

## 2. 本地检查入口

新增：

```bash
python scripts/run_local_checks.py
```

默认运行：

- 离线 demo
- intent eval
- tool eval
- dataset report
- metrics test
- model gateway test
- prompt registry test
- rate limiter test
- ReAct controller test
- compileall

也可以只跑某几个：

```bash
python scripts/run_local_checks.py demo intent_eval tool_eval
```

## 3. 面试可讲点

可以这样说：

> 我后面做了一次工程收敛，把 RAG 包的重依赖从 `__init__` 里移出去。原来任何子模块导入都会触发 Docling、Qdrant 等依赖加载，导致纯逻辑测试也要完整环境。现在 `MedicalRAG` 放到 `medical_rag.py`，包入口只做懒加载，既保留原导入兼容，又让 ReAct controller、评测脚本和离线 demo 可以在无模型、无向量库环境下快速运行。
