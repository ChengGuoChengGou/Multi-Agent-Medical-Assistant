# 模型并发限制与排队机制

大模型接口通常有并发数、RPM、TPM 或请求队列限制。如果所有 Agent 都直接调用模型，流量高峰时容易出现：

- provider 限流
- 请求雪崩重试
- 用户等待时间不可控
- 某个耗时 Agent 占满全部模型资源

因此本项目把模型调用统一收敛到 `ModelGateway`，并增加了进程内公平排队限流器。

## 1. 当前实现

核心代码：

- `core/rate_limiter.py`
- `core/model_gateway.py`

当前链路：

```text
Agent / RAG Component
  -> ModelGateway.invoke()
    -> CircuitBreaker.allow_call()
    -> FairRateLimiter.acquire()
    -> RetryPolicy.run(model.invoke)
    -> release permit
    -> TokenUsageEstimate
```

默认每个 `model_id` 都有一个独立 limiter。这样查询扩展、RAG 回答、对话模型可以分别设置独立额度。

## 2. 排队机制怎么保证原子性

`FairRateLimiter` 使用 `Condition + Lock + FIFO deque`：

1. 请求进入后生成 `request_id`，追加到队列尾部。
2. 在同一把锁内判断：
   - 当前请求是不是队首
   - 当前 active permit 是否小于 `max_concurrent`
3. 如果两个条件同时满足，就在同一个临界区里：
   - 从队首移除请求
   - `active += 1`
   - 返回 permit
4. 模型调用结束后释放 permit，并唤醒等待请求。

因为“判断队首、占用许可、出队”在同一把锁里完成，所以不会出现两个线程同时认为自己拿到了同一个名额的问题。

## 3. 超时与失败处理

- 如果请求等待超过 `queue_timeout_seconds`，会从队列中移除并抛出 `RateLimitTimeout`。
- 如果模型调用失败，`with limiter.acquire()` 的 `finally` 会释放 permit，避免死锁。
- `ModelGateway` 会继续把失败交给熔断器统计，达到阈值后进入 OPEN 状态，后续请求快速失败。

每次成功调用会返回限流元数据：

```json
{
  "rate_limit_request_id": "uuid",
  "queue_wait_ms": 17
}
```

这可以用于观测平均排队时间、P95 排队时间和是否需要扩容模型额度。

## 4. 面试回答模板

**问题：大模型并发限制怎么处理的？排队机制怎么保证原子性？**

可以这样回答：

> 我把所有模型调用统一封装到 `ModelGateway`，在网关里做熔断、重试、token 估算和并发限流。限流器按 `model_id` 维度维护一个 FIFO 队列和 active permit 数量，请求进来先入队，只有当自己是队首并且还有可用 permit 时才会出队执行。
>
> 原子性上，我把“判断是否队首、判断是否有容量、出队、active 加一”放在同一个锁保护的临界区里完成，所以不会出现并发请求同时抢到同一个名额。模型调用结束后通过 `finally` 释放 permit 并唤醒后续请求。等待超时的请求会从队列中移除，避免队列里残留无效请求。

## 5. 和 Ragent 的参考关系

Ragent 的生产级方案是分布式公平限流：Redis/Redisson sorted set 排队，Lua 脚本原子 claim，permit semaphore 控并发，Pub/Sub 唤醒等待节点。

本项目当前实现是进程内版本，适合单机 Demo 和面试展示工程意识。两者思路一致：

- 都是 FIFO 排队
- 都区分排队和执行中的 permit
- 都要求 claim permit 的过程原子化
- 都需要超时清理和释放保护

如果要升级成分布式部署，可以把 `FairRateLimiter` 替换为 Redis 版本：

- `deque` -> Redis sorted set
- `Lock` 临界区 -> Lua 脚本
- `active` 计数 -> Redis semaphore
- `Condition.notify_all()` -> Pub/Sub 唤醒
