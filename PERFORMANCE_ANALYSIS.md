# /chat Response Speed Analysis

## Current Status
- **Response time**: 77.2s (down from 101s, -23%)
- **Target**: <30s
- **Status**: Code optimizations exhausted; bottleneck is LLM provider latency

## Timing Breakdown (from server.log, single request)

| Phase | Duration | Description |
|-------|----------|-------------|
| Planner | ~5s | Fails, falls back to PARALLEL_RETRIEVAL |
| RAG LLM | ~30s | xiaomi-mimo API call for RAG agent |
| WebSearch | ~13s | primp/startpage + google search |
| Synthesis LLM | ~12s | Combine RAG + WebSearch results |
| **Total** | **~77s** | POST /chat → 200 OK |

## Root Cause
**xiaomi-mimo API latency = ~20-30s per LLM call.** The pipeline makes 3 sequential LLM calls (planner + RAG + synthesis), each adding 20-30s. Even with RAG+WebSearch running in parallel, the sequential LLM calls dominate.

## Optimizations Applied (3 commits)

1. **d460807** - DDGS timeout fix + RAG-first routing + Planner fix
2. **6e57343** - Executor __exit__ hang fix (ThreadPoolExecutor manual shutdown)
3. **273bef4** - True parallel RAG+WebSearch (submit both simultaneously, cancel on RAG success)

## Options to Reach <30s

1. **Switch LLM provider** - Use a faster API (e.g., deepseek, qwen, zhipu) with <5s latency
2. **Reduce LLM calls** - Skip planner, use keyword-based routing; skip synthesis if RAG confidence is high
3. **Add response caching** - Cache common queries (hash query → cached response)
4. **Use streaming** - /chat/stream (has unpack bug to fix first) gives perceived faster UX
5. **Accept current speed** - 77s is slow but functional

## Known Issues
- /chat/stream: "not enough values to unpack (expected 2, got 1)" - NOT FIXED
- Planner always fails → PARALLEL_RETRIEVAL used for all queries
- WebSearch (primp) is slow (~13s) due to startpage redirect chain
