# Entity Extraction Caching for Mem0 Graph Search

## Problem
Every `mem0.search()` call with graph store enabled triggers an **LLM call** to extract entities, adding **300-1500ms latency**. This happens in `_retrieve_nodes_from_data()` across all graph implementations.

## Solution
Add a caching layer for entity extraction results. Since entity extraction is deterministic for the same input, caching eliminates redundant LLM calls.

**Expected improvement**: 99%+ latency reduction on cache hits (from 300-1500ms to <1ms)

---

## Implementation Plan

### Phase 1: Configuration Classes

**Create `mem0/configs/cache/` directory with:**

| File | Description |
|------|-------------|
| `__init__.py` | Export cache configs |
| `base.py` | `BaseCacheConfig` - ttl (default: 3600s), enabled, key_prefix |
| `memory.py` | `MemoryCacheConfig` - extends base, adds max_size (default: 1000) |
| `redis.py` | `RedisCacheConfig` - extends base, adds redis_url, pool_size |
| `config.py` | `CacheConfig` wrapper with provider (default: "redis") + config |

**Modify `mem0/configs/base.py:54`:**
```python
# After reranker field (line 51-54), add:
cache: Optional[CacheConfig] = Field(
    description="Configuration for entity extraction caching",
    default=None,
)
```

---

### Phase 2: Cache Backends

**Create `mem0/cache/` directory with:**

| File | Description |
|------|-------------|
| `__init__.py` | Export `MemoryCache`, `RedisCache`, `BaseCacheBackend` |
| `base.py` | Abstract base class with `generate_cache_key()`, `get()`, `set()`, `delete()`, `clear()` + async variants |
| `memory_cache.py` | Thread-safe LRU cache with TTL using `OrderedDict` + `threading.RLock` |
| `redis_cache.py` | Redis backend with sync/async support using existing `redis` dependency |

**Cache Key Generation** (in `base.py`):
```python
@staticmethod
def generate_cache_key(data: str, user_id: str, llm_provider: str,
                       agent_id: str = None, run_id: str = None) -> str:
    key_parts = {
        "data": data.lower().strip(),
        "user_id": user_id,
        "llm_provider": llm_provider,
    }
    if agent_id: key_parts["agent_id"] = agent_id
    if run_id: key_parts["run_id"] = run_id
    return hashlib.sha256(json.dumps(key_parts, sort_keys=True).encode()).hexdigest()
```

---

### Phase 3: Factory Integration

**Modify `mem0/utils/factory.py` (after line ~284):**

Add `CacheFactory` class following existing pattern:
```python
class CacheFactory:
    provider_to_class = {
        "memory": ("mem0.cache.memory_cache.MemoryCache", "mem0.configs.cache.memory.MemoryCacheConfig"),
        "redis": ("mem0.cache.redis_cache.RedisCache", "mem0.configs.cache.redis.RedisCacheConfig"),
    }

    @classmethod
    def create(cls, provider_name: str, config: Optional[dict] = None):
        # Load class, validate config, return instance
```

---

### Phase 4: Graph Memory Integration

**Modify each graph implementation:**

| File | Line | Change |
|------|------|--------|
| `mem0/memory/graph_memory.py` | 74 | Add `self.entity_cache = None` in `__init__` |
| `mem0/memory/graph_memory.py` | 196-227 | Wrap `_retrieve_nodes_from_data()` with cache check/set |
| `mem0/memory/memgraph_memory.py` | ~80 | Add `self.entity_cache = None` in `__init__` |
| `mem0/memory/memgraph_memory.py` | 199-230 | Wrap `_retrieve_nodes_from_data()` with cache |
| `mem0/memory/kuzu_memory.py` | ~67 | Add `self.entity_cache = None` in `__init__` |
| `mem0/memory/kuzu_memory.py` | 222-253 | Wrap `_retrieve_nodes_from_data()` with cache |

**Modified `_retrieve_nodes_from_data()` pattern:**
```python
def _retrieve_nodes_from_data(self, data, filters):
    # Cache check
    if self.entity_cache and self.entity_cache.enabled:
        cache_key = BaseCacheBackend.generate_cache_key(
            data=data, user_id=filters.get('user_id', ''),
            llm_provider=self.llm_provider,
            agent_id=filters.get('agent_id'), run_id=filters.get('run_id')
        )
        cached = self.entity_cache.get(cache_key)
        if cached is not None:
            return cached

    # Original LLM extraction logic (lines 198-227)
    # ...

    # Cache result
    if self.entity_cache and self.entity_cache.enabled and entity_type_map:
        self.entity_cache.set(cache_key, entity_type_map)

    return entity_type_map
```

---

### Phase 5: Memory Class Integration

**Modify `mem0/memory/main.py`:**

| Line | Change |
|------|--------|
| ~17 | Add import: `from mem0.configs.cache.config import CacheConfig` |
| ~197-206 | Initialize cache and inject into graph store |

```python
# After reranker init (~line 197)
self.entity_cache = None
if config.cache:
    from mem0.utils.factory import CacheFactory
    self.entity_cache = CacheFactory.create(config.cache.provider, config.cache.config)

# When creating graph store (~line 202-206)
if self.config.graph_store.config:
    self.graph = GraphStoreFactory.create(provider, self.config)
    if self.entity_cache:
        self.graph.entity_cache = self.entity_cache  # Inject cache
    self.enable_graph = True
```

---

## Files to Create

```
mem0/
  cache/
    __init__.py
    base.py
    memory_cache.py
    redis_cache.py
  configs/
    cache/
      __init__.py
      base.py
      memory.py
      redis.py
      config.py

tests/
  cache/
    __init__.py
    test_memory_cache.py
    test_redis_cache.py
    test_cache_factory.py
    test_cache_integration.py
```

## Files to Modify

| File | Changes |
|------|---------|
| `mem0/configs/base.py:54` | Add `cache` field to `MemoryConfig` |
| `mem0/utils/factory.py:284+` | Add `CacheFactory` class |
| `mem0/memory/main.py:197-206` | Initialize cache, inject to graph |
| `mem0/memory/graph_memory.py:74,196-227` | Add cache attribute + wrap extraction |
| `mem0/memory/memgraph_memory.py` | Same pattern as graph_memory |
| `mem0/memory/kuzu_memory.py` | Same pattern as graph_memory |

---

## Usage Example

```python
from mem0 import Memory

# In-memory cache (development)
config = {
    "graph_store": {"provider": "neo4j", "config": {...}},
    "cache": {
        "provider": "memory",
        "config": {"ttl": 3600, "max_size": 1000}
    }
}

# Redis cache (production)
config = {
    "graph_store": {"provider": "neo4j", "config": {...}},
    "cache": {
        "provider": "redis",
        "config": {"redis_url": "redis://localhost:6379", "ttl": 1800}
    }
}

memory = Memory.from_config(config)
```

---

## Phase 6: Tests

**Create `tests/cache/` directory with:**

| File | Description |
|------|-------------|
| `__init__.py` | Test package init |
| `test_memory_cache.py` | Unit tests for in-memory LRU cache (TTL, eviction, thread-safety) |
| `test_redis_cache.py` | Unit tests for Redis cache with mocked Redis client |
| `test_cache_factory.py` | Factory creation tests for both providers |
| `test_cache_integration.py` | Integration tests with graph memory (cache hit/miss verification) |

**Test Coverage:**
- Cache key generation determinism
- TTL expiration behavior
- LRU eviction (memory cache)
- Thread-safety under concurrent access
- Cache hit/miss metrics
- Graceful degradation when cache unavailable

---

## Notes

- `CacheError` exception already exists at `mem0/exceptions.py:286-300`
- `redis` package already in dependencies (`pyproject.toml:54`)
- TTL-based invalidation is sufficient since entity extraction is deterministic
- Thread-safe implementation required (concurrent searches in ThreadPoolExecutor)
- **Default provider**: Redis (user preference)
- **Scope**: Entity extraction only (embedding cache can be added later)
