# Mem0 Search Function Deep Dive

## Executive Summary

This report provides a comprehensive analysis of how the `mem0.search()` function works, with particular focus on the graph database integration and LLM usage. **Mem0 uses an LLM to extract entities from natural language queries**, which enables semantic understanding of user intent. This entity extraction approach, while powerful, **introduces latency overhead** as each search requires an LLM call before the graph database can be queried.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Performance Considerations](#performance-considerations)
3. [Search Entry Point](#search-entry-point)
4. [Parallel Execution Model](#parallel-execution-model)
5. [Graph Search Pipeline](#graph-search-pipeline)
6. [Entity Extraction with LLM](#entity-extraction-with-llm)
7. [Real-World Example: Production Data](#real-world-example-production-data)
8. [Vector-Based Graph Queries](#vector-based-graph-queries)
9. [Supported Graph Databases](#supported-graph-databases)
10. [Code Examples](#code-examples)
11. [Flow Diagrams](#flow-diagrams)
12. [Key Files Reference](#key-files-reference)

---

## Architecture Overview

Mem0's search functionality operates on two parallel tracks:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Memory.search(query)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │     ThreadPoolExecutor        │
              │     (Parallel Execution)      │
              └───────────────┬───────────────┘
                              │
         ┌────────────────────┴────────────────────┐
         │                                         │
         ▼                                         ▼
┌─────────────────────┐                 ┌─────────────────────┐
│  Vector Store       │                 │  Graph Store        │
│  Search             │                 │  Search             │
│                     │                 │                     │
│  - Embed query      │                 │  - LLM extracts     │
│  - Cosine similarity│                 │    entities (SLOW)  │
│  - Return memories  │                 │  - Embed entities   │
│                     │                 │  - Query graph DB   │
└─────────────────────┘                 └─────────────────────┘
         │                                         │
         └────────────────────┬────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Merge Results  │
                    │  & Rerank       │
                    └─────────────────┘
                              │
                              ▼
              {"results": [...], "relations": [...]}
```

---

## Performance Considerations

### LLM Entity Extraction Overhead

**Every graph search requires an LLM call** to extract entities from the query. This adds significant latency:

| Operation | Typical Latency | Notes |
|-----------|-----------------|-------|
| LLM Entity Extraction | **300-1500ms** | Depends on model (gpt-4o-mini vs gpt-4) |
| Embedding Generation | 50-200ms | Per entity extracted |
| Graph Database Query | 10-100ms | Vector similarity search |
| BM25 Reranking | 1-10ms | Local computation |

**Total graph search time: 400-2000ms+** (dominated by LLM call)

### Why This Matters

```python
# Each search() call triggers:
# 1. Vector store search (fast, ~100ms)
# 2. Graph store search (slow due to LLM, ~500-1500ms)
#    └── LLM call to extract entities (the bottleneck)
#    └── Embedding generation
#    └── Cypher query execution

# The searches run in parallel, so total time = max(vector_time, graph_time)
# But graph search is almost always slower due to mandatory LLM call
```

### Optimization Strategies

1. **Use faster models**: `gpt-4o-mini` instead of `gpt-4` for entity extraction
2. **Disable graph store**: If you don't need relationship data, set `enable_graph=False`
3. **Cache entities**: For repeated similar queries, entity caching could help
4. **Batch searches**: Process multiple queries together when possible

---

## Search Entry Point

**File**: `mem0/memory/main.py` (lines 758-856)

### Method Signature

```python
def search(
    self,
    query: str,
    *,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    limit: int = 100,
    filters: Optional[Dict[str, Any]] = None,
    threshold: Optional[float] = None,
    rerank: bool = True,
):
```

### Parameters Explained

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Natural language search query |
| `user_id` | `str` | Scope search to specific user |
| `agent_id` | `str` | Scope search to specific agent |
| `run_id` | `str` | Scope search to specific session/run |
| `limit` | `int` | Maximum results to return (default: 100) |
| `filters` | `dict` | Additional metadata filters |
| `threshold` | `float` | Minimum similarity score |
| `rerank` | `bool` | Whether to rerank results (default: True) |

### Example Usage

```python
from mem0 import Memory

# Initialize with graph store enabled
config = {
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "password"
        }
    }
}

memory = Memory.from_config(config)

# Search with user scope
results = memory.search(
    query="What food does Alice like?",
    user_id="user_123",
    limit=10
)

# Results structure when graph is enabled:
# {
#     "results": [
#         {"id": "mem_1", "memory": "Alice loves Italian food", "score": 0.92},
#         {"id": "mem_2", "memory": "Alice prefers pasta over rice", "score": 0.87}
#     ],
#     "relations": [
#         {"source": "alice", "relationship": "LIKES", "destination": "italian_food"},
#         {"source": "alice", "relationship": "PREFERS", "destination": "pasta"}
#     ]
# }
```

---

## Parallel Execution Model

**File**: `mem0/memory/main.py` (lines 832-843)

Mem0 uses `concurrent.futures.ThreadPoolExecutor` to run vector and graph searches simultaneously:

```python
with concurrent.futures.ThreadPoolExecutor() as executor:
    # Submit vector store search
    future_memories = executor.submit(
        self._search_vector_store,
        query,
        effective_filters,
        limit,
        threshold
    )

    # Submit graph store search (if enabled)
    # NOTE: This triggers an LLM call for entity extraction!
    future_graph_entities = (
        executor.submit(self.graph.search, query, effective_filters, limit)
        if self.enable_graph
        else None
    )

    # Wait for both to complete
    concurrent.futures.wait(
        [future_memories, future_graph_entities]
        if future_graph_entities
        else [future_memories]
    )

    # Collect results
    original_memories = future_memories.result()
    graph_entities = future_graph_entities.result() if future_graph_entities else None
```

### Async Version

For async code, `asyncio.gather()` is used instead:

```python
# mem0/memory/main.py (lines 1807-1913)
vector_store_task = asyncio.create_task(
    self._search_vector_store(query, effective_filters, limit, threshold)
)

if self.enable_graph:
    graph_task = asyncio.create_task(
        asyncio.to_thread(self.graph.search, query, effective_filters, limit)
    )
    original_memories, graph_entities = await asyncio.gather(
        vector_store_task, graph_task
    )
```

---

## Graph Search Pipeline

The graph search pipeline involves **LLM-based entity extraction** as a critical first step:

```
┌──────────────────────────────────────────────────────────────────┐
│                    graph.search(query, filters)                  │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 1: _retrieve_nodes_from_data(query, filters)               │
│  ⚠️  LLM CALL - ADDS LATENCY                                     │
│                                                                  │
│  Uses LLM + EXTRACT_ENTITIES_TOOL to understand the query        │
│                                                                  │
│  Input:  "What food does Alice like?"                            │
│  Output: {"alice": "person", "food": "category"}                 │
│                                                                  │
│  Time: 300-1500ms depending on model                             │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 2: _search_graph_db(node_list, filters, limit)             │
│                                                                  │
│  For each extracted entity:                                      │
│    1. Generate embedding for entity name                         │
│    2. Execute Cypher query with vector similarity                │
│    3. Return matching nodes and their relationships              │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 3: Rerank with BM25                                        │
│                                                                  │
│  Reorder results based on query relevance                        │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                         Return relations
```

---

## Entity Extraction with LLM

**This is the core of mem0's graph search intelligence.** The LLM extracts semantic entities from natural language, enabling the system to understand what the user is asking about.

**File**: `mem0/graphs/graph_memory.py` (lines 196-227)

### The Code

```python
def _retrieve_nodes_from_data(self, data, filters):
    """Extracts all the entities mentioned in the query.

    NOTE: This method calls the LLM, which adds latency to every search.
    """

    # Select the appropriate tool based on LLM provider
    _tools = [EXTRACT_ENTITIES_TOOL]
    if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
        _tools = [EXTRACT_ENTITIES_STRUCT_TOOL]

    # Call LLM with entity extraction tool
    # ⚠️ THIS IS THE LATENCY BOTTLENECK
    search_results = self.llm.generate_response(
        messages=[
            {
                "role": "system",
                "content": f"""You are a smart assistant who understands entities
                and their types in a given text. If user message contains self
                reference such as 'I', 'me', 'my' etc. then use {filters['user_id']}
                as the source entity. Extract all the entities from the text.
                ***DO NOT*** answer the question itself if the given text is a question.""",
            },
            {"role": "user", "content": data},
        ],
        tools=_tools,
    )

    # Parse the tool call response
    entity_type_map = {}
    try:
        for tool_call in search_results["tool_calls"]:
            if tool_call["name"] != "extract_entities":
                continue
            for item in tool_call["arguments"]["entities"]:
                entity_type_map[item["entity"]] = item["entity_type"]
    except Exception as e:
        logger.exception(f"Error in search tool: {e}")

    # Normalize entity names (lowercase, underscores)
    entity_type_map = {
        k.lower().replace(" ", "_"): v.lower().replace(" ", "_")
        for k, v in entity_type_map.items()
    }

    return entity_type_map
```

### The Tool Definition

**File**: `mem0/graphs/tools.py` (lines 124-150)

```python
EXTRACT_ENTITIES_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_entities",
        "description": "Extract entities and their types from the text.",
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity": {
                                "type": "string",
                                "description": "The name or identifier of the entity."
                            },
                            "entity_type": {
                                "type": "string",
                                "description": "The type or category of the entity."
                            },
                        },
                        "required": ["entity", "entity_type"],
                    },
                }
            },
            "required": ["entities"],
        },
    },
}
```

### Example: Entity Extraction in Action

**Input Query**: `"What programming languages does John know?"`

**LLM Tool Call Response**:
```json
{
    "tool_calls": [
        {
            "name": "extract_entities",
            "arguments": {
                "entities": [
                    {"entity": "John", "entity_type": "person"},
                    {"entity": "programming languages", "entity_type": "skill_category"}
                ]
            }
        }
    ]
}
```

**Normalized Output**:
```python
{
    "john": "person",
    "programming_languages": "skill_category"
}
```

---

## Real-World Example: Production Data

The following is a real example from the **mem0.ai hosted platform** showing how relationships are stored and retrieved. This data comes from the `app.mem0.ai/dashboard/graph-memory` interface:

### Sample Relations from Production

| Source Name | Source Type | Relationship | Target Name | Target Type |
|-------------|-------------|--------------|-------------|-------------|
| `senior_+16478308908` | unknown | `met` | `lions` | animal |
| `senior_+16478308908` | unknown | `shared_story_about` | `adventure` | interest |
| `senior_+16478308908` | unknown | `planning` | `safari_adventure` | activity |
| `senior_+16478308908` | unknown | `planning` | `safari` | activity |
| `senior_+16478308908` | unknown | `interested_in` | `tours` | activity |
| `senior_+16478308908` | unknown | `interested_in` | `destinations` | location |
| `senior_+16478308908` | unknown | `interested_in` | `inspirational_literature` | genre |
| `senior_+16478308908` | unknown | `works_at` | `three_jobs` | job |
| `senior_+16478308908` | unknown | `feels_overwhelmed_with` | `work` | emotion |
| `senior_+16478308908` | unknown | `enjoys` | `pickleball` | sport |
| `senior_+16478308908` | unknown | `prefers` | `rallying` | activity |
| `senior_+16478308908` | unknown | `dislikes` | `playing_for_score` | activity |
| `senior_+16478308908` | unknown | `spends_time_with` | `partner` | person |
| `senior_+16478308908` | unknown | `is_spouse_of` | `evelyn` | person |
| `senior_+16478308908` | unknown | `has_interest_in` | `books_about_turtles` | interest |
| `senior_+16478308908` | unknown | `prefers` | `short_responses` | unknown |
| `senior_+16478308908` | unknown | `tends_to_answer_with` | `single_words` | unknown |
| `senior_+16478308908` | unknown | `associated_with` | `totals_alliance` | organization |
| `senior_+16478308908` | unknown | `finds_too_bright` | `lamps` | object |
| `exciting` | unknown | `is_type_of` | `experience` | event |

### How Search Uses These Relations

When you search: `"What does the user enjoy doing?"`

1. **LLM extracts entities**: `{"user": "person", "enjoy": "action"}`
2. **System maps user_id**: `senior_+16478308908`
3. **Graph query finds**: Relationships where source matches and relationship contains "enjoy"
4. **Returns**: `{"source": "senior_+16478308908", "relationship": "enjoys", "destination": "pickleball"}`

### Graph Visualization

```
                              ┌─────────────┐
                              │   safari    │
                              │ (activity)  │
                              └──────▲──────┘
                                     │ planning
                                     │
┌─────────────┐  is_spouse_of  ┌─────┴─────────────┐  enjoys     ┌─────────────┐
│   evelyn    │◄───────────────│ senior_+164783...│────────────►│ pickleball  │
│  (person)   │                │    (user)         │             │  (sport)    │
└─────────────┘                └─────┬─────────────┘             └─────────────┘
                                     │
                                     │ interested_in
                                     ▼
                              ┌─────────────┐
                              │   tours     │
                              │ (activity)  │
                              └─────────────┘
```

### Relationship Types Observed

The LLM extracts diverse relationship types that capture human experiences:

- **Actions**: `met`, `enjoys`, `dislikes`, `prefers`
- **States**: `feels_overwhelmed_with`, `finds_too_bright`
- **Associations**: `works_at`, `is_spouse_of`, `associated_with`
- **Interests**: `interested_in`, `has_interest_in`, `planning`
- **Communication**: `tends_to_answer_with`, `shared_story_about`

---

## Vector-Based Graph Queries

**File**: `mem0/graphs/graph_memory.py` (lines 271-320)

After the LLM extracts entities, the code performs vector similarity search in the graph using pre-defined Cypher query templates:

### Neo4j Implementation

```python
def _search_graph_db(self, node_list, filters, limit=100):
    """Search similar nodes and their relationships."""
    result_relations = []

    for node in node_list:
        # Step 1: Embed the entity name
        n_embedding = self.embedding_model.embed(node)

        # Step 2: Cypher query with vector similarity
        cypher_query = f"""
        MATCH (n {self.node_label} {{{node_props_str}}})
        WHERE n.embedding IS NOT NULL
        WITH n, round(
            2 * vector.similarity.cosine(n.embedding, $n_embedding) - 1,
            4
        ) AS similarity
        WHERE similarity >= $threshold
        CALL {{
            WITH n
            MATCH (n)-[r]->(m {self.node_label} {{{node_props_str}}})
            RETURN n.name AS source,
                   elementId(n) AS source_id,
                   type(r) AS relationship,
                   m.name AS destination
            UNION
            WITH n
            MATCH (n)<-[r]-(m {self.node_label} {{{node_props_str}}})
            RETURN m.name AS source,
                   elementId(m) AS source_id,
                   type(r) AS relationship,
                   n.name AS destination
        }}
        RETURN source, source_id, relationship, destination, similarity
        ORDER BY similarity DESC
        LIMIT $limit
        """

        # Step 3: Execute query
        params = {
            "n_embedding": n_embedding,
            "threshold": self.threshold,
            "user_id": filters["user_id"],
            "limit": limit,
        }

        ans = self.graph.query(cypher_query, params=params)
        result_relations.extend(ans)

    return result_relations
```

### Query Breakdown

| Part | Purpose |
|------|---------|
| `MATCH (n ...)` | Find nodes with user_id filter |
| `WHERE n.embedding IS NOT NULL` | Only nodes with embeddings |
| `vector.similarity.cosine()` | Calculate cosine similarity |
| `WHERE similarity >= $threshold` | Filter by minimum similarity |
| `MATCH (n)-[r]->(m)` | Get outgoing relationships |
| `MATCH (n)<-[r]-(m)` | Get incoming relationships |
| `ORDER BY similarity DESC` | Rank by relevance |

---

## Supported Graph Databases

### Neo4j (`mem0/graphs/graph_memory.py`)

```python
# Uses langchain_neo4j.Neo4jGraph
from langchain_neo4j import Neo4jGraph

self.graph = Neo4jGraph(
    url=config.url,
    username=config.username,
    password=config.password,
    database=config.database,
)

# Vector search using built-in function
vector.similarity.cosine(n.embedding, $n_embedding)
```

### Memgraph (`mem0/graphs/memgraph_memory.py`)

```python
# Uses langchain_memgraph.Memgraph
from langchain_memgraph import Memgraph

# Vector search using MAGE module
cypher_query = """
CALL vector_search.search("memzero", $limit, $n_embedding)
YIELD node, similarity
"""
```

### Kuzu (`mem0/graphs/kuzu_memory.py`)

```python
# Uses embedded Kuzu database
import kuzu

# Vector search using array function
WHERE array_cosine_similarity(n.embedding, $embedding) >= $threshold
```

### Neptune (AWS)

Uses OpenCypher queries compatible with AWS Neptune graph database.

---

## Code Examples

### Example 1: Basic Search with Graph

```python
from mem0 import Memory
import time

# Configuration with Neo4j
config = {
    "llm": {
        "provider": "openai",
        "config": {"model": "gpt-4o-mini"}  # Use faster model to reduce latency
    },
    "embedder": {
        "provider": "openai",
        "config": {"model": "text-embedding-3-small"}
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "password"
        }
    }
}

memory = Memory.from_config(config)

# Add some memories first
memory.add(
    "Alice is a software engineer who loves Python and Italian food",
    user_id="user_123"
)
memory.add(
    "Alice is working on a machine learning project",
    user_id="user_123"
)

# Search (note: will include LLM call latency)
start = time.time()
results = memory.search(
    query="What does Alice work on?",
    user_id="user_123"
)
print(f"Search took: {time.time() - start:.2f}s")  # Typically 0.5-2s

print(results)
# Output:
# {
#     "results": [
#         {
#             "id": "...",
#             "memory": "Alice is working on a machine learning project",
#             "score": 0.94
#         }
#     ],
#     "relations": [
#         {"source": "alice", "relationship": "WORKS_ON", "destination": "machine_learning_project"},
#         {"source": "alice", "relationship": "IS_A", "destination": "software_engineer"}
#     ]
# }
```

### Example 2: Understanding the LLM Entity Extraction

```python
# When you call memory.search("What does Alice work on?", user_id="user_123")
#
# STEP 1: The LLM receives this prompt:

messages = [
    {
        "role": "system",
        "content": """You are a smart assistant who understands entities
        and their types in a given text. If user message contains self
        reference such as 'I', 'me', 'my' etc. then use user_123 as the
        source entity. Extract all the entities from the text.
        ***DO NOT*** answer the question itself if the given text is a question."""
    },
    {
        "role": "user",
        "content": "What does Alice work on?"
    }
]

# STEP 2: The LLM responds with a structured tool call:
{
    "tool_calls": [
        {
            "name": "extract_entities",
            "arguments": {
                "entities": [
                    {"entity": "Alice", "entity_type": "person"}
                ]
            }
        }
    ]
}

# STEP 3: mem0 uses the extracted entity "alice" to:
#   a. Generate an embedding vector for "alice"
#   b. Run a Cypher query to find nodes similar to "alice"
#   c. Return all relationships connected to matching nodes
```

### Example 3: Self-Reference Handling

```python
# When the query contains self-references like "I", "me", "my"
results = memory.search(
    query="What do I like to eat?",
    user_id="john_doe"
)

# The LLM receives this system prompt:
# "If user message contains self reference such as 'I', 'me', 'my' etc.
#  then use john_doe as the source entity"

# LLM extracts (replacing "I" with user_id):
{
    "entities": [
        {"entity": "john_doe", "entity_type": "person"},
        {"entity": "eat", "entity_type": "action"}
    ]
}
```

### Example 4: Measuring LLM Overhead

```python
import time

# Search WITHOUT graph (fast)
memory_no_graph = Memory.from_config({
    "vector_store": {"provider": "qdrant", ...}
    # No graph_store configured
})

start = time.time()
results = memory_no_graph.search("What does Alice like?", user_id="user_123")
print(f"Without graph: {time.time() - start:.3f}s")  # ~0.1-0.3s

# Search WITH graph (slower due to LLM call)
memory_with_graph = Memory.from_config({
    "vector_store": {"provider": "qdrant", ...},
    "graph_store": {"provider": "neo4j", ...}
})

start = time.time()
results = memory_with_graph.search("What does Alice like?", user_id="user_123")
print(f"With graph: {time.time() - start:.3f}s")  # ~0.5-2.0s

# The difference is primarily the LLM entity extraction call
```

---

## Flow Diagrams

### Complete Search Flow with Timing

```
User Query: "What programming languages does Alice know?"
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                    Memory.search()                            │
│                    mem0/memory/main.py:758                    │
└───────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│   Vector Store Search   │     │   Graph Store Search    │
│   (Thread 1)            │     │   (Thread 2)            │
│   ~100-300ms            │     │   ~500-2000ms           │
│                         │     │                         │
│   1. Embed query        │     │   1. ⚠️ LLM extracts:   │
│   2. Search Qdrant/etc  │     │      (300-1500ms)       │
│   3. Return memories    │     │      - "alice": person  │
│                         │     │      - "programming     │
└─────────────────────────┘     │         languages": cat │
            │                   │   2. Embed "alice"      │
            │                   │   3. Run Cypher query   │
            │                   │   4. Return relations   │
            │                   └─────────────────────────┘
            │                               │
            └───────────────┬───────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                      Merge & Rerank                           │
│                                                               │
│   results = [vector store memories]                           │
│   relations = [graph relationships]                           │
└───────────────────────────────────────────────────────────────┘
                            │
                            ▼
{
    "results": [
        {"memory": "Alice knows Python, JavaScript, and Rust", "score": 0.91}
    ],
    "relations": [
        {"source": "alice", "relationship": "KNOWS", "destination": "python"},
        {"source": "alice", "relationship": "KNOWS", "destination": "javascript"},
        {"source": "alice", "relationship": "KNOWS", "destination": "rust"}
    ]
}
```

### Entity Extraction Detail

```
Input: "What food does Alice like?"
              │
              ▼
┌─────────────────────────────────────────┐
│           LLM Call                      │
│           ⚠️ 300-1500ms latency         │
│                                         │
│  System: "Extract entities..."          │
│  User: "What food does Alice like?"     │
│  Tools: [EXTRACT_ENTITIES_TOOL]         │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│        LLM Tool Response                │
│                                         │
│  {                                      │
│    "tool_calls": [{                     │
│      "name": "extract_entities",        │
│      "arguments": {                     │
│        "entities": [                    │
│          {"entity": "Alice",            │
│           "entity_type": "person"},     │
│          {"entity": "food",             │
│           "entity_type": "category"}    │
│        ]                                │
│      }                                  │
│    }]                                   │
│  }                                      │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│         Normalized Output               │
│                                         │
│  {"alice": "person", "food": "category"}│
└─────────────────────────────────────────┘
              │
              ▼
    For each entity, embed and search graph
```

---

## Key Files Reference

| File | Purpose | Key Lines |
|------|---------|-----------|
| `mem0/memory/main.py` | Main Memory class with search() | 758-856 (sync), 1807-1913 (async) |
| `mem0/graphs/graph_memory.py` | Neo4j graph implementation | 196-227 (entity extraction), 271-320 (graph search) |
| `mem0/graphs/memgraph_memory.py` | Memgraph implementation | 199-230 (entity extraction) |
| `mem0/graphs/kuzu_memory.py` | Kuzu implementation | 222-253 (entity extraction) |
| `mem0/graphs/tools.py` | LLM tool definitions | 124-150 (EXTRACT_ENTITIES_TOOL), 85-121 (RELATIONS_TOOL) |
| `mem0/graphs/configs.py` | Graph store configurations | Provider enum and configs |
| `mem0/utils/factory.py` | Factory for creating components | GraphStoreFactory |

---

## Summary: LLM Usage and Performance Impact

### Where the LLM is Used

| Stage | LLM Used? | Time Impact | What Happens |
|-------|-----------|-------------|--------------|
| Query received | No | 0ms | `search("What does Alice like?")` |
| **Entity extraction** | **YES** | **300-1500ms** | LLM + tool calling extracts `{"alice": "person"}` |
| Entity embedding | No | 50-200ms | Embedding model converts "alice" to vector |
| Graph query | No | 10-100ms | Cypher query with vector similarity |
| Reranking | No | 1-10ms | BM25 algorithm |
| Return results | No | 0ms | Merge and return |

### Key Takeaways

1. **LLM entity extraction is the primary performance bottleneck** for graph searches
2. The LLM provides semantic understanding of natural language queries
3. Entity extraction enables flexible relationship discovery without keyword matching
4. Consider using faster models (e.g., `gpt-4o-mini`) to reduce latency
5. If graph relationships aren't needed, disable the graph store for faster searches

---

## Appendix: Relationship Extraction (For Adding Data)

When adding new memories, the LLM also extracts relationships using `RELATIONS_TOOL`:

**File**: `mem0/graphs/tools.py` (lines 85-121)

```python
RELATIONS_TOOL = {
    "type": "function",
    "function": {
        "name": "establish_relationships",
        "description": "Establish relationships among entities based on text.",
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "relationship": {"type": "string"},
                            "destination": {"type": "string"},
                        },
                        "required": ["source", "relationship", "destination"],
                    },
                }
            },
            "required": ["entities"],
        },
    },
}
```

**Example**:

Input: `"Alice loves Python programming"`

LLM extracts:
```json
{
    "entities": [
        {
            "source": "Alice",
            "relationship": "LOVES",
            "destination": "Python programming"
        }
    ]
}
```

This creates the graph structure: `(alice)-[:LOVES]->(python_programming)`
