# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mem0 ("mem-zero") is an intelligent memory layer for AI assistants and agents. It provides persistent, personalized memory capabilities with multi-level memory (User, Session, Agent state), cross-platform SDKs (Python, TypeScript/JavaScript), and a hosted platform option.

## Common Commands

### Python Development
```bash
make install              # Create hatch environment
make install_all          # Install all optional dependencies
make test                 # Run tests (default Python)
make test-py-3.10         # Test on specific Python version (3.9, 3.10, 3.11, 3.12)
make lint                 # Lint with ruff
make lint-fix             # Lint and auto-fix issues
make format               # Format with ruff
make sort                 # Sort imports with isort
make build                # Build distribution
make docs                 # Start Mintlify dev server
```

### TypeScript SDK (mem0-ts/)
```bash
npm run build             # Build with tsup (CJS + ESM)
npm run dev               # Watch mode with nodemon
npm run test              # Run Jest tests
npm run format            # Format with Prettier
```

### Vercel AI SDK (vercel-ai-sdk/)
```bash
npm run build             # Build with tsup
npm run test              # Jest tests
npm run type-check        # TypeScript type checking
```

## Architecture

### Plugin Architecture with Factory Pattern

All major components are pluggable via factory classes in `mem0/utils/factory.py`:
- **LlmFactory**: OpenAI, Anthropic, Groq, Ollama, Azure, Bedrock, Gemini, +8 more
- **EmbedderFactory**: OpenAI, Gemini, HuggingFace, FastEmbed, Together, Ollama, +6 more
- **VectorStoreFactory**: Qdrant (default), Chroma, Weaviate, Pinecone, FAISS, MongoDB, PostgreSQL, +18 more
- **GraphStoreFactory**: Memgraph, Neptune, Neo4j, Kuzu
- **RerankerFactory**: LLM-based, HuggingFace, Cohere, Sentence Transformer

Components are instantiated via configuration:
```python
config = MemoryConfig(
    llm={"provider": "openai", "config": {...}},
    embedder={"provider": "openai", "config": {...}},
    vector_store={"provider": "qdrant", "config": {...}},
)
memory = Memory(config=config)
```

### Dual API Models

- **Self-hosted**: `Memory` and `AsyncMemory` classes in `mem0/memory/main.py`
- **Cloud-hosted**: `MemoryClient` and `AsyncMemoryClient` in `mem0/client/main.py`

### Memory Processing Pipeline

1. Input parsing (messages/text)
2. Embedding generation via embedder
3. Vector storage in configured vector store
4. Semantic search with optional reranking
5. LLM-based memory update (compress/synthesize)
6. Metadata extraction and SQLite persistence

### Session Scoping

All memory operations support scoping via:
- `user_id` (primary identifier)
- `agent_id` (optional agent context)
- `run_id` (optional session/run context)

## Key Files

- `mem0/memory/main.py` - Core Memory and AsyncMemory classes
- `mem0/utils/factory.py` - Component factories for pluggable architecture
- `mem0/configs/base.py` - Pydantic configuration models
- `mem0/configs/prompts.py` - System prompts for different memory types
- `mem0/client/main.py` - API client for hosted platform
- `mem0/exceptions.py` - Custom exception hierarchy

## Code Organization

```
mem0/                  # Main Python package
├── memory/            # Core memory implementation
├── client/            # API client for hosted platform
├── configs/           # Pydantic config models
├── llms/              # LLM provider integrations
├── embeddings/        # Embedding model providers
├── vector_stores/     # Vector database integrations (25+)
├── graphs/            # Graph database integration
├── reranker/          # Reranking algorithms
└── utils/             # Factories and utilities

mem0-ts/               # TypeScript SDK (v2.2.0)
vercel-ai-sdk/         # Vercel AI integration
server/                # FastAPI server with Docker support
tests/                 # Python tests (pytest)
```

## Development Standards

- **Python**: 3.9-3.12 supported, tested in CI matrix
- **Linting**: Ruff for Python, ESLint for TypeScript
- **Formatting**: Ruff format (Python), Prettier (TypeScript)
- **Pre-commit**: ruff check --fix, isort --profile black
- **Type Validation**: Pydantic for all configs and data models

## When Adding Features

1. Use factory pattern for any pluggable component
2. Implement both sync and async versions
3. Add Pydantic config models in `configs/`
4. Include tests for Python 3.9-3.12
5. Update `configs/prompts.py` if modifying memory behavior
