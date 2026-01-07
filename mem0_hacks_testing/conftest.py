"""
Pytest fixtures for mem0 integration tests.
"""
import os
import pytest
import shutil
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@pytest.fixture(scope="session")
def openai_api_key():
    """Ensure OpenAI API key is available."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set in environment")
    return key


@pytest.fixture
def temp_db_path():
    """Create a temporary directory for ChromaDB."""
    path = tempfile.mkdtemp(prefix="mem0_test_")
    yield path
    # Cleanup after test
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def memory_config(temp_db_path):
    """Base memory configuration with ChromaDB."""
    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4o-mini",
                "temperature": 0.1,
            }
        },
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": "test_memories",
                "path": temp_db_path,
            }
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small",
            }
        },
    }


@pytest.fixture
def memory_instance(memory_config, openai_api_key):
    """Create a Memory instance with ChromaDB."""
    from mem0 import Memory
    return Memory.from_config(memory_config)


@pytest.fixture
def persistent_db_path():
    """Create a persistent database path for manual testing."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "db")
    os.makedirs(db_path, exist_ok=True)
    return db_path


@pytest.fixture
def persistent_memory_config(persistent_db_path):
    """Memory configuration with persistent ChromaDB storage."""
    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4o-mini",
                "temperature": 0.1,
            }
        },
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": "persistent_memories",
                "path": persistent_db_path,
            }
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small",
            }
        },
    }
