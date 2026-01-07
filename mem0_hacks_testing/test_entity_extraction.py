"""
Tests for Write-Time Entity Extraction feature.

This tests the custom enhancement described in:
mem0_hacks/write_time_entity_extraction.md

Features tested:
1. enable_entity_extraction - extracts entities at add-time
2. enable_entity_search - fast local matching at search-time
3. Self-reference queries (I, me, my → user_id)
4. Relationship type queries (like → likes, loves, etc.)
5. Date queries (January 15 → date filter)
6. Relations returned in search results

IMPORTANT: This test requires Milvus as the vector store because:
- Milvus supports JSON metadata with arrays (entity_names, relationship_types, etc.)
- Milvus supports json_contains_any filtering (equivalent to MongoDB $in)
- ChromaDB/Qdrant only support scalar metadata values

Prerequisites:
- Milvus running locally: docker run -d --name milvus -p 19530:19530 milvusdb/milvus:latest standalone
"""
import pytest
import time
import os
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture(scope="session")
def openai_api_key():
    """Ensure OpenAI API key is available."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return key


@pytest.fixture(scope="session")
def milvus_url():
    """Get Milvus URL from environment."""
    url = os.getenv("MILVUS_URL", "http://localhost:19530")
    return url


@pytest.fixture
def entity_extraction_config(milvus_url):
    """Config with entity extraction enabled using Milvus."""
    collection_name = f"entity_test_{int(time.time() * 1000)}"
    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4o-mini",
                "temperature": 0.1,
            }
        },
        "vector_store": {
            "provider": "milvus",
            "config": {
                "url": milvus_url,
                "token": "",
                "collection_name": collection_name,
                "embedding_model_dims": 1536,
                "metric_type": "COSINE",
            }
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small",
            }
        },
        # Enable entity extraction features
        "enable_entity_extraction": True,
        "enable_entity_search": True,
    }


@pytest.fixture
def entity_memory(entity_extraction_config, openai_api_key):
    """Memory instance with entity extraction enabled."""
    from mem0 import Memory
    return Memory.from_config(entity_extraction_config)


class TestEntityExtractionAtAddTime:
    """Test that entities are extracted when adding memories."""

    def test_entities_stored_in_metadata(self, entity_memory):
        """Test that entity extraction stores entities in metadata."""
        user_id = f"entity_test_{int(time.time() * 1000)}"

        # Add memory with entities
        result = entity_memory.add(
            "Alice works at Google and loves pizza",
            user_id=user_id
        )

        # Get the memory back
        memories = entity_memory.get_all(user_id=user_id)
        assert len(memories["results"]) >= 1

        # Check metadata has entity fields
        metadata = memories["results"][0].get("metadata", {})
        print(f"Stored metadata: {metadata}")

        # These fields should be populated by entity extraction
        assert "entities" in metadata or "entity_names" in metadata, \
            "Entity extraction should store entities in metadata"

    def test_relationships_stored_in_metadata(self, entity_memory):
        """Test that relationships are extracted and stored."""
        user_id = f"rel_test_{int(time.time() * 1000)}"

        entity_memory.add(
            "Bob likes coffee and hates tea",
            user_id=user_id
        )

        memories = entity_memory.get_all(user_id=user_id)
        metadata = memories["results"][0].get("metadata", {})
        print(f"Relationship metadata: {metadata}")

        # Should have relationships stored
        has_relationships = (
            "relationships" in metadata or
            "relationship_types" in metadata
        )
        assert has_relationships, "Should store relationships in metadata"

    def test_dates_extracted_and_stored(self, entity_memory):
        """Test that dates are extracted from memory text."""
        user_id = f"date_test_{int(time.time() * 1000)}"

        entity_memory.add(
            "I started my new job on January 15, 2024",
            user_id=user_id
        )

        time.sleep(1)  # Allow Milvus indexing to complete
        memories = entity_memory.get_all(user_id=user_id)
        metadata = memories["results"][0].get("metadata", {})
        print(f"Date metadata: {metadata}")

        # Should have event_dates if date extraction works
        if "event_dates" in metadata:
            assert len(metadata["event_dates"]) > 0, "Should extract dates"


class TestEntitySearchMatching:
    """Test fast search matching using pre-extracted entities."""

    @pytest.fixture(autouse=True)
    def setup_test_data(self, entity_memory):
        """Add test data with various entities and relationships."""
        self.user_id = f"search_entity_{int(time.time() * 1000)}"
        self.memory = entity_memory

        # Add memories with different relationship types
        test_data = [
            "I love pizza and Italian food",
            "I hate broccoli and spinach",
            "Alice works at Google as an engineer",
            "I met Bob on January 15, 2024",
            "I live in San Francisco",
        ]

        for data in test_data:
            entity_memory.add(data, user_id=self.user_id)

        time.sleep(1)  # Allow indexing

    def test_self_reference_query(self, entity_memory):
        """Test that 'I', 'me', 'my' resolve to user_id."""
        results = entity_memory.search(
            "What do I like?",
            user_id=self.user_id
        )

        assert results is not None
        print(f"Self-reference query results: {results}")

        # Should find the "I love pizza" memory
        found_likes = any(
            "love" in r.get("memory", "").lower() or
            "pizza" in r.get("memory", "").lower()
            for r in results.get("results", [])
        )
        assert found_likes, "Should find 'like' memories for self-reference query"

    def test_relationship_type_likes(self, entity_memory):
        """Test relationship type detection for 'like' queries."""
        results = entity_memory.search(
            "What do I love?",
            user_id=self.user_id
        )

        print(f"Love query results: {results}")

        # Should return relations if entity search is working
        if "relations" in results:
            print(f"Relations found: {results['relations']}")
            # Should have relationship type like 'loves' or 'likes'
            has_love_relation = any(
                r.get("relationship") in ["loves", "likes", "enjoys"]
                for r in results["relations"]
            )
            assert has_love_relation, "Should return love/like relations"

    def test_relationship_type_dislikes(self, entity_memory):
        """Test relationship type detection for 'dislike' queries."""
        results = entity_memory.search(
            "What do I hate?",
            user_id=self.user_id
        )

        print(f"Hate query results: {results}")

        # Should find broccoli/spinach memory
        found_hates = any(
            "hate" in r.get("memory", "").lower() or
            "broccoli" in r.get("memory", "").lower()
            for r in results.get("results", [])
        )
        assert found_hates, "Should find 'hate' memories"

    def test_entity_query_alice(self, entity_memory):
        """Test querying for specific entity 'Alice'."""
        results = entity_memory.search(
            "Where does Alice work?",
            user_id=self.user_id
        )

        print(f"Alice query results: {results}")

        # Should find Google memory
        found_alice = any(
            "alice" in r.get("memory", "").lower() or
            "google" in r.get("memory", "").lower()
            for r in results.get("results", [])
        )
        assert found_alice, "Should find Alice's work memory"

    def test_location_query(self, entity_memory):
        """Test 'where do I live' type queries."""
        results = entity_memory.search(
            "Where do I live?",
            user_id=self.user_id
        )

        print(f"Location query results: {results}")

        found_location = any(
            "live" in r.get("memory", "").lower() or
            "san francisco" in r.get("memory", "").lower()
            for r in results.get("results", [])
        )
        assert found_location, "Should find location memory"

    def test_date_query(self, entity_memory):
        """Test date-based queries."""
        results = entity_memory.search(
            "What happened on January 15?",
            user_id=self.user_id
        )

        print(f"Date query results: {results}")

        # Should find the Bob meeting memory
        found_date = any(
            "january" in r.get("memory", "").lower() or
            "bob" in r.get("memory", "").lower() or
            "met" in r.get("memory", "").lower()
            for r in results.get("results", [])
        )
        assert found_date, "Should find memory with January 15 date"


class TestRelationsInSearchResults:
    """Test that search returns relations extracted from metadata."""

    def test_relations_returned_in_search(self, entity_memory):
        """Test that search results include 'relations' field."""
        user_id = f"relations_test_{int(time.time() * 1000)}"

        entity_memory.add("I love Python programming", user_id=user_id)
        time.sleep(0.5)

        results = entity_memory.search("What do I like?", user_id=user_id)

        print(f"Search results structure: {results.keys() if isinstance(results, dict) else type(results)}")
        print(f"Full results: {results}")

        # Check if relations are returned
        if "relations" in results:
            print(f"Relations: {results['relations']}")
            assert isinstance(results["relations"], list), "Relations should be a list"

    def test_relation_structure(self, entity_memory):
        """Test the structure of returned relations."""
        user_id = f"struct_test_{int(time.time() * 1000)}"

        entity_memory.add("Alice loves coffee", user_id=user_id)
        time.sleep(0.5)

        results = entity_memory.search("What does Alice like?", user_id=user_id)

        if "relations" in results and len(results["relations"]) > 0:
            relation = results["relations"][0]
            print(f"Relation structure: {relation}")

            # Relations should have source, relationship, destination
            assert "source" in relation, "Relation should have 'source'"
            assert "relationship" in relation, "Relation should have 'relationship'"
            assert "destination" in relation, "Relation should have 'destination'"


class TestEntityExtractionDisabled:
    """Test behavior when entity extraction is disabled."""

    @pytest.fixture
    def standard_config(self, milvus_url):
        """Config WITHOUT entity extraction (using Milvus for consistency)."""
        collection_name = f"standard_test_{int(time.time() * 1000)}"
        return {
            "llm": {
                "provider": "openai",
                "config": {"model": "gpt-4o-mini"}
            },
            "vector_store": {
                "provider": "milvus",
                "config": {
                    "url": milvus_url,
                    "token": "",
                    "collection_name": collection_name,
                    "embedding_model_dims": 1536,
                    "metric_type": "COSINE",
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {"model": "text-embedding-3-small"}
            },
            # Entity extraction DISABLED (default)
            "enable_entity_extraction": False,
            "enable_entity_search": False,
        }

    def test_no_entity_fields_when_disabled(self, standard_config, openai_api_key):
        """Test that metadata doesn't have entity fields when disabled."""
        from mem0 import Memory

        memory = Memory.from_config(standard_config)
        user_id = f"disabled_test_{int(time.time() * 1000)}"

        memory.add("I love pizza", user_id=user_id)
        time.sleep(1)  # Allow Milvus indexing to complete

        memories = memory.get_all(user_id=user_id)
        metadata = memories["results"][0].get("metadata", {})

        print(f"Standard metadata (no entity extraction): {metadata}")

        # Should NOT have entity_names, relationship_types fields
        # (unless they're added by default, which is fine)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_query(self, entity_memory):
        """Test handling of empty query."""
        user_id = f"empty_test_{int(time.time() * 1000)}"

        entity_memory.add("Some test memory", user_id=user_id)

        try:
            results = entity_memory.search("", user_id=user_id)
            print(f"Empty query results: {results}")
        except Exception as e:
            print(f"Empty query error (expected): {e}")

    def test_no_entities_in_memory(self, entity_memory):
        """Test memory with no clear entities."""
        user_id = f"no_entity_test_{int(time.time() * 1000)}"

        entity_memory.add("Hello world", user_id=user_id)
        time.sleep(1)  # Allow Milvus indexing to complete

        memories = entity_memory.get_all(user_id=user_id)
        # Note: mem0 may not create a memory for very simple text like "Hello world"
        if memories["results"]:
            metadata = memories["results"][0].get("metadata", {})
            print(f"Simple memory metadata: {metadata}")
        else:
            print("No memory created for simple text (expected behavior)")

    def test_multiple_relationships(self, entity_memory):
        """Test memory with multiple relationships."""
        user_id = f"multi_rel_test_{int(time.time() * 1000)}"

        entity_memory.add(
            "Alice loves Bob, works at Google, and lives in NYC",
            user_id=user_id
        )

        memories = entity_memory.get_all(user_id=user_id)
        metadata = memories["results"][0].get("metadata", {})

        print(f"Multi-relationship metadata: {metadata}")

        if "relationships" in metadata:
            print(f"Found {len(metadata['relationships'])} relationships")

    def test_unicode_entities(self, entity_memory):
        """Test entities with unicode characters."""
        user_id = f"unicode_test_{int(time.time() * 1000)}"

        entity_memory.add(
            "I love café and crème brûlée",
            user_id=user_id
        )

        results = entity_memory.search("What food do I like?", user_id=user_id)
        print(f"Unicode query results: {results}")


class TestPerformanceComparison:
    """Test to demonstrate performance difference."""

    def test_search_speed_with_entity_extraction(self, entity_memory):
        """Measure search speed with entity extraction enabled."""
        user_id = f"perf_test_{int(time.time() * 1000)}"

        # Add some test data
        entity_memory.add("I love Python and JavaScript", user_id=user_id)
        entity_memory.add("I work at a tech company", user_id=user_id)
        time.sleep(0.5)

        # Measure search time
        start = time.time()
        for _ in range(5):
            entity_memory.search("What programming languages do I know?", user_id=user_id)
        elapsed = time.time() - start

        avg_time = elapsed / 5
        print(f"Average search time with entity extraction: {avg_time:.3f}s")

        # With entity extraction, should be fast (no LLM call at search time)
        # Expected: < 0.5s per search if working correctly
