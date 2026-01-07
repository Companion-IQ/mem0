"""
Semantic search functionality tests.
"""
import pytest
import time


class TestMemorySearch:
    """Test semantic search functionality."""

    @pytest.fixture(autouse=True)
    def setup_test_data(self, memory_instance):
        """Add test data before each test."""
        self.user_id = f"search_user_{int(time.time() * 1000)}"
        self.memory = memory_instance

        # Add diverse memories
        memories = [
            "I love Italian food, especially pasta and pizza",
            "My favorite programming language is Python",
            "I work at a tech startup in San Francisco",
            "I have a golden retriever named Max",
            "I enjoy hiking on weekends in the mountains",
        ]

        for mem in memories:
            memory_instance.add(mem, user_id=self.user_id)

        # Small delay to ensure indexing
        time.sleep(0.5)

    def test_semantic_search_food(self, memory_instance):
        """Test searching for food-related memories."""
        results = memory_instance.search(
            "What food do I like?",
            user_id=self.user_id
        )

        assert results is not None
        assert "results" in results

        # Should find Italian food memory
        found_food = any(
            "Italian" in r.get("memory", "") or
            "pasta" in r.get("memory", "") or
            "pizza" in r.get("memory", "")
            for r in results["results"]
        )

        print(f"Food search results: {results}")
        assert found_food, "Should find food-related memory"

    def test_semantic_search_work(self, memory_instance):
        """Test searching for work-related memories."""
        results = memory_instance.search(
            "Where do I work?",
            user_id=self.user_id
        )

        assert results is not None

        found_work = any(
            "startup" in r.get("memory", "").lower() or
            "san francisco" in r.get("memory", "").lower()
            for r in results.get("results", [])
        )

        print(f"Work search results: {results}")
        assert found_work, "Should find work-related memory"

    def test_semantic_search_pet(self, memory_instance):
        """Test searching for pet-related memories."""
        results = memory_instance.search(
            "Do I have any pets?",
            user_id=self.user_id
        )

        assert results is not None

        found_pet = any(
            "retriever" in r.get("memory", "").lower() or
            "max" in r.get("memory", "").lower() or
            "dog" in r.get("memory", "").lower()
            for r in results.get("results", [])
        )

        print(f"Pet search results: {results}")
        assert found_pet, "Should find pet-related memory"

    def test_semantic_search_hobby(self, memory_instance):
        """Test searching for hobby-related memories."""
        results = memory_instance.search(
            "What do I like to do for fun?",
            user_id=self.user_id
        )

        assert results is not None

        found_hobby = any(
            "hiking" in r.get("memory", "").lower() or
            "mountains" in r.get("memory", "").lower()
            for r in results.get("results", [])
        )

        print(f"Hobby search results: {results}")
        assert found_hobby, "Should find hobby-related memory"

    def test_search_with_limit(self, memory_instance):
        """Test search with result limit."""
        results = memory_instance.search(
            "Tell me about myself",
            user_id=self.user_id,
            limit=2
        )

        assert results is not None
        assert len(results.get("results", [])) <= 2
        print(f"Limited search results (limit=2): {len(results.get('results', []))} results")

    def test_search_no_results(self, memory_instance):
        """Test search that shouldn't match anything strongly."""
        results = memory_instance.search(
            "quantum physics experiments with black holes",
            user_id=self.user_id
        )

        # Should return results (possibly low relevance) or empty
        assert results is not None
        print(f"Unrelated search results: {results}")

    def test_search_cross_user_isolation(self, memory_instance):
        """Test that search doesn't return other users' memories."""
        other_user = f"other_user_{int(time.time() * 1000)}"

        # Add memory for another user
        memory_instance.add("I hate Italian food", user_id=other_user)

        # Search for current user
        results = memory_instance.search(
            "What food do I like?",
            user_id=self.user_id
        )

        # Should not find the other user's memory
        for r in results.get("results", []):
            assert "hate" not in r.get("memory", "").lower(), \
                "Should not find other user's memories"

    def test_search_with_threshold(self, memory_instance):
        """Test search with similarity threshold."""
        try:
            results = memory_instance.search(
                "What is my name?",
                user_id=self.user_id,
                threshold=0.5  # Higher threshold = more relevant results only
            )
            print(f"Threshold search results: {results}")
        except TypeError:
            # threshold might not be supported in all versions
            pytest.skip("Threshold parameter not supported")

    def test_search_programming_context(self, memory_instance):
        """Test searching with programming context."""
        results = memory_instance.search(
            "What programming languages do I know?",
            user_id=self.user_id
        )

        assert results is not None

        found_programming = any(
            "python" in r.get("memory", "").lower()
            for r in results.get("results", [])
        )

        print(f"Programming search results: {results}")
        assert found_programming, "Should find programming-related memory"
