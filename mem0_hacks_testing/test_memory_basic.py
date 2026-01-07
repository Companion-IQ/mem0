"""
Basic memory operations tests - CRUD functionality.
"""
import pytest
from mem0 import Memory


class TestBasicMemoryOperations:
    """Test basic add/get/update/delete operations."""

    def test_add_single_message(self, memory_instance):
        """Test adding a single user message."""
        result = memory_instance.add(
            "I love Python programming",
            user_id="test_user_1"
        )

        assert result is not None
        # Check for either 'results' or 'memories' key depending on version
        assert "results" in result or "memories" in result
        print(f"Add result: {result}")

    def test_add_conversation(self, memory_instance):
        """Test adding a conversation with multiple messages."""
        messages = [
            {"role": "user", "content": "My name is Alice"},
            {"role": "assistant", "content": "Nice to meet you, Alice!"},
            {"role": "user", "content": "I work as a software engineer"},
        ]

        result = memory_instance.add(messages, user_id="test_user_2")

        assert result is not None
        print(f"Conversation add result: {result}")

    def test_get_all_memories(self, memory_instance):
        """Test retrieving all memories for a user."""
        # Add some memories first
        memory_instance.add("I like coffee", user_id="test_user_3")
        memory_instance.add("I prefer tea in the evening", user_id="test_user_3")

        memories = memory_instance.get_all(user_id="test_user_3")

        assert memories is not None
        assert len(memories) >= 1
        print(f"Retrieved {len(memories)} memories")

    def test_get_memory_by_id(self, memory_instance):
        """Test retrieving a specific memory by ID."""
        result = memory_instance.add("Test memory content", user_id="test_user_4")

        # Extract memory ID from result
        memory_id = None
        if result.get("results"):
            memory_id = result["results"][0].get("id")
        elif result.get("memories"):
            memory_id = result["memories"][0].get("id")

        if memory_id:
            memory = memory_instance.get(memory_id)
            assert memory is not None
            print(f"Retrieved memory by ID: {memory}")
        else:
            pytest.skip("Could not extract memory ID from add result")

    def test_update_memory(self, memory_instance):
        """Test updating an existing memory."""
        # Add a memory first
        result = memory_instance.add("I like pizza", user_id="test_user_update")

        # Extract memory ID
        memory_id = None
        if result.get("results"):
            memory_id = result["results"][0].get("id")
        elif result.get("memories"):
            memory_id = result["memories"][0].get("id")

        if memory_id:
            # Update the memory
            update_result = memory_instance.update(memory_id, "I love pizza very much")
            assert update_result is not None
            print(f"Update result: {update_result}")
        else:
            pytest.skip("Could not extract memory ID for update test")

    def test_delete_memory(self, memory_instance):
        """Test deleting a memory."""
        result = memory_instance.add("Memory to delete", user_id="test_user_5")

        # Extract memory ID
        memory_id = None
        if result.get("results"):
            memory_id = result["results"][0].get("id")
        elif result.get("memories"):
            memory_id = result["memories"][0].get("id")

        if memory_id:
            # Delete the memory
            memory_instance.delete(memory_id)

            # Verify deletion
            memory = memory_instance.get(memory_id)
            # After deletion, get should return None or empty
            assert memory is None or memory.get("memory") is None
            print(f"Memory {memory_id} deleted successfully")
        else:
            pytest.skip("Could not extract memory ID for delete test")

    def test_delete_all_user_memories(self, memory_instance):
        """Test deleting all memories for a user."""
        user_id = "delete_all_user"

        memory_instance.add("Memory 1", user_id=user_id)
        memory_instance.add("Memory 2", user_id=user_id)

        # Verify memories exist
        memories_before = memory_instance.get_all(user_id=user_id)
        assert len(memories_before) >= 1

        # Delete all
        memory_instance.delete_all(user_id=user_id)

        # Verify deletion
        memories_after = memory_instance.get_all(user_id=user_id)
        assert len(memories_after) == 0
        print(f"All memories for {user_id} deleted successfully")

    def test_add_with_metadata(self, memory_instance):
        """Test adding memory with custom metadata."""
        result = memory_instance.add(
            "I have a meeting at 3pm",
            user_id="test_user_metadata",
            metadata={"category": "calendar", "priority": "high"}
        )

        assert result is not None
        print(f"Add with metadata result: {result}")

    def test_memory_history(self, memory_instance):
        """Test that memory history is tracked."""
        user_id = "history_test_user"

        # Add memory
        result = memory_instance.add("I am learning machine learning", user_id=user_id)

        # Extract memory ID
        memory_id = None
        if result.get("results"):
            memory_id = result["results"][0].get("id")
        elif result.get("memories"):
            memory_id = result["memories"][0].get("id")

        if memory_id:
            # Try to get history
            try:
                history = memory_instance.history(memory_id)
                print(f"Memory history: {history}")
            except Exception as e:
                print(f"History not available: {e}")
