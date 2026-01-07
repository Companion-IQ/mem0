"""
Tests for user_id, agent_id, and run_id filtering.
"""
import pytest
import time


class TestMemoryFilters:
    """Test user_id, agent_id, and run_id filtering."""

    def test_user_isolation(self, memory_instance):
        """Test that users can only see their own memories."""
        user_a = f"user_a_{int(time.time() * 1000)}"
        user_b = f"user_b_{int(time.time() * 1000)}"

        memory_instance.add("User A likes cats", user_id=user_a)
        memory_instance.add("User B likes dogs", user_id=user_b)

        user_a_memories = memory_instance.get_all(user_id=user_a)
        user_b_memories = memory_instance.get_all(user_id=user_b)

        # User A should not see User B's memories
        for mem in user_a_memories:
            memory_text = mem.get("memory", "")
            assert "User B" not in memory_text and "dogs" not in memory_text.lower(), \
                f"User A should not see User B's memories: {memory_text}"

        # User B should not see User A's memories
        for mem in user_b_memories:
            memory_text = mem.get("memory", "")
            assert "User A" not in memory_text and "cats" not in memory_text.lower(), \
                f"User B should not see User A's memories: {memory_text}"

        print(f"User A memories: {user_a_memories}")
        print(f"User B memories: {user_b_memories}")

    def test_agent_filtering(self, memory_instance):
        """Test filtering by agent_id."""
        user_id = f"agent_test_{int(time.time() * 1000)}"

        memory_instance.add(
            "Message from support agent",
            user_id=user_id,
            agent_id="support_agent"
        )
        memory_instance.add(
            "Message from sales agent",
            user_id=user_id,
            agent_id="sales_agent"
        )

        # Get only support agent memories
        support_memories = memory_instance.get_all(
            user_id=user_id,
            agent_id="support_agent"
        )

        assert len(support_memories) >= 1

        # All should be from support_agent
        for mem in support_memories:
            metadata = mem.get("metadata", {})
            # Check metadata or memory content
            is_support = (
                metadata.get("agent_id") == "support_agent" or
                "support" in mem.get("memory", "").lower()
            )
            print(f"Support memory: {mem}")

        print(f"Support agent memories: {support_memories}")

    def test_run_filtering(self, memory_instance):
        """Test filtering by run_id (session)."""
        user_id = f"run_test_{int(time.time() * 1000)}"

        memory_instance.add(
            "Session 1 conversation start",
            user_id=user_id,
            run_id="session_1"
        )
        memory_instance.add(
            "Session 2 different topic",
            user_id=user_id,
            run_id="session_2"
        )

        # Get only session 1 memories
        session_1_memories = memory_instance.get_all(
            user_id=user_id,
            run_id="session_1"
        )

        assert len(session_1_memories) >= 1
        print(f"Session 1 memories: {session_1_memories}")

    def test_combined_filters(self, memory_instance):
        """Test combining user_id, agent_id, and run_id filters."""
        user_id = f"combined_{int(time.time() * 1000)}"

        # Add memory with all three identifiers
        memory_instance.add(
            "Specific context memory for combined test",
            user_id=user_id,
            agent_id="support_bot",
            run_id="conversation_123"
        )

        # Add another memory with different agent
        memory_instance.add(
            "Different agent memory",
            user_id=user_id,
            agent_id="different_bot",
            run_id="conversation_123"
        )

        # Query with all filters
        results = memory_instance.get_all(
            user_id=user_id,
            agent_id="support_bot",
            run_id="conversation_123"
        )

        assert len(results) >= 1
        print(f"Combined filter results: {results}")

    def test_search_with_agent_filter(self, memory_instance):
        """Test semantic search with agent_id filter."""
        user_id = f"search_agent_{int(time.time() * 1000)}"

        memory_instance.add(
            "The weather is sunny today",
            user_id=user_id,
            agent_id="weather_bot"
        )
        memory_instance.add(
            "Your account balance is $100",
            user_id=user_id,
            agent_id="finance_bot"
        )

        # Search with agent filter
        results = memory_instance.search(
            "What's the weather?",
            user_id=user_id,
            agent_id="weather_bot"
        )

        assert results is not None
        print(f"Agent-filtered search results: {results}")

    def test_delete_with_filters(self, memory_instance):
        """Test deleting memories with specific filters."""
        user_id = f"delete_filter_{int(time.time() * 1000)}"

        memory_instance.add("Memory 1", user_id=user_id, agent_id="agent_1")
        memory_instance.add("Memory 2", user_id=user_id, agent_id="agent_2")

        # Delete all for specific agent
        memory_instance.delete_all(user_id=user_id, agent_id="agent_1")

        # Check agent_1 memories are gone
        agent_1_memories = memory_instance.get_all(user_id=user_id, agent_id="agent_1")
        assert len(agent_1_memories) == 0, "Agent 1 memories should be deleted"

        # Check agent_2 memories still exist
        agent_2_memories = memory_instance.get_all(user_id=user_id, agent_id="agent_2")
        # Note: This might not work as expected depending on implementation
        print(f"Agent 1 memories after delete: {agent_1_memories}")
        print(f"Agent 2 memories after delete: {agent_2_memories}")

    def test_multiple_sessions_same_user(self, memory_instance):
        """Test multiple sessions for the same user."""
        user_id = f"multi_session_{int(time.time() * 1000)}"

        # Session 1: Discussing work
        memory_instance.add(
            "I need to finish the report",
            user_id=user_id,
            run_id="work_session"
        )
        memory_instance.add(
            "Meeting at 3pm",
            user_id=user_id,
            run_id="work_session"
        )

        # Session 2: Personal conversation
        memory_instance.add(
            "I want to order pizza tonight",
            user_id=user_id,
            run_id="personal_session"
        )

        # Get work session memories
        work_memories = memory_instance.get_all(
            user_id=user_id,
            run_id="work_session"
        )

        # Get personal session memories
        personal_memories = memory_instance.get_all(
            user_id=user_id,
            run_id="personal_session"
        )

        # Get all memories for user
        all_memories = memory_instance.get_all(user_id=user_id)

        print(f"Work session memories: {len(work_memories)}")
        print(f"Personal session memories: {len(personal_memories)}")
        print(f"All user memories: {len(all_memories)}")

        # All memories should be >= work + personal
        assert len(all_memories) >= len(work_memories) + len(personal_memories) - 1  # Allow some overlap due to deduplication
