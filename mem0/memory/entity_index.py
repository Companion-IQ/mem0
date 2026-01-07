"""
Entity index management for fast entity lookup.

This module handles building and caching an index of pre-extracted entities
from memory metadata, enabling fast entity matching during search without LLM calls.
"""

import logging
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)


class EntityIndexManager:
    """
    Manages cached entity indices per user for fast lookup during search.

    The index is built from entity metadata stored in memories and cached
    in-memory for fast repeated lookups. Cache is invalidated when new
    memories are added.
    """

    def __init__(self, vector_store):
        """
        Initialize the EntityIndexManager.

        Args:
            vector_store: The vector store instance for querying memories
        """
        self.vector_store = vector_store
        self._cache: Dict[str, Set[str]] = {}

    def get_known_entities(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Set[str]:
        """
        Get all known entity names for a user.

        Args:
            user_id: The user ID to get entities for
            agent_id: Optional agent ID filter
            run_id: Optional run ID filter

        Returns:
            Set of entity names known for this user
        """
        cache_key = self._build_cache_key(user_id, agent_id, run_id)

        if cache_key in self._cache:
            return self._cache[cache_key]

        # Build index from vector store
        entities = self._build_index(user_id, agent_id, run_id)
        self._cache[cache_key] = entities

        logger.debug(f"Built entity index for {cache_key}: {len(entities)} entities")
        return entities

    def invalidate(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        """
        Invalidate the cached index for a user.

        Should be called when new memories are added.

        Args:
            user_id: The user ID to invalidate
            agent_id: Optional agent ID filter
            run_id: Optional run ID filter
        """
        cache_key = self._build_cache_key(user_id, agent_id, run_id)
        if cache_key in self._cache:
            del self._cache[cache_key]
            logger.debug(f"Invalidated entity index cache for {cache_key}")

        # Also invalidate the base user key if we have a more specific key
        base_key = self._build_cache_key(user_id, None, None)
        if base_key != cache_key and base_key in self._cache:
            del self._cache[base_key]

    def _build_cache_key(
        self,
        user_id: str,
        agent_id: Optional[str],
        run_id: Optional[str],
    ) -> str:
        """Build a cache key from user/agent/run IDs."""
        return f"{user_id}:{agent_id or ''}:{run_id or ''}"

    def _build_index(
        self,
        user_id: str,
        agent_id: Optional[str],
        run_id: Optional[str],
    ) -> Set[str]:
        """
        Build entity index by querying vector store for memories with entity metadata.

        Args:
            user_id: The user ID to build index for
            agent_id: Optional agent ID filter
            run_id: Optional run ID filter

        Returns:
            Set of entity names
        """
        entities: Set[str] = set()

        # Build filters
        filters: Dict[str, Any] = {"user_id": user_id}
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        try:
            # Query memories - use list method if available, otherwise use get_all
            memories = self._get_memories_with_entities(filters)

            for memory in memories:
                # Extract entity names from memory payload
                payload = getattr(memory, "payload", None) or {}

                # Check for entity_names field (flat list for filtering)
                entity_names = payload.get("entity_names", [])
                if entity_names and isinstance(entity_names, list):
                    entities.update(entity_names)

                # Also check entities field (structured list of dicts)
                structured_entities = payload.get("entities", [])
                if structured_entities and isinstance(structured_entities, list):
                    for entity in structured_entities:
                        if isinstance(entity, dict) and "name" in entity:
                            entities.add(entity["name"])
                        elif isinstance(entity, str):
                            entities.add(entity)

        except Exception as e:
            logger.warning(f"Error building entity index: {e}")

        return entities

    def _get_memories_with_entities(self, filters: Dict[str, Any]):
        """
        Query vector store for memories, handling different vector store implementations.
        """
        # Try different methods that might be available
        if hasattr(self.vector_store, "list"):
            # Use list method with high limit
            return self.vector_store.list(filters=filters, limit=10000)

        if hasattr(self.vector_store, "get_all"):
            # Some stores have get_all
            return self.vector_store.get_all(filters=filters)

        # Fallback: try to search with empty query
        # This may not return all results but is better than nothing
        if hasattr(self.vector_store, "search"):
            try:
                # Some vector stores support listing via search with special parameters
                results = self.vector_store.search(
                    query="",
                    vectors=[0.0] * 1536,  # Dummy vector - will be ignored with empty query
                    limit=10000,
                    filters=filters,
                )
                return results
            except Exception:
                pass

        logger.warning("Vector store does not support listing memories for entity index")
        return []

    def clear_cache(self) -> None:
        """Clear all cached indices."""
        self._cache.clear()
        logger.debug("Cleared all entity index caches")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        total_entities = sum(len(entities) for entities in self._cache.values())
        return {
            "cached_users": len(self._cache),
            "total_entities": total_entities,
        }
