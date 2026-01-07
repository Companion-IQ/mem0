"""
Migrate memories from mem0 cloud to local Milvus.

Usage:
    python migrate_cloud_to_local.py --user-id <user_id>

Prerequisites:
    - MEM0_API_KEY in .env (for cloud access)
    - OPENAI_API_KEY in .env (for local embeddings)
    - Milvus running locally (port 19530)
"""
import argparse
import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()


def migrate_memories(user_id: str, collection_name: str = None, dry_run: bool = False):
    """Migrate all memories for a user from mem0 cloud to local Milvus."""

    # Validate API keys
    mem0_api_key = os.getenv("MEM0_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not mem0_api_key:
        print("ERROR: MEM0_API_KEY not set in environment")
        print("Get your API key from: https://app.mem0.ai/dashboard/api-keys")
        sys.exit(1)

    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY not set in environment")
        sys.exit(1)

    # 1. Connect to mem0 cloud
    print("=" * 60)
    print("MEMORY MIGRATION: mem0 Cloud -> Local Milvus")
    print("=" * 60)
    print(f"User ID: {user_id}")
    print()

    from mem0 import MemoryClient

    print("Connecting to mem0 cloud...")
    cloud_client = MemoryClient(api_key=mem0_api_key)

    # 2. Fetch all memories for the user
    print(f"Fetching memories for user: {user_id}")
    try:
        response = cloud_client.get_all(
            filters={"AND": [{"user_id": user_id}]}
        )
        # Handle both list and dict response formats
        if isinstance(response, dict):
            cloud_memories = response.get("results", response.get("memories", []))
        elif isinstance(response, list):
            cloud_memories = response
        else:
            cloud_memories = []
    except Exception as e:
        print(f"ERROR fetching memories: {e}")
        sys.exit(1)

    if not cloud_memories:
        print("No memories found for this user.")
        sys.exit(0)

    print(f"Found {len(cloud_memories)} memories to migrate")
    print()

    # Show preview
    print("Preview (first 5 memories):")
    print("-" * 40)
    for mem in cloud_memories[:5]:
        memory_text = mem.get("memory", "")[:60] if mem.get("memory") else ""
        created = mem.get("created_at", "")[:10] if mem.get("created_at") else "no-date"
        print(f"  [{created}] {memory_text}...")
    if len(cloud_memories) > 5:
        print(f"  ... and {len(cloud_memories) - 5} more")
    print()

    if dry_run:
        print("DRY RUN: No changes will be made.")
        print("Remove --dry-run flag to perform actual migration.")
        return

    # 3. Setup local Milvus memory
    milvus_url = os.getenv("MILVUS_URL", "http://localhost:19530")
    if not collection_name:
        collection_name = f"migrated_{user_id}_{int(time.time())}"

    print(f"Setting up local Milvus...")
    print(f"  URL: {milvus_url}")
    print(f"  Collection: {collection_name}")
    print()

    from mem0 import Memory

    local_memory = Memory.from_config({
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
        "enable_entity_extraction": True,
        "enable_entity_search": True,
    })

    # 4. Migrate each memory
    print("=" * 60)
    print("MIGRATING MEMORIES")
    print("=" * 60)

    success_count = 0
    error_count = 0
    start_time = time.time()

    for i, mem in enumerate(cloud_memories, 1):
        memory_text = mem.get("memory", "")
        metadata = mem.get("metadata") or {}

        # Extract date from created_at for event_date
        created_at = mem.get("created_at", "")
        event_date = created_at[:10] if created_at and len(created_at) >= 10 else None

        # Display progress
        display_text = memory_text[:45] + "..." if len(memory_text) > 45 else memory_text
        print(f"[{i:3d}/{len(cloud_memories)}] {display_text}")

        try:
            local_memory.add(
                memory_text,
                user_id=user_id,
                metadata=metadata,
                event_date=event_date,
            )
            success_count += 1
        except Exception as e:
            print(f"         ERROR: {e}")
            error_count += 1

        # Small delay to avoid rate limiting
        time.sleep(0.3)

    elapsed = time.time() - start_time

    # 5. Summary
    print()
    print("=" * 60)
    print("MIGRATION COMPLETE")
    print("=" * 60)
    print(f"Total memories: {len(cloud_memories)}")
    print(f"Successfully migrated: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Time elapsed: {elapsed:.1f}s")
    print(f"Average: {elapsed/len(cloud_memories):.2f}s per memory")
    print()
    print(f"Collection '{collection_name}' created in Milvus.")
    print()
    print("To use the migrated memories:")
    print(f'  memory = Memory.from_config({{"vector_store": {{"provider": "milvus", "config": {{"collection_name": "{collection_name}"}}}}}})')
    print(f'  results = memory.search("query", user_id="{user_id}")')


def main():
    parser = argparse.ArgumentParser(
        description="Migrate memories from mem0 cloud to local Milvus"
    )
    parser.add_argument(
        "--user-id",
        required=True,
        help="User ID to migrate memories for"
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Milvus collection name (default: migrated_<user_id>_<timestamp>)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview migration without making changes"
    )

    args = parser.parse_args()

    migrate_memories(
        user_id=args.user_id,
        collection_name=args.collection,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
