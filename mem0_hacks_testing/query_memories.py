"""
Interactive memory query tool.

Usage:
    python query_memories.py --collection anton_memories --user-id senior_+16478328908

Commands:
    Type your query and press Enter
    /quit or /exit - Exit the program
    /limit N - Set result limit (default: 5)
    /relations - Toggle showing relations (default: on)
"""
import argparse
import os
import sys
from dotenv import load_dotenv

load_dotenv()


def run_query_repl(collection_name: str, user_id: str):
    """Interactive query REPL for memories."""

    # Validate API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set in environment")
        sys.exit(1)

    milvus_url = os.getenv("MILVUS_URL", "http://localhost:19530")

    print("=" * 60)
    print("MEMORY QUERY TOOL")
    print("=" * 60)
    print(f"Collection: {collection_name}")
    print(f"User ID: {user_id}")
    print(f"Milvus: {milvus_url}")
    print()
    print("Loading memory system...")

    from mem0 import Memory

    memory = Memory.from_config({
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

    print("Ready!")
    print()
    print("Commands:")
    print("  /quit, /exit  - Exit")
    print("  /limit N      - Set result limit")
    print("  /relations    - Toggle relations display")
    print("  /help         - Show commands")
    print()
    print("-" * 60)

    limit = 5
    show_relations = True

    while True:
        try:
            query = input("\nQuery> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue

        # Handle commands
        if query.lower() in ["/quit", "/exit", "/q"]:
            print("Goodbye!")
            break

        if query.lower() == "/help":
            print("Commands:")
            print("  /quit, /exit  - Exit")
            print("  /limit N      - Set result limit (current: {})".format(limit))
            print("  /relations    - Toggle relations display (current: {})".format(show_relations))
            continue

        if query.lower().startswith("/limit"):
            try:
                limit = int(query.split()[1])
                print(f"Result limit set to {limit}")
            except (IndexError, ValueError):
                print("Usage: /limit N (e.g., /limit 10)")
            continue

        if query.lower() == "/relations":
            show_relations = not show_relations
            print(f"Relations display: {'ON' if show_relations else 'OFF'}")
            continue

        # Execute search
        try:
            import time
            start = time.time()
            results = memory.search(query, user_id=user_id, limit=limit)
            elapsed = time.time() - start

            memories = results.get("results", [])
            relations = results.get("relations", [])

            print()
            print(f"Found {len(memories)} results ({elapsed:.3f}s)")
            print("-" * 40)

            for i, mem in enumerate(memories, 1):
                score = mem.get("score", 0)
                text = mem.get("memory", "")

                # Get event dates if available
                metadata = mem.get("metadata") or {}
                event_dates = metadata.get("event_dates", [])
                date_str = f" [{', '.join(event_dates)}]" if event_dates else ""

                print(f"{i}. [{score:.3f}]{date_str} {text}")

            if show_relations and relations:
                print()
                print("Relations:")
                for rel in relations[:5]:
                    src = rel.get("source", "?")
                    relationship = rel.get("relationship", "?")
                    dest = rel.get("destination", "?")
                    print(f"  - {src} --[{relationship}]--> {dest}")
                if len(relations) > 5:
                    print(f"  ... and {len(relations) - 5} more")

        except Exception as e:
            print(f"ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive memory query tool"
    )
    parser.add_argument(
        "--collection",
        required=True,
        help="Milvus collection name"
    )
    parser.add_argument(
        "--user-id",
        required=True,
        help="User ID to query memories for"
    )

    args = parser.parse_args()

    run_query_repl(
        collection_name=args.collection,
        user_id=args.user_id,
    )


if __name__ == "__main__":
    main()
