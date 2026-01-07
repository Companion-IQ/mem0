"""
Search Performance Demo Script

Demonstrates search speed and result quality with realistic user data.
Tests entity extraction with hobbies, food preferences, and skiing at Lake Louise.

Usage:
    python demo_search_performance.py              # Add memories + search
    python demo_search_performance.py --search-only  # Search existing memories

Output logged to: demo_search_results.log

Prerequisites:
    - Milvus running locally (port 19530)
    - OPENAI_API_KEY in .env
"""
import argparse
import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

# Log file path
LOG_FILE = "demo_search_results.log"
STATE_FILE = "demo_state.txt"  # Stores last collection/user for --search-only


class TeeLogger:
    """Write to both stdout and log file."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def get_config(collection_name: str, milvus_url: str):
    """Memory configuration with entity extraction enabled."""
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
        "enable_entity_extraction": True,
        "enable_entity_search": True,
    }


# User memories to add - tuples of (text, event_date) where event_date is YYYY-MM-DD or None
USER_MEMORIES = [
    # Activities with SPECIFIC DATES - using new event_date parameter
    ("I went skiing at Lake Louise on January 2, 2026", "2026-01-02"),
    ("I had an amazing powder day at Lake Louise on January 3, 2026", "2026-01-03"),
    ("On January 5, 2026 I tried snowboarding for the first time", "2026-01-05"),
    ("I went rock climbing at the indoor gym on December 15, 2025", "2025-12-15"),
    ("I hiked Banff National Park on October 12, 2025", "2025-10-12"),
    ("I attended a guitar concert on November 20, 2025", "2025-11-20"),
    ("I traveled to Japan on March 15, 2025", "2025-03-15"),
    ("I got my new Rossignol skis on December 1, 2025", "2025-12-01"),

    # Hobbies (general) - no specific date
    ("I love skiing at Lake Louise every winter, it's my favorite ski resort", None),
    ("Mountain biking is my summer passion, I ride trails every weekend", None),
    ("I'm really into rock climbing, both indoor and outdoor", None),
    ("Photography is one of my favorite hobbies, I love landscape shots", None),
    ("I read a lot of sci-fi novels, especially Isaac Asimov", None),
    ("I play guitar and have been learning for 5 years", None),
    ("I love cooking Italian dishes at home", None),

    # Food preferences - no specific date
    ("I absolutely love sushi and Japanese cuisine", None),
    ("Italian pasta is my comfort food, especially carbonara", None),
    ("Thai food is amazing, pad thai is my go-to order", None),
    ("My favorite dessert is tiramisu, I could eat it every day", None),
    ("I prefer coffee over tea, especially espresso", None),
    ("I'm allergic to shellfish, so I avoid shrimp and crab", None),

    # Lake Louise skiing details - no specific date
    ("I've been skiing at Lake Louise for 10 years now", None),
    ("I prefer black diamond runs, the steeper the better", None),
    ("Lake Louise in Alberta has the best powder snow I've ever experienced", None),
]

# Queries to test
TEST_QUERIES = [
    # Date-specific queries
    "Where did I go on January 2?",
    "What did I do on January 3, 2026?",
    "What happened in December 2025?",
    "When did I go to Japan?",

    # General queries
    "What are my hobbies?",
    "What food do I like?",
    "Where do I ski?",
    "What do I love?",
    "Tell me about Lake Louise",
    "What am I allergic to?",
]


def save_state(collection_name, user_id):
    """Save state for --search-only mode."""
    with open(STATE_FILE, "w") as f:
        f.write(f"{collection_name}\n{user_id}\n")


def load_state():
    """Load state from previous run."""
    if not os.path.exists(STATE_FILE):
        return None, None
    with open(STATE_FILE, "r") as f:
        lines = f.read().strip().split("\n")
        if len(lines) >= 2:
            return lines[0], lines[1]
    return None, None


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Search Performance Demo")
    parser.add_argument("--search-only", action="store_true",
                        help="Skip adding memories, just run searches")
    args = parser.parse_args()

    # Setup logging to file and stdout
    logger = TeeLogger(LOG_FILE)
    sys.stdout = logger

    print(f"Logging output to: {LOG_FILE}")
    print()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set in environment")
        logger.close()
        return

    milvus_url = os.getenv("MILVUS_URL", "http://localhost:19530")

    # Handle search-only mode
    if args.search_only:
        collection_name, user_id = load_state()
        if not collection_name or not user_id:
            print("ERROR: No previous state found. Run without --search-only first.")
            logger.close()
            return
        print("MODE: Search-only (using existing memories)")
    else:
        collection_name = f"demo_perf_{int(time.time() * 1000)}"
        user_id = f"demo_user_{int(time.time() * 1000)}"
        print("MODE: Full (adding memories + search)")

    print()
    print("=" * 60)
    print("SEARCH PERFORMANCE DEMO")
    print("=" * 60)
    print(f"Milvus URL: {milvus_url}")
    print(f"Collection: {collection_name}")
    print(f"User ID: {user_id}")
    print()

    # Initialize memory
    from mem0 import Memory

    config = get_config(collection_name, milvus_url)
    memory = Memory.from_config(config)

    # Add user memories (skip if search-only)
    if not args.search_only:
        print("=" * 60)
        print("ADDING USER MEMORIES")
        print("=" * 60)

        add_start = time.time()
        for i, (mem_text, event_date) in enumerate(USER_MEMORIES, 1):
            # Use new event_date parameter for explicit date storage
            memory.add(mem_text, user_id=user_id, event_date=event_date)
            date_str = f" [date: {event_date}]" if event_date else ""
            print(f"  [{i:2d}/{len(USER_MEMORIES)}] {mem_text[:45]}...{date_str}")
        add_elapsed = time.time() - add_start

        print()
        print(f"Added {len(USER_MEMORIES)} memories in {add_elapsed:.2f}s")
        print(f"Average: {add_elapsed/len(USER_MEMORIES):.2f}s per memory")

        # Save state for future --search-only runs
        save_state(collection_name, user_id)

        # Wait for indexing
        print()
        print("Waiting for Milvus indexing...")
        time.sleep(2)

    # Run search tests
    print()
    print("=" * 60)
    print("SEARCH PERFORMANCE TESTS")
    print("=" * 60)

    search_times = []

    for query in TEST_QUERIES:
        print()
        print(f'Query: "{query}"')
        print("-" * 40)

        start = time.time()
        results = memory.search(query, user_id=user_id, limit=5)
        elapsed = time.time() - start
        search_times.append(elapsed)

        print(f"Time: {elapsed:.3f}s")
        print("Results:")

        for i, result in enumerate(results.get("results", []), 1):
            mem_text = result.get("memory", "N/A")
            score = result.get("score", 0)
            # Truncate long memories
            if len(mem_text) > 60:
                mem_text = mem_text[:57] + "..."
            print(f"  {i}. [{score:.3f}] {mem_text}")

        # Show relations if available
        if "relations" in results and results["relations"]:
            print("Relations:")
            for rel in results["relations"][:3]:
                src = rel.get("source", "?")
                relationship = rel.get("relationship", "?")
                dest = rel.get("destination", "?")
                print(f"  - {src} --[{relationship}]--> {dest}")

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total queries: {len(TEST_QUERIES)}")
    print(f"Total search time: {sum(search_times):.3f}s")
    print(f"Average search time: {sum(search_times)/len(search_times):.3f}s")
    print(f"Min search time: {min(search_times):.3f}s")
    print(f"Max search time: {max(search_times):.3f}s")

    # Cleanup option
    print()
    print(f"Collection '{collection_name}' left in Milvus for inspection.")
    print("To clean up, use: memory.vector_store.delete_col()")

    # Close logger
    print()
    print(f"Results saved to: {LOG_FILE}")
    logger.close()
    sys.stdout = logger.terminal


if __name__ == "__main__":
    main()
