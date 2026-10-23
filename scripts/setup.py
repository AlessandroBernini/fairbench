#!/usr/bin/env python3
"""Setup FairBench database on Neon.

Usage:
    pip install psycopg2-binary
    python scripts/setup.py
"""

import os
import sys


def setup():
    try:
        import psycopg2
    except ImportError:
        print("❌ Install psycopg2 first: pip install psycopg2-binary")
        print("   Then run: python scripts/setup.py")
        sys.exit(1)

    neon_url = os.environ.get("NEON_DATABASE_URL")
    if not neon_url:
        print("❌ Set NEON_DATABASE_URL environment variable")
        print("   NEON_DATABASE_URL=postgresql://... python scripts/setup.py")
        sys.exit(1)

    print(f"🔌 Connecting to Neon...")
    conn = psycopg2.connect(neon_url)
    conn.autocommit = True
    cur = conn.cursor()

    # Read schema
    schema_path = os.path.join(os.path.dirname(__file__), "..", "schema.sql")
    with open(schema_path) as f:
        schema_sql = f.read()

    print("📋 Applying schema...")
    cur.execute(schema_sql)
    print("✅ Schema applied!")

    # Verify
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
    tables = [row[0] for row in cur.fetchall()]
    print(f"📋 Tables created: {', '.join(tables)}")

    cur.close()
    conn.close()
    print("🎉 Done!")


if __name__ == "__main__":
    setup()
