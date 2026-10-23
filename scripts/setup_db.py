"""Setup script to create FairBench database schema on Neon."""

import asyncio
import sys


async def main():
    try:
        from neon_core.database import NeonDatabase
    except ImportError:
        print("❌ neon-core not installed. Install with: pip install neon-core")
        sys.exit(1)

    import os
    neon_url = os.environ.get("NEON_DATABASE_URL")
    if not neon_url:
        print("❌ NEON_DATABASE_URL not set")
        sys.exit(1)

    db = NeonDatabase(neon_url)
    await db.connect()

    # Read schema
    schema_sql = open("schema.sql").read()

    # Apply schema
    try:
        await db.execute(schema_sql)
        print("✅ Schema applied successfully")
    except Exception as e:
        print(f"⚠️  Schema error (may be normal on first run): {e}")

    # Verify tables
    result = await db.fetch("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
    tables = [row["table_name"] for row in result]
    print(f"📋 Tables: {', '.join(tables)}")

    await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
