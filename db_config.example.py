"""Database connection config — sanitised template.

Copy this file to db_config.py and fill in your real Postgres
credentials. db_config.py is gitignored so credentials won't be
committed.

The scripts in this repo expect a Postgres database containing a
public.speeches table (the Hansard corpus) and the speech_ai_scores
table created by the migrations in migrations/. See README.md for the
expected schema.
"""

import os

DB_CONFIG = {
    "dbname": os.environ.get("GR_DB_NAME", "your_database_name"),
    "user": os.environ.get("GR_DB_USER", "your_username"),
    "password": os.environ.get("GR_DB_PASSWORD", ""),
    "host": os.environ.get("GR_DB_HOST", "localhost"),
    "port": int(os.environ.get("GR_DB_PORT", 5432)),
}
