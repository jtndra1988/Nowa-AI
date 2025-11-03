from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool, event, Table
from alembic import context

from app.db.models import Base
import os

# Alembic Config object
config = context.config

# Load environment variables
POSTGRES_USER = os.getenv("POSTGRES_USER", "appuser")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "yoursecurepassword")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "db")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "appdb")

# Build connection URL dynamically
db_url = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
config.set_main_option("sqlalchemy.url", db_url)

# Logging
fileConfig(config.config_file_name)

# Target metadata
target_metadata = Base.metadata

# Extend alembic_version.version_num column size
@event.listens_for(Table, "after_parent_attach")
def extend_version_num(table, metadata):
    if table.name == "alembic_version":
        table.c.version_num.type.length = 255

# Offline mode
def run_migrations_offline():
    context.configure(
        url=db_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()

# Online mode
def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()

# Run
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
