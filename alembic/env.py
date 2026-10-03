"""Alembic runtime with externally supplied database credentials."""

from typing import cast

from sqlalchemy import create_engine, pool
from sqlalchemy.engine import URL

from alembic import context

config = context.config

target_metadata = None


def _connection_settings() -> dict[str, object]:
    """Return migration settings supplied programmatically."""
    value = config.attributes.get("connection_settings")

    if not isinstance(value, dict):
        raise RuntimeError("Alembic connection settings were not supplied")

    return cast(
        dict[str, object],
        value,
    )


def run_migrations_online() -> None:
    """Run migrations through a short-lived connection."""
    settings = _connection_settings()

    url = URL.create(
        drivername="postgresql+psycopg",
        username=str(settings["user"]),
        password=str(settings["password"]),
        host=str(settings["host"]),
        port=int(str(settings["port"])),
        database=str(settings["database"]),
        query={
            "sslmode": str(settings["sslmode"]),
        },
    )

    engine = create_engine(
        url,
        poolclass=pool.NullPool,
        hide_parameters=True,
        connect_args={
            "connect_timeout": int(str(settings["connect_timeout"])),
            "application_name": ("ai-platform-alembic"),
        },
    )

    try:
        with engine.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                compare_type=True,
                transactional_ddl=True,
            )

            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


if context.is_offline_mode():
    raise RuntimeError("Offline migrations are intentionally disabled")

run_migrations_online()
