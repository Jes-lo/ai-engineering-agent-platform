"""Provision the PostgreSQL login used by the application runtime."""

import argparse
import re
from pathlib import Path

import psycopg
from psycopg import sql

EXPECTED_BOOTSTRAP_KEYS = {
    "POSTGRES_USER",
    "POSTGRES_DB",
    "POSTGRES_PASSWORD",
}

EXPECTED_RUNTIME_KEYS = {
    "AI_PLATFORM_POSTGRES_HOST",
    "AI_PLATFORM_POSTGRES_PORT",
    "AI_PLATFORM_POSTGRES_DATABASE",
    "AI_PLATFORM_POSTGRES_USER",
    "AI_PLATFORM_POSTGRES_PASSWORD",
    "AI_PLATFORM_POSTGRES_SSLMODE",
    "AI_PLATFORM_POSTGRES_CONNECT_TIMEOUT_SECONDS",
    "AI_PLATFORM_POSTGRES_POOL_MIN_SIZE",
    "AI_PLATFORM_POSTGRES_POOL_MAX_SIZE",
    "AI_PLATFORM_POSTGRES_POOL_TIMEOUT_SECONDS",
}


def _load_env(
    path: Path,
    expected_keys: set[str],
) -> dict[str, str]:
    """Load an exact KEY=VALUE environment file."""
    values: dict[str, str] = {}

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line:
            continue

        key, separator, value = raw_line.partition("=")

        if separator != "=":
            raise RuntimeError(f"Malformed environment file: {path}")

        if key in values:
            raise RuntimeError(f"Duplicate environment key: {key}")

        values[key] = value

    if set(values) != expected_keys:
        raise RuntimeError(f"Unexpected environment keys: {path}")

    return values


def _parse_args() -> argparse.Namespace:
    """Parse provisioning arguments."""
    parser = argparse.ArgumentParser()

    config_root = Path.home() / ".config" / "ai-engineering-agent-platform"

    parser.add_argument(
        "--bootstrap-env",
        type=Path,
        default=config_root / "postgres.env",
    )

    parser.add_argument(
        "--runtime-env",
        type=Path,
        default=config_root / "runtime.env",
    )

    return parser.parse_args()


def main() -> None:
    """Create or normalize the managed runtime login."""
    args = _parse_args()

    bootstrap = _load_env(
        args.bootstrap_env,
        EXPECTED_BOOTSTRAP_KEYS,
    )

    runtime = _load_env(
        args.runtime_env,
        EXPECTED_RUNTIME_KEYS,
    )

    runtime_user = runtime["AI_PLATFORM_POSTGRES_USER"]

    runtime_database = runtime["AI_PLATFORM_POSTGRES_DATABASE"]

    runtime_password = runtime["AI_PLATFORM_POSTGRES_PASSWORD"]

    if runtime_user != "ai_platform_runtime":
        raise RuntimeError("Runtime role must be ai_platform_runtime")

    if runtime_database != bootstrap["POSTGRES_DB"]:
        raise RuntimeError("Bootstrap and runtime databases differ")

    if not re.fullmatch(
        r"[0-9a-f]{64}",
        runtime_password,
    ):
        raise RuntimeError("Runtime password must be 256-bit hexadecimal")

    port = int(runtime["AI_PLATFORM_POSTGRES_PORT"])

    connect_timeout = int(runtime["AI_PLATFORM_POSTGRES_CONNECT_TIMEOUT_SECONDS"])

    if not 1 <= port <= 65535:
        raise RuntimeError("PostgreSQL port must be between 1 and 65535")

    if connect_timeout <= 0:
        raise RuntimeError("PostgreSQL connect timeout must be positive")

    with (
        psycopg.connect(
            host=runtime["AI_PLATFORM_POSTGRES_HOST"],
            port=port,
            dbname=bootstrap["POSTGRES_DB"],
            user=bootstrap["POSTGRES_USER"],
            password=bootstrap["POSTGRES_PASSWORD"],
            sslmode=runtime["AI_PLATFORM_POSTGRES_SSLMODE"],
            connect_timeout=connect_timeout,
            application_name=("ai-platform-role-provisioner"),
        ) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            """
                SELECT count(*)
                FROM pg_auth_members AS membership
                JOIN pg_roles AS member
                  ON member.oid = membership.member
                WHERE member.rolname = %s
                """,
            (runtime_user,),
        )

        membership_row = cursor.fetchone()

        if membership_row is None:
            raise RuntimeError("Unable to inspect runtime role memberships")

        if membership_row[0] != 0:
            raise RuntimeError("Runtime role unexpectedly has role memberships")

        cursor.execute(
            """
                SELECT 1
                FROM pg_roles
                WHERE rolname = %s
                """,
            (runtime_user,),
        )

        if cursor.fetchone() is None:
            cursor.execute(
                sql.SQL("CREATE ROLE {}").format(sql.Identifier(runtime_user))
            )

        cursor.execute(
            sql.SQL(
                """
                    ALTER ROLE {}
                    WITH
                        LOGIN
                        NOSUPERUSER
                        NOCREATEDB
                        NOCREATEROLE
                        NOINHERIT
                        NOREPLICATION
                        NOBYPASSRLS
                        CONNECTION LIMIT 10
                        PASSWORD {}
                    """
            ).format(
                sql.Identifier(runtime_user),
                sql.Literal(runtime_password),
            )
        )

        cursor.execute(
            sql.SQL(
                """
                    REVOKE ALL PRIVILEGES
                    ON DATABASE {}
                    FROM {}
                    """
            ).format(
                sql.Identifier(runtime_database),
                sql.Identifier(runtime_user),
            )
        )

        cursor.execute(
            sql.SQL(
                """
                    GRANT CONNECT
                    ON DATABASE {}
                    TO {}
                    """
            ).format(
                sql.Identifier(runtime_database),
                sql.Identifier(runtime_user),
            )
        )


if __name__ == "__main__":
    main()
