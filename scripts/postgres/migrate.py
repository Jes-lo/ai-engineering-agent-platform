"""Execute Alembic without storing PostgreSQL credentials in the repository."""

import argparse
from pathlib import Path

from alembic.config import Config

from alembic import command

EXPECTED_BOOTSTRAP_KEYS = {
    "POSTGRES_USER",
    "POSTGRES_DB",
    "POSTGRES_PASSWORD",
}


def _load_bootstrap_env(
    path: Path,
) -> dict[str, str]:
    """Load the exact bootstrap environment file."""
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

    if set(values) != EXPECTED_BOOTSTRAP_KEYS:
        raise RuntimeError(f"Unexpected environment keys: {path}")

    if not values["POSTGRES_PASSWORD"]:
        raise RuntimeError("Bootstrap PostgreSQL password must not be empty")

    return values


def _parse_args() -> argparse.Namespace:
    """Parse migration command arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "action",
        choices=(
            "upgrade",
            "downgrade",
            "current",
        ),
    )

    parser.add_argument(
        "revision",
        nargs="?",
    )

    parser.add_argument(
        "--bootstrap-env",
        type=Path,
        default=(
            Path.home() / ".config" / "ai-engineering-agent-platform" / "postgres.env"
        ),
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5432,
    )

    parser.add_argument(
        "--sslmode",
        choices=(
            "disable",
            "allow",
            "prefer",
            "require",
            "verify-ca",
            "verify-full",
        ),
        default="prefer",
    )

    parser.add_argument(
        "--connect-timeout",
        type=int,
        default=5,
    )

    return parser.parse_args()


def main() -> None:
    """Execute one supported Alembic operation."""
    args = _parse_args()

    if not 1 <= args.port <= 65535:
        raise RuntimeError("PostgreSQL port must be between 1 and 65535")

    if args.connect_timeout <= 0:
        raise RuntimeError("PostgreSQL connect timeout must be positive")

    bootstrap = _load_bootstrap_env(args.bootstrap_env)

    repository_root = Path(__file__).resolve().parents[2]

    config = Config(str(repository_root / "alembic.ini"))

    config.attributes["connection_settings"] = {
        "host": args.host,
        "port": args.port,
        "database": bootstrap["POSTGRES_DB"],
        "user": bootstrap["POSTGRES_USER"],
        "password": bootstrap["POSTGRES_PASSWORD"],
        "sslmode": args.sslmode,
        "connect_timeout": args.connect_timeout,
    }

    if args.action == "upgrade":
        command.upgrade(
            config,
            args.revision or "head",
        )
        return

    if args.action == "downgrade":
        command.downgrade(
            config,
            args.revision or "base",
        )
        return

    if args.revision is not None:
        raise RuntimeError("'current' does not accept a revision")

    command.current(
        config,
        verbose=True,
    )


if __name__ == "__main__":
    main()
