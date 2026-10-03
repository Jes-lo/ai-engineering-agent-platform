"""Static checks for architectural dependency direction."""

import ast
from pathlib import Path

PACKAGE_NAME = "ai_engineering_agent_platform"
PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / PACKAGE_NAME


def _project_imports_from_source(
    path: Path,
    source: str,
) -> set[str]:
    """Return resolved imports referenced by Python source."""
    tree = ast.parse(
        source,
        filename=str(path),
    )
    imports: set[str] = set()

    relative = path.relative_to(PACKAGE_ROOT).with_suffix("")

    package_parts = (
        PACKAGE_NAME,
        *relative.parts[:-1],
    )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
            continue

        if not isinstance(node, ast.ImportFrom):
            continue

        module: str | None

        if node.level == 0:
            module = node.module
        else:
            ascents = node.level - 1

            if ascents >= len(package_parts):
                imports.add("<invalid-relative-import>")
                continue

            base_parts = package_parts[: len(package_parts) - ascents]

            if node.module is None:
                module = ".".join(base_parts)
            else:
                module = ".".join(
                    (
                        *base_parts,
                        *node.module.split("."),
                    )
                )

        if module is None:
            continue

        imports.add(module)

        for alias in node.names:
            if alias.name == "*":
                continue

            imports.add(f"{module}.{alias.name}")

    return imports


def _project_imports(path: Path) -> set[str]:
    """Return resolved imports referenced by a Python file."""
    return _project_imports_from_source(
        path,
        path.read_text(encoding="utf-8"),
    )


def _assert_no_forbidden_imports(
    relative_directory: str,
    forbidden_prefixes: tuple[str, ...],
) -> None:
    """Assert that a package does not depend on forbidden layers."""
    directory = PACKAGE_ROOT / relative_directory

    for path in directory.rglob("*.py"):
        for imported in _project_imports(path):
            assert not imported.startswith(forbidden_prefixes), (
                f"{path.relative_to(PACKAGE_ROOT)} imports forbidden layer {imported}"
            )


def test_import_parser_resolves_relative_and_root_aliases() -> None:
    """Architecture analysis must not be bypassed by import syntax."""
    synthetic_path = PACKAGE_ROOT / "contracts" / "synthetic.py"

    imports = _project_imports_from_source(
        synthetic_path,
        """
from . import provider
from .. import adapters
from ai_engineering_agent_platform import app
""",
    )

    assert f"{PACKAGE_NAME}.contracts.provider" in imports
    assert f"{PACKAGE_NAME}.adapters" in imports
    assert f"{PACKAGE_NAME}.app" in imports


def test_contracts_are_independent_of_higher_layers() -> None:
    """Contracts must remain at the lowest project dependency layer."""
    _assert_no_forbidden_imports(
        "contracts",
        (
            f"{PACKAGE_NAME}.adapters",
            f"{PACKAGE_NAME}.api",
            f"{PACKAGE_NAME}.app",
            f"{PACKAGE_NAME}.config",
            f"{PACKAGE_NAME}.domain",
        ),
    )


def test_domain_does_not_depend_on_outer_layers() -> None:
    """Domain code may use contracts but not infrastructure layers."""
    _assert_no_forbidden_imports(
        "domain",
        (
            f"{PACKAGE_NAME}.adapters",
            f"{PACKAGE_NAME}.api",
            f"{PACKAGE_NAME}.app",
            f"{PACKAGE_NAME}.config",
        ),
    )


def test_services_depend_only_on_domain_and_contracts() -> None:
    """Application services must not depend on concrete outer layers."""
    _assert_no_forbidden_imports(
        "services",
        (
            f"{PACKAGE_NAME}.adapters",
            f"{PACKAGE_NAME}.api",
            f"{PACKAGE_NAME}.app",
            f"{PACKAGE_NAME}.config",
            f"{PACKAGE_NAME}.runtime",
        ),
    )
