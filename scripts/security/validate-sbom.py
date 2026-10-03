"""Validate the generated CycloneDX SBOM."""

import json
from pathlib import Path

SBOM_PATH = Path("reports/sbom.cdx.json")


def main() -> None:
    """Validate expected SBOM properties."""

    if not SBOM_PATH.is_file():
        raise SystemExit("FAIL: SBOM does not exist")

    data = json.loads(SBOM_PATH.read_text())

    if data.get("bomFormat") != "CycloneDX":
        raise SystemExit("FAIL: unexpected SBOM format")

    if data.get("specVersion") != "1.6":
        raise SystemExit(
            "FAIL: unexpected CycloneDX specification version"
        )

    components = data.get("components")

    if not isinstance(components, list) or not components:
        raise SystemExit("FAIL: SBOM contains no components")

    print(f"bomFormat={data['bomFormat']}")
    print(f"specVersion={data['specVersion']}")
    print(f"components={len(components)}")
    print("PASS: CycloneDX SBOM validated")


if __name__ == "__main__":
    main()
