#!/usr/bin/env python3
"""Generate README.md from README.md.in with current module tree."""

import subprocess
from pathlib import Path

TEMPLATE_PATH = Path("README.md.in")
OUTPUT_PATH = Path("README.md")


def main():
    # Get module tree from inspect_modules.py
    result = subprocess.run(
        ["python", "inspect_modules.py"],
        capture_output=True,
        text=True,
        check=True,
    )
    module_tree = result.stdout.strip()

    # Read template and replace placeholder
    template = TEMPLATE_PATH.read_text()
    readme = template.replace("{{MODULE_TREE}}", module_tree)

    # Write output
    OUTPUT_PATH.write_text(readme)


if __name__ == "__main__":
    main()
