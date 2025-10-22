#!/usr/bin/env python3
"""Display module tree with public functions/classes."""

import importlib
import inspect
import pkgutil
import tomllib
from pathlib import Path
from typing import Any


def get_packages() -> list[str]:
    """Read package list from pyproject.toml."""
    data = tomllib.loads(Path("pyproject.toml").read_text())
    return data["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]


def get_submodules(pkg_name: str, pkg) -> list[str]:
    """Get all non-test submodules of a package."""
    if not hasattr(pkg, "__path__"):
        return []

    submods = []
    for _importer, modname, _ispkg in pkgutil.walk_packages(
        pkg.__path__, prefix=f"{pkg_name}."
    ):
        if not modname.endswith(".test"):
            submods.append(modname)
    return submods


def get_module_items(modname: str) -> list[str]:
    """Extract public functions and classes from a module."""
    items = []
    try:
        mod = importlib.import_module(modname)
        for name in dir(mod):
            if name.startswith("_"):
                continue
            obj = getattr(mod, name)
            if getattr(obj, "__module__", None) == modname:
                if inspect.isfunction(obj):
                    items.append(f"{name}()")
                elif inspect.isclass(obj):
                    items.append(name)
    except Exception:
        pass
    return sorted(items)


def build_tree(submods: list[str], pkg_name: str) -> dict[str, Any]:
    """Build nested tree structure from flat module list."""
    tree = {}
    for modname in sorted(submods):
        short_name = modname[len(pkg_name) + 1 :]
        parts = short_name.split(".")

        current = tree
        for part in parts:
            if part not in current:
                current[part] = {}
            current = current[part]

        current["__items__"] = get_module_items(modname)
    return tree


def render_tree(tree: dict[str, Any], prefix: str = "") -> None:
    """Recursively render tree structure with box-drawing characters."""
    keys = [k for k in tree.keys() if k != "__items__"]
    for i, key in enumerate(keys):
        is_last = i == len(keys) - 1
        items = tree[key].get("__items__", [])

        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{key}")

        extension = "    " if is_last else "│   "
        for j, item in enumerate(items):
            item_connector = "└── " if j == len(items) - 1 else "├── "
            print(f"{prefix}{extension}{item_connector}{item}")

        if [k for k in tree[key].keys() if k != "__items__"]:
            render_tree(tree[key], prefix + extension)


def main():
    packages = get_packages()
    for pkg_name in sorted(packages):
        pkg = importlib.import_module(pkg_name)
        submods = get_submodules(pkg_name, pkg)

        if not submods:
            continue

        print(pkg_name)
        tree = build_tree(submods, pkg_name)
        render_tree(tree)


if __name__ == "__main__":
    main()
