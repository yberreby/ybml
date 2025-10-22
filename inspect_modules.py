#!/usr/bin/env python3
"""Display module tree with public functions/classes."""

import importlib
import inspect
import pkgutil
import tomllib
from pathlib import Path


def main():
    mods = tomllib.loads(Path("pyproject.toml").read_text())["tool"]["hatch"]["build"][
        "targets"
    ]["wheel"]["packages"]

    for pkg_name in sorted(mods):
        pkg = importlib.import_module(pkg_name)

        if not hasattr(pkg, "__path__"):
            continue

        submods = []
        for _importer, modname, _ispkg in pkgutil.walk_packages(
            pkg.__path__, prefix=f"{pkg_name}."
        ):
            if not modname.endswith(".test"):
                submods.append(modname)

        print(pkg_name)

        # Build tree structure
        tree = {}
        for modname in sorted(submods):
            short_name = modname[len(pkg_name) + 1 :]
            parts = short_name.split(".")

            # Get items
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

            # Store in tree
            current = tree
            for part in parts:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current["__items__"] = sorted(items)

        # Render tree
        def render(d, prefix="", is_last=True):
            keys = [k for k in d.keys() if k != "__items__"]
            for i, key in enumerate(keys):
                is_last_key = i == len(keys) - 1
                items = d[key].get("__items__", [])

                connector = "└── " if is_last_key else "├── "
                print(f"{prefix}{connector}{key}")

                # Render items
                for j, item in enumerate(items):
                    item_is_last = j == len(items) - 1
                    item_connector = "└── " if item_is_last else "├── "
                    extension = "    " if is_last_key else "│   "
                    print(f"{prefix}{extension}{item_connector}{item}")

                # Recurse
                if len([k for k in d[key].keys() if k != "__items__"]) > 0:
                    extension = "    " if is_last_key else "│   "
                    render(d[key], prefix + extension, is_last_key)

        render(tree)


if __name__ == "__main__":
    main()
