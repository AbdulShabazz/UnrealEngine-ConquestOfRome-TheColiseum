#!/usr/bin/env python3
"""
flatten_taxonomy.py

Read a JSON taxonomy (the structure you posted) and output a flat list of
double-underscore-separated name-paths, e.g.:

    soundFxTaxonomy__Ambient Environment
    soundFxTaxonomy__Ambient Environment__Coliseum Ambient
    soundFxTaxonomy__Footsteps & Movement__Stone / Marble Floor
    ...

The script works for any depth of nesting - if you ever add a third level of
subcategories it will be handled automatically.
"""

import json
import sys
from pathlib import Path
from typing import Any, List


# ----------------------------------------------------------------------
# 1  Recursive walker that collects every node that has a "name"
# ----------------------------------------------------------------------
def _collect_paths(
    node: Any,
    cur_path: List[str],
    out: List[str],
    delimiter: str = "__",
) -> None:
    """
    Walk *node* (dict / list / scalar) and append a ``delimiter``-joined
    path to *out* each time a dictionary contains a ``name`` key.

    Parameters
    ----------
    node
        Current JSON fragment.
    cur_path
        Accumulated list of name components that lead to *node*.
    out
        List that receives the fully-qualified paths.
    delimiter
        String used to separate the components (default: ``__``).
    """
    if isinstance(node, dict):
        # If this dict has a "name", extend the path.
        name = node.get("name")
        new_path = cur_path + [name] if name else cur_path

        # Every node that *has* a name should appear in the output,
        # even if it also has children.
        if name:
            out.append(delimiter.join(new_path))

        # Recurse into anything that can contain further dicts:
        # - subcategories (the most common case)
        # - any other list / dict value that isn’t just a scalar.
        for key, value in node.items():
            if isinstance(value, (list, dict)):
                _collect_paths(value, new_path, out, delimiter)

    elif isinstance(node, list):
        for item in node:
            _collect_paths(item, cur_path, out, delimiter)

    # Scalars (str, int, etc.) are ignored - they can’t hold a "name". 


# ----------------------------------------------------------------------
# 2  Public helper - start the walk at the root key
# ----------------------------------------------------------------------
def flatten_taxonomy(data: Any, root_key: str = "soundFxTaxonomy") -> List[str]:
    """
    Entry point that receives the whole JSON payload (already loaded) and
    returns a flat list of name-paths.

    Parameters
    ----------
    data
        The JSON object read from the file.
    root_key
        Top-level container that should become the first element of every
        path (defaults to the key used in your example).

    Returns
    -------
    List[str]
        All paths, e.g. ``soundFxTaxonomy__Ambient Environment__Coliseum Ambient``.
    """
    results: List[str] = []

    # The example JSON is a *list* whose first (and only) element is a dict
    # that holds the root key.
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict) and root_key in entry:
                _collect_paths(entry[root_key], [root_key], results)
    elif isinstance(data, dict) and root_key in data:
        _collect_paths(data[root_key], [root_key], results)
    else:
        # Fallback - treat the whole payload as a container.
        _collect_paths(data, [], results)

    # Preserve order while removing accidental duplicates.
    seen = set()
    uniq = []
    for p in results:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


# ----------------------------------------------------------------------
# 3  CLI - simple, zero-dependency entry point
# ----------------------------------------------------------------------
def _cli() -> None:
    if len(sys.argv) != 2:
        print(
            "Usage: python flatten_taxonomy.py <path-to-taxonomy.json>\n"
            "The script prints the flattened list to STDOUT (one line per entry)."
        )
        sys.exit(1)

    json_path = Path(sys.argv[1])
    if not json_path.is_file():
        print(f"X  File not found: {json_path}")
        sys.exit(2)

    with json_path.open(encoding="utf-8") as fp:
        try:
            raw = json.load(fp)
        except json.JSONDecodeError as exc:
            print(f"X  Could not parse JSON: {exc}")
            sys.exit(3)

    flat = flatten_taxonomy(raw)

    # Print each entry on its own line - easy to pipe into a file or another tool.
    for line in flat:
        print(line)


if __name__ == "__main__":
    _cli()
