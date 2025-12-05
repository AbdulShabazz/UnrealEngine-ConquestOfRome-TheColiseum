#!/usr/bin/env python3
"""
Flatten a hierarchical "sound-FX taxonomy" JSON into a list of
dot-separated (or double-underscore) name-paths.

Input (example) - a file that contains the JSON you posted:
{
    "soundFxTaxonomy": [
        {
            "name": "Ambient Environment",
            "subcategories": [ ... ]
        },
        {
            "name": "Footsteps & Movement",
            "subcategories": [ ... ]
        }
    ]
}

Output (example) - a Python list (or a plain-text file, one entry per line)!

Usage:

$ python flatten_taxonomy.py taxonomy.json
[
  "soundFxTaxonomy__Ambient Environment__Coliseum Ambient",
  "soundFxTaxonomy__Ambient Environment__City / Marketplace Ambient",
  "soundFxTaxonomy__Footsteps & Movement__Stone / Marble Floor",
  "soundFxTaxonomy__Footsteps & Movement__Sand / Dust",
  "soundFxTaxonomy__Footsteps & Movement__Wooden Deck / Bridge",
  "soundFxTaxonomy__Footsteps & Movement__Grass / Meadow",
  "soundFxTaxonomy__Footsteps & Movement__Water / Shallow Stream"
]

"""

import json
import sys
from pathlib import Path
from typing import Any, List


# ----------------------------------------------------------------------
# 1  Recursive walker
# ----------------------------------------------------------------------
def _walk(node: Any, path: List[str], out: List[str]) -> None:
    """
    Recursively walk a JSON node, building a path whenever a dict contains a
    ``name`` key.

    * ``node`` - current JSON fragment (dict, list or scalar)
    * ``path`` - accumulated list of name components
    * ``out``  - list that will receive the final ``__``-joined strings
    """
    # ---- Dictionaries ----------------------------------------------------
    if isinstance(node, dict):
        # If the dict has a "name" field we extend the current path.
        name = node.get("name")
        new_path = path + [name] if name else path

        # Find children that are themselves containers (list / dict).  We
        # deliberately ignore simple scalars such as "description", "tags", ...
        child_containers = [
            v for k, v in node.items()
            if k != "name" and isinstance(v, (list, dict))
        ]

        if child_containers:
            # Not a leaf - keep walking deeper.
            for child in child_containers:
                _walk(child, new_path, out)
        else:
            # Leaf node - store the fully-qualified name.
            out.append("__".join(new_path))

    # ---- Lists -----------------------------------------------------------
    elif isinstance(node, list):
        for item in node:
            _walk(item, path, out)

    # ---- Anything else (int, str, ...) is ignored -------------------------
    else:
        return


# ----------------------------------------------------------------------
# 2  Public helper
# ----------------------------------------------------------------------
def flatten_taxonomy(data: Any, root_key: str = "soundFxTaxonomy") -> List[str]:
    """
    Entry point that receives the raw JSON (already loaded with ``json.load``)
    and returns a flat list of name-paths.

    Parameters
    ----------
    data
        The top-level JSON object (usually a ``dict`` or a ``list`` that
        contains the root key).
    root_key
        The name of the top-level container that should become the first
        element of every path (defaults to the key used in the example).

    Returns
    -------
    List[str]
        All leaf-paths in the form ``root__Category__Subcategory``.
    """
    results: List[str] = []

    # The JSON you posted is a list whose first element holds the root key.
    # Accept both a list or a dict for flexibility.
    if isinstance(data, list):
        # Look for the dictionary that contains ``root_key``.
        for entry in data:
            if isinstance(entry, dict) and root_key in entry:
                _walk(entry[root_key], [root_key], results)
    elif isinstance(data, dict) and root_key in data:
        _walk(data[root_key], [root_key], results)
    else:
        # Fallback - treat the whole payload as a container.
        _walk(data, [], results)

    # Remove possible duplicates while preserving order.
    seen = set()
    unique = []
    for item in results:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


# ----------------------------------------------------------------------
#   Command-line interface
# ----------------------------------------------------------------------
def _cli() -> None:
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(
            "Usage: python flatten_taxonomy.py <input-json> [output-json]\n"
            "If *output-json* is omitted the result is printed to stdout."
        )
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.is_file():
        print(f"X  Target Filename not found: {input_path}")
        sys.exit(2)

    # Load JSON (utf-8 is the de-facto default for modern projects)
    with input_path.open(encoding="utf-8") as fp:
        try:
            raw = json.load(fp)
        except json.JSONDecodeError as exc:
            print(f"X  Failed to parse JSON - {exc}")
            sys.exit(3)

        flat = flatten_taxonomy(raw)

        # --------------------------------------------------------------
        # Write / print the result
        # --------------------------------------------------------------
        if len(sys.argv) == 3:                     # write to a file
            out_path = Path(sys.argv[2])
            with out_path.open("w", encoding="utf-8") as fp:
                for line in flat:
                    fp.write(line + "\n")
            print(f"  Written {len(flat)} lines to {out_path}")
        else:                                       # just pretty-print
            print(f"X  Destination Filename not found: ... \n\n")
            print(json.dumps(flat, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    print(
        "\n"
        "Usage: py -m flatten_taxonomy <input-json> [output-json]\n"
        "If *output-json* is omitted the result is printed to stdout."
    )
    _cli()
