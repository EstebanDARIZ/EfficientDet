from pathlib import Path
import argparse
import sys

#!/usr/bin/env python3
"""
clean_filename.py

Rename files in a directory by adding a prefix (single letter or string) before each filename.

Usage:
    python clean_filename.py /path/to/dir A        # dry-run by default
    python clean_filename.py /path/to/dir A -r -x  # recursive, execute
"""


def add_prefix_to_files(directory: Path, prefix: str, recursive: bool = False, execute: bool = False, keep_unique: bool = True):
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    if recursive:
        iterator = directory.rglob('*')
    else:
        iterator = directory.iterdir()

    actions = []
    for p in iterator:
        if not p.is_file():
            continue
        new_name = prefix + p.name
        new_path = p.with_name(new_name)

        # If target already exists, optionally make a unique name
        if new_path.exists() and new_path != p:
            if keep_unique:
                stem = new_path.stem
                suffix = new_path.suffix
                counter = 1
                # Build a unique filename like prefix + original_stem + _1 + suffix
                while True:
                    candidate = new_path.with_name(f"{stem}_{counter}{suffix}")
                    if not candidate.exists():
                        new_path = candidate
                        break
                    counter += 1
            else:
                # skip renaming to avoid overwrite
                actions.append((p, None, "target exists, skipped"))
                continue

        if new_path == p:
            actions.append((p, None, "no change"))
            continue

        actions.append((p, new_path, "will rename" if not execute else "renamed"))
        if execute:
            p.rename(new_path)

    return actions

def main():
    parser = argparse.ArgumentParser(description="Add a prefix to every file in a directory.")
    parser.add_argument("directory", nargs="?", default=".", help="Target directory (default: current)")
    parser.add_argument("prefix", help="Prefix to add to filenames (e.g. a single letter)")
    parser.add_argument("-r", "--recursive", action="store_true", help="Recurse into subdirectories")
    parser.add_argument("-x", "--execute", action="store_true", help="Perform the renames (default is dry-run)")
    parser.add_argument("--no-unique", action="store_true", help="Do not auto-uniqueify colliding names; skip instead")
    args = parser.parse_args()

    directory = Path(args.directory).resolve()
    try:
        actions = add_prefix_to_files(directory, args.prefix, recursive=args.recursive, execute=args.execute, keep_unique=not args.no_unique)
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    for src, dst, status in actions:
        if dst is None:
            print(f"{src} -> (skipped) : {status}")
        # else:
            # print(f"{src} -> {dst} : {status}")

    if not args.execute:
        print("\nDry-run complete. Rerun with -x to apply changes.")

if __name__ == "__main__":
    main()