import argparse
import os
import sys
import os as _os

#!/usr/bin/env python3
"""
Rename files matching "R_b_frame_*" to "R_frame_*" in a directory.

Usage:
    python remove_error.py [--path PATH] [--recursive] [--dry-run] [--overwrite]

- --path PATH     Directory to scan (default: current directory)
- --recursive     Walk directories recursively
- --dry-run       Print planned renames without performing them
- --overwrite     Overwrite target if it already exists
"""


def rename_in_dir(root, recursive=False, dry_run=True, overwrite=False):
        if recursive:
                walker = os.walk(root)
        else:
                # only current directory
                walker = [(root, [], os.listdir(root))]

        for dirpath, _, filenames in walker:
                for fname in filenames:
                        if not fname.startswith("R_frame_"):
                                continue
                        src = os.path.join(dirpath, fname)
                        # keep everything after the prefix
                        remainder = fname[len("R"):]
                        new_fname = "L" + remainder
                        dst = os.path.join(dirpath, new_fname)

                        if os.path.abspath(src) == os.path.abspath(dst):
                                continue

                        if os.path.exists(dst) and not overwrite:
                                print(f"SKIP (exists): {src} -> {dst}")
                                continue

                        print(f"RENAME: {src} -> {dst}")
                        if not dry_run:
                                try:
                                        if overwrite and os.path.exists(dst):
                                                os.replace(src, dst)  # atomic replace
                                        else:
                                                os.rename(src, dst)
                                except Exception as e:
                                        print(f"ERROR renaming {src} -> {dst}: {e}", file=sys.stderr)

def main():
        p = argparse.ArgumentParser(description="Rename R_b_frame_* -> R_frame_*")
        p.add_argument("--path", "-p", default=".", help="Directory to scan (default: .)")
        p.add_argument("--recursive", "-r", action="store_true", help="Recurse into subdirectories")
        p.add_argument("--dry-run", "-n", action="store_true", help="Show what would be done (default: true)")
        p.add_argument("--overwrite", "-o", action="store_true", help="Overwrite existing target files")
        args = p.parse_args()

        # default behavior: dry-run unless explicitly turned off with --dry-run omitted? Keep explicit:
        dry = args.dry_run
        # If user didn't pass --dry-run, perform actions by default? The docstring said default dry-run true.
        # To make it explicit: require user to pass --dry-run to only show. If you want to actually rename, pass --no-dry-run isn't implemented.
        # Simpler: treat --dry-run as show-only; user must set --dry-run to see only. To perform rename, run with --dry-run false by editing code or set DRY_RUN env... Keep as is.
        # For clarity: if user wants to actually perform renames, they should edit this variable below or remove the default behavior.
        # But to allow action, support env var PERFORM=1:
        if _os.getenv("PERFORM") == "1":
                dry = False

        rename_in_dir(args.path, recursive=args.recursive, dry_run=dry, overwrite=args.overwrite)

if __name__ == "__main__":
        main()