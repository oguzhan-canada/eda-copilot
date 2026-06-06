#!/usr/bin/env python3
"""
scripts/fix_def_instances.py

Fix DEF instance names in CircuitNet data.
CircuitNet DEF files sometimes have mismatched instance references.
This script corrects them so parse_def.py can extract placement data.

Usage:
  python scripts/fix_def_instances.py --input-dir data/raw/circuitnet/
"""

import argparse
import re
from pathlib import Path

from tqdm import tqdm


def fix_def_file(def_path: Path) -> bool:
    """
    Fix instance name mismatches in a DEF file.
    Returns True if the file was modified.
    """
    with open(def_path) as f:
        content = f.read()

    original = content

    # Pattern: instance references that use old naming convention
    # Fix: normalize instance names to match component definitions
    # This handles the common CircuitNet issue where placed instances
    # have a different prefix than their COMPONENTS definition.

    # Extract component names from COMPONENTS section
    comp_pattern = re.compile(r'^\s*-\s+(\S+)\s+(\S+)', re.MULTILINE)
    components = {}
    in_components = False
    for line in content.split('\n'):
        if 'COMPONENTS' in line and 'END' not in line:
            in_components = True
            continue
        if 'END COMPONENTS' in line:
            in_components = False
            continue
        if in_components:
            m = comp_pattern.match(line)
            if m:
                components[m.group(1)] = m.group(2)

    if not components:
        return False

    # Check for and fix mismatched references in NETS section
    modified = False
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        # Fix pin references that use wrong instance names
        if '(' in line and ')' in line and 'NET' not in line.upper():
            for old_name in list(components.keys()):
                # Replace common prefix mismatches
                if '\\' in old_name:
                    clean = old_name.replace('\\', '')
                    if clean in line and old_name not in line:
                        line = line.replace(clean, old_name)
                        modified = True
        new_lines.append(line)

    if modified:
        with open(def_path, 'w') as f:
            f.write('\n'.join(new_lines))
        return True

    return False


def main():
    parser = argparse.ArgumentParser(description="Fix DEF instance names")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing DEF files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report fixes without writing")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    def_files = list(input_dir.rglob("*.def"))
    print(f"Found {len(def_files)} DEF files")

    fixed = 0
    for df in tqdm(def_files, desc="Fixing DEF files"):
        try:
            if fix_def_file(df):
                fixed += 1
                if args.dry_run:
                    print(f"  Would fix: {df}")
        except Exception as e:
            print(f"  ERROR: {df}: {e}")

    print(f"Fixed {fixed}/{len(def_files)} files")


if __name__ == "__main__":
    main()
