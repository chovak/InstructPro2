#!/usr/bin/env python3
"""Drop vestigial top-level keys from JSONL dataset files, in place.

The released rows carry placeholder keys from earlier versions of the data
pipeline -- `dna` everywhere, plus `antibody: []` in ligand rows and
`ligand: []` in antibody rows. The loader reads only `ligand` and `antigen`,
both with `.get(key, [])`, so removing an empty one changes nothing at load
time.

    # see what would change
    python scripts/drop_keys.py --dry-run release_data/*/*.jsonl

    # drop `dna` (the default)
    python scripts/drop_keys.py release_data/*/*.jsonl

    # also drop the empty cross-modality placeholders
    python scripts/drop_keys.py --keys dna,antibody,ligand --only-empty release_data/*/*.jsonl

--only-empty is a guard: a listed key is removed just when its value is empty,
so a populated `ligand` list is never dropped by accident.
"""

import argparse
import json
import os
import sys
import tempfile

DEFAULT_KEYS = ["dna"]


def main():
    p = argparse.ArgumentParser(description="Drop top-level keys from JSONL files.")
    p.add_argument("files", nargs="+", help=".jsonl files to rewrite in place")
    p.add_argument("--keys", default=",".join(DEFAULT_KEYS),
                   help=f"comma-separated keys to drop (default: {','.join(DEFAULT_KEYS)})")
    p.add_argument("--only-empty", action="store_true",
                   help="only drop a key when its value is empty ([], {}, \"\", null)")
    p.add_argument("--dry-run", action="store_true",
                   help="report counts without rewriting anything")
    args = p.parse_args()

    keys = [k for k in args.keys.split(",") if k]
    print(f"Keys: {keys}" + ("  (only when empty)" if args.only_empty else ""))

    for path in args.files:
        if not os.path.isfile(path):
            print(f"  skip {path} -- not a file")
            continue

        counts = {k: 0 for k in keys}
        kept_nonempty = {k: 0 for k in keys}
        rows = 0

        out_dir = os.path.dirname(os.path.abspath(path))
        tmp = None
        if not args.dry_run:
            fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".tmp")
            os.close(fd)

        try:
            with open(path) as fin:
                fout = open(tmp, "w") if tmp else None
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    rows += 1
                    for k in keys:
                        if k not in row:
                            continue
                        empty = row[k] in ([], {}, "", None)
                        if args.only_empty and not empty:
                            kept_nonempty[k] += 1
                            continue
                        counts[k] += 1
                        if fout:
                            del row[k]
                    if fout:
                        fout.write(json.dumps(row) + "\n")
                if fout:
                    fout.close()

            before = os.path.getsize(path)
            if args.dry_run:
                after = before
            else:
                after = os.path.getsize(tmp)
                os.replace(tmp, path)
                tmp = None

            dropped = ", ".join(f"{k}={counts[k]:,}" for k in keys if counts[k])
            kept = ", ".join(f"{k} kept non-empty={kept_nonempty[k]:,}"
                             for k in keys if kept_nonempty[k])
            print(f"  {os.path.basename(path):32s} {rows:9,d} rows  "
                  f"{before/1e6:7.1f}M -> {after/1e6:7.1f}M  "
                  f"[{dropped or 'nothing to drop'}]" + (f"  ({kept})" if kept else ""))
        finally:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)

    if args.dry_run:
        print("\n(dry run -- nothing rewritten)")


if __name__ == "__main__":
    main()
