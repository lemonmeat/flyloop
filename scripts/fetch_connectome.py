"""Download and cache the T4/T5 -> HS/VS and LPi circuit from neuPrint.

Needs NEUPRINT_APPLICATION_CREDENTIALS (in the environment or the repo's .env).
Needs internet access: on a cluster, run it on a login node.

    python tasks.py connectome            # skip if already cached
    python tasks.py connectome --force    # re-query
"""

import argparse
import json

from flyloop.connectome.neuprint_ol import CACHE_DIR, fetch_all


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-query even if cached")
    args = ap.parse_args()
    out = fetch_all(CACHE_DIR, force=args.force)
    meta = json.loads((out / "meta.json").read_text())
    print(f"{meta['dataset']} (server {meta['server_version']}, queried {meta['queried']})")
    for name, n in meta["rows"].items():
        print(f"  {name:<16}{n:>9} rows")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
