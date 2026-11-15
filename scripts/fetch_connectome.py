"""Download and cache the connectome data: the T4/T5 -> HS/VS and LPi circuit from
neuPrint, and the eye map (the viewing direction of every ommatidium, Zhao et al.
2025) from GitHub.

The neuPrint part needs NEUPRINT_APPLICATION_CREDENTIALS (in the environment or
the repo's .env); the eye map needs no account. Both need internet access: on a
cluster, run this on a login node.

    python tasks.py connectome            # skip what is already cached
    python tasks.py connectome --force    # download again
"""

import argparse
import json

from flyloop.connectome.eyemap import EYEMAP_DIR, fetch_eyemap
from flyloop.connectome.neuprint_ol import CACHE_DIR, fetch_all


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="download again even if cached")
    args = ap.parse_args()

    out = fetch_eyemap(EYEMAP_DIR, force=args.force)
    meta = json.loads((out / "meta.json").read_text())
    print(f"eye map: {meta['ommatidia']} ommatidia of the {meta['eye']} eye, scan "
          f"{meta['scan']} ({meta['source']} at {meta['commit'][:7]}, {meta['license']})")
    print(f"-> {out}")

    out = fetch_all(CACHE_DIR, force=args.force)
    meta = json.loads((out / "meta.json").read_text())
    print(f"{meta['dataset']} (server {meta['server_version']}, queried {meta['queried']})")
    for name, n in meta["rows"].items():
        print(f"  {name:<16}{n:>9} rows")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
