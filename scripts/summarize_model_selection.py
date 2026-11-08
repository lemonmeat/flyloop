"""Rank pretrained flyvis models by how well their T4/T5 tuning matches biology.

Reads the per-model tuning runs written by, for model i = 0..49,

    python tasks.py run openloop_tuning sweep=model_selection index=<i> model.name=flow/0000/<iii>

(<iii> = i with three digits, e.g. 007) and writes runs/model_selection/selection.csv
with one row per model. A model passes when, for the central column, all 8
subtypes respond more to their expected polarity (T4: ON, T5: OFF) and prefer a
direction within --max-pd-error degrees of the known one.

    python tasks.py run summarize_model_selection [--runs runs/model_selection]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf


def summarize(run_dir: Path, max_pd_error: float) -> dict | None:
    summary_csv = run_dir / "summary.csv"
    if not summary_csv.exists():
        return None
    cfg = OmegaConf.load(run_dir / "config.yaml")
    s = pd.read_csv(summary_csv)
    s = s[s.readout == "central"].set_index("cell_type")
    polarity_ok = s.dominant_polarity == s.expected_polarity
    pd_ok = s.pd_error_deg.abs() <= max_pd_error
    row = dict(
        model=cfg.model.name,
        dt=cfg.model.dt,
        passes=bool((polarity_ok & pd_ok).all()),
        n_ok=int((polarity_ok & pd_ok).sum()),
        failing=" ".join(s.index[~(polarity_ok & pd_ok)]),
        mean_dsi=s.dsi.mean(),
        min_dsi=s.dsi.min(),
        max_abs_pd_error=s.pd_error_deg.abs().max(),
    )
    row.update({f"pd_err_{ct}": s.loc[ct, "pd_error_deg"] for ct in s.index})
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs/model_selection")
    ap.add_argument("--max-pd-error", type=float, default=45.0)
    args = ap.parse_args()

    runs = Path(args.runs)
    rows = [r for d in sorted(runs.iterdir()) if d.is_dir()
            if (r := summarize(d, args.max_pd_error))]
    if not rows:
        raise SystemExit(f"no tuning runs found in {runs}")
    table = pd.DataFrame(rows).sort_values(["model", "dt"]).reset_index(drop=True)
    table.to_csv(runs / "selection.csv", index=False)
    cols = ["model", "dt", "passes", "n_ok", "failing", "mean_dsi", "min_dsi",
            "max_abs_pd_error"]
    with pd.option_context("display.width", 160, "display.precision", 2):
        print(table[cols].to_string(index=False))
    print(f"\n-> {runs / 'selection.csv'}")


if __name__ == "__main__":
    main()
