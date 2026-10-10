"""Run an evaluation experiment from a YAML config (or merge several outputs).

python scripts/evaluate.py --config configs/evaluation/main.yaml
python scripts/evaluate.py --merge results/eval_base results/eval_learned --out results/eval_main
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from tennis.config import from_dict
from tennis.evaluation.experiments import EvalExperimentConfig, merge, run_experiment


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="YAML under configs/evaluation/")
    p.add_argument("--merge", nargs="+", help="experiment output directories to merge")
    p.add_argument("--out", help="output directory (merge / override)")
    p.add_argument("--workers", type=int, default=None)
    a = p.parse_args(argv)
    if a.merge:
        merge([Path(s) for s in a.merge], Path(a.out))
        return
    data = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    if a.out:
        data["output"] = a.out
    if a.workers is not None:
        data["workers"] = a.workers
    cfg = from_dict(EvalExperimentConfig, data)
    df = run_experiment(cfg)
    cols = ["point_win_rate", "in_rate", "return_rate", "avg_rally_hits"]
    print(df.groupby(["controller", "condition"])[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
