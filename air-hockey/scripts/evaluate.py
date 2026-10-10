"""Run the evaluation matrix (controllers x opponents x conditions x seeds).

python scripts/evaluate.py --config configs/evaluation/main.yaml
python scripts/evaluate.py --config configs/evaluation/quick.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hockey.config import from_dict, load_yaml
from hockey.evaluation.experiments import EvalExperimentConfig, merge, run_experiment


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--merge", nargs="+", default=None, help="merge experiment dirs into --output")
    a = p.parse_args(argv)
    if a.merge:
        df = merge([Path(m) for m in a.merge], Path(a.output or "results/eval_main"))
    else:
        data = load_yaml(a.config)
        if a.workers is not None:
            data["workers"] = a.workers
        if a.output is not None:
            data["output"] = a.output
        cfg = from_dict(EvalExperimentConfig, data)
        df = run_experiment(cfg)
    cols = ["controller", "condition", "win", "goals_for", "goals_against", "defense_success_rate"]
    print(
        df[cols].groupby(["controller", "condition"]).mean(numeric_only=True).round(3).to_string()
    )


if __name__ == "__main__":
    main()
