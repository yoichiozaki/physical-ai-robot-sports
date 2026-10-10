"""Render an annotated clip (MP4) plus a JSON simulation log.

Examples:
    # Phase 2 acceptance: the robot strikes a stationary puck
    python scripts/render.py --scene strike --out videos/clips/strike.mp4
    # a defended bank shot, slow motion
    python scripts/render.py --scene bank_defense --speed 0.5 --out videos/clips/bank_defense.mp4
    # a match clip between two controllers
    python scripts/render.py --scene match --home physics --away aggressive --duration 20 \
        --out videos/clips/match.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hockey.controllers.factory import make_controller
from hockey.evaluation.conditions import CONDITIONS
from hockey.evaluation.match import play_match
from hockey.observation import Geometry
from hockey.simulation import HockeySim
from hockey.simulation.game import GameConfig
from hockey.visualization.recorder import ClipRecorder, RecorderConfig
from hockey.visualization.scenes import SCENES, scene_setup
from hockey.visualization.video import VideoWriter, probe


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--scene", default="match", choices=["match", *SCENES])
    p.add_argument(
        "--home", default="physics", help="controller spec (see hockey.controllers.factory)"
    )
    p.add_argument("--away", default="stationary")
    p.add_argument("--home-name", default=None)
    p.add_argument("--away-name", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--condition", default="nominal", choices=sorted(CONDITIONS))
    p.add_argument(
        "--duration", type=float, default=None, help="game seconds (default: scene specific)"
    )
    p.add_argument("--speed", type=float, default=1.0, help="playback speed (0.5 = slow motion)")
    p.add_argument("--camera", default="broadcast", choices=["broadcast", "top", "home_view"])
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--title", default="")
    p.add_argument("--caption", default="")
    p.add_argument("--away-overlays", action="store_true")
    p.add_argument("--no-overlays", action="store_true")
    p.add_argument("--stop-on-goal", action="store_true")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    sim = HockeySim()
    geom = Geometry.from_sim_config(sim.cfg)
    home = make_controller(a.home, geom)
    away = make_controller(a.away, geom)
    duration = a.duration or (3.0 if a.scene == "strike" else 4.0 if a.scene != "match" else 20.0)
    rec_cfg = RecorderConfig(
        width=a.width,
        height=a.height,
        fps=a.fps,
        speed=a.speed,
        camera=a.camera,
        home_name=a.home_name or home.name,
        away_name=a.away_name or away.name,
        overlays_home=not a.no_overlays,
        overlays_away=a.away_overlays,
        title=a.title,
        caption=a.caption,
    )
    out = Path(a.out)
    with VideoWriter(out, a.width, a.height, a.fps) as writer:
        recorder = ClipRecorder(writer, rec_cfg)
        try:
            result = play_match(
                sim,
                home,
                away,
                a.seed,
                CONDITIONS[a.condition],
                GameConfig(match_duration=duration),
                on_step=recorder,
                first="home",
                setup=scene_setup(a.scene),
                stop_on_goal=a.stop_on_goal,
            )
        finally:
            recorder.close()
    meta = {"args": vars(a), "summary": result.summary(), "video": probe(out)}
    recorder.save_log(out.with_suffix(".json"), meta)
    print(f"wrote {out} {meta['video']} summary={result.summary()}")


if __name__ == "__main__":
    main()
