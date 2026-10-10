"""Robot-Arm Tennis AI.

Two 8-DOF robots (2-axis floor gantry + 6-DOF arm with a racket) play tennis on a
scaled court in MuJoCo.

Packages:
    sim         MJCF generation, arm kinematics, physics stepping, rules (referee)
    control     ball-flight prediction, shot solving, swing planning, players
    learning    Gymnasium environments (per-shot decisions), PPO training, residual model
    evaluation  matches, metrics, conditions, experiment runner
    viz         offscreen rendering with overlays, HUD, video writing
"""

__version__ = "0.1.0"
ENV_VERSION = "tennis-env-v1"
