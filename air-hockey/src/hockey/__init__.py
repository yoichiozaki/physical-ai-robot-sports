"""Adaptive Air Hockey AI.

Packages:
    simulation     MuJoCo model generation, physics stepping, robot motion control, game rules
    observation    player-frame views, sensor noise, observation vector encoding
    controllers    reactive / physics-based / learned controllers and scripted opponents
    learning       Gymnasium environment, rewards, curriculum, PPO training
    evaluation     matches, metrics, experiment runner, plots
    visualization  offscreen rendering, overlays, HUD, video writing
    infrastructure paths, run metadata, environment doctor, Colab helpers
"""

__version__ = "0.1.0"
ENV_VERSION = "hockey-env-v1"
