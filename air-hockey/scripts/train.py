"""Train a PPO agent.

Examples (inside the Dev Container):
    python scripts/train.py --config configs/training/smoke.yaml
    python scripts/train.py --config configs/training/ppo_hybrid.yaml
    python scripts/train.py --config configs/training/ppo_direct.yaml \
        --resume checkpoints/ppo_direct/checkpoints/ckpt_4000000_steps.zip
"""

from hockey.learning.train import main

if __name__ == "__main__":
    main()
