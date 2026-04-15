"""
Compute norm_stats for RoboFactory LiftBarrier data.
Standalone script that doesn't modify openpi code.

Usage:
cd /root/projects/openpi
uv run python /root/projects/RoboFactory/scripts/compute_norm_stats.py \
    --data_root /vepfs-mlp2/c20250510/250404002/robofactory_lerobot \
    --repo_id robofactory_liftbarrier_agent0 \
    --output_dir /root/projects/openpi/assets/pi05_liftbarrier_lora/robofactory_liftbarrier_agent0
"""

import argparse
import json
import pathlib

import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class RunningStats:
    """Online computation of mean and std."""

    def __init__(self):
        self.n = 0
        self.mean = None
        self.m2 = None

    def update(self, x: np.ndarray):
        """Update with a batch of data (..., dim)."""
        x = x.reshape(-1, x.shape[-1])  # flatten to (N, dim)
        for row in x:
            self.n += 1
            if self.mean is None:
                self.mean = row.copy()
                self.m2 = np.zeros_like(row)
            else:
                delta = row - self.mean
                self.mean += delta / self.n
                delta2 = row - self.mean
                self.m2 += delta * delta2

    def get_statistics(self) -> dict:
        if self.n < 2:
            return {"mean": self.mean.tolist(), "std": np.zeros_like(self.mean).tolist()}
        std = np.sqrt(self.m2 / self.n)
        return {"mean": self.mean.tolist(), "std": std.tolist()}


def main(data_root: str, repo_id: str, output_dir: str):
    dataset = LeRobotDataset(repo_id, root=data_root)
    print(f"Dataset: {len(dataset)} frames, {dataset.num_episodes} episodes")

    stats = {"state": RunningStats(), "actions": RunningStats()}

    for i in range(len(dataset)):
        sample = dataset[i]
        stats["state"].update(sample["observation.state"].numpy().reshape(1, -1))
        stats["actions"].update(sample["actions"].numpy().reshape(1, -1))

        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} frames")

    norm_stats = {key: s.get_statistics() for key, s in stats.items()}

    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stats_file = output_path / "norm_stats.json"
    with open(stats_file, "w") as f:
        json.dump({"norm_stats": norm_stats}, f, indent=2)

    print(f"Norm stats saved to {stats_file}")
    for key, s in norm_stats.items():
        print(f"  {key}: mean={np.array(s['mean'][:4])}..., std={np.array(s['std'][:4])}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,
                        default="/vepfs-mlp2/c20250510/250404002/robofactory_lerobot")
    parser.add_argument("--repo_id", type=str, default="robofactory_liftbarrier_agent0")
    parser.add_argument("--output_dir", type=str,
                        default="/root/projects/openpi/assets/pi05_liftbarrier_lora/robofactory_liftbarrier_agent0")
    args = parser.parse_args()
    main(args.data_root, args.repo_id, args.output_dir)
