"""
Convert RoboFactory zarr data to LeRobot v2 format for pi0.5 fine-tuning.

Compatible with both:
  - LeRobot (Plan A): uses observation.images.agent0 / observation.state / action
  - openpi (Plan B): uses observation.image / observation.state / actions

Usage:
conda activate lerobot  # or openpi
python scripts/convert_zarr_to_lerobot.py --zarr_path /path/to/zarr --output_dir /path/to/output

Example:
python scripts/convert_zarr_to_lerobot.py \
    --zarr_path /root/projects/RoboFactory/data/zarr_data/LiftBarrier-rf_Agent0_150.zarr \
    --output_dir /vepfs-mlp2/c20250510/250404002/robofactory_lerobot
"""

import shutil
import argparse

import numpy as np
import zarr
from PIL import Image
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

REPO_NAME = "robofactory_liftbarrier_agent0"
TASK_DESCRIPTION = "Lift the barrier cooperatively with the other robot"
IMAGE_SIZE = (224, 224)


def main(zarr_path: str, output_dir: str = None, *, push_to_hub: bool = False):
    # Determine output path
    if output_dir:
        output_path = __import__("pathlib").Path(output_dir)
    else:
        output_path = HF_LEROBOT_HOME / REPO_NAME

    # Clean up any existing dataset
    if output_path.exists():
        shutil.rmtree(output_path)

    # Load zarr data
    z = zarr.open(zarr_path, "r")
    head_camera = z["data/head_camera"]  # (N, 3, H, W) uint8
    state = z["data/state"]  # (N, 8) float32
    action = z["data/action"]  # (N, 8) float32
    episode_ends = z["meta/episode_ends"][:]  # (num_episodes,) int64

    num_episodes = len(episode_ends)
    print(f"Loaded {num_episodes} episodes, {head_camera.shape[0]} total frames")

    # Create LeRobot dataset
    # Use key names compatible with both LeRobot and openpi:
    #   - "observation.images.agent0" / "observation.image" → for image (openpi repack maps observation/image)
    #   - "observation.state" → for state
    #   - "actions" → for action (openpi expects "actions", LeRobot convention uses "action" but "actions" also works)
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        root=str(output_path),
        features={
            "observation.images.agent0": {
                "dtype": "image",
                "shape": (*IMAGE_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["observation.state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Convert episodes
    for ep_idx in range(num_episodes):
        ep_end = episode_ends[ep_idx]
        ep_start = episode_ends[ep_idx - 1] if ep_idx > 0 else 0

        for step_idx in range(ep_start, ep_end):
            # Convert image: CHW -> HWC, resize to 224x224
            img_chw = head_camera[step_idx]  # (3, H, W) uint8
            img_hwc = np.transpose(img_chw, (1, 2, 0))  # (H, W, 3)
            img_pil = Image.fromarray(img_hwc)
            img_pil = img_pil.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

            dataset.add_frame(
                {
                    "observation.images.agent0": img_pil,
                    "observation.state": state[step_idx].astype(np.float32),
                    "actions": action[step_idx].astype(np.float32),
                    "task": TASK_DESCRIPTION,
                }
            )

        dataset.save_episode()

        if (ep_idx + 1) % 10 == 0:
            print(f"  Converted {ep_idx + 1}/{num_episodes} episodes")

    print(f"Done! Dataset saved to {output_path}")
    print(f"Total episodes: {num_episodes}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["robofactory", "panda", "liftbarrier"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RoboFactory zarr to LeRobot format")
    parser.add_argument("--zarr_path", type=str, required=True, help="Path to zarr dataset")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for LeRobot dataset")
    parser.add_argument("--push_to_hub", action="store_true", help="Push to HuggingFace Hub")
    args = parser.parse_args()
    main(args.zarr_path, args.output_dir, push_to_hub=args.push_to_hub)
