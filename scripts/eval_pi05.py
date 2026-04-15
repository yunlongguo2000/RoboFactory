"""
Evaluate pi0.5 LoRA fine-tuned model on RoboFactory LiftBarrier task.

Uses Server-Client mode: openpi serve_policy runs the model, this script
runs in the RoboFactory conda env and requests actions via websocket.

Usage:
    # Terminal 1: start inference server
    cd /root/projects/openpi
    uv run scripts/serve_policy.py policy:checkpoint \\
        --policy.config=pi05_liftbarrier_lora \\
        --policy.dir=/vepfs-mlp2/c20250510/250404002/robofactory_checkpoints/pi05_liftbarrier_lora/liftbarrier_agent0/29999

    # Terminal 2: run evaluation
    conda activate RoboFactory
    cd /root/projects/RoboFactory
    python scripts/eval_pi05.py --num_episodes 10 --seed 10000
"""

import sys
sys.path.append("./")

import os
import numpy as np
import yaml
import json
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Union, Annotated
import transforms3d as t3d

import gymnasium as gym
import torch
from PIL import Image
import sapien

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy
from openpi_client import image_tools

from robofactory.tasks import *
from robofactory.utils.wrappers.record import RecordEpisodeMA
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs.pose import to_sapien_pose

import tyro


@dataclass
class Args:
    host: str = "127.0.0.1"
    """Server host."""

    port: int = 8777
    """Server port."""

    num_episodes: int = 10
    """Number of episodes to evaluate."""

    max_steps: int = 250
    """Maximum steps per episode."""

    action_horizon: int = 10
    """Number of actions to execute per inference call."""

    seed: Annotated[int, tyro.conf.arg(aliases=["-s"])] = 10000
    """Starting seed."""

    config: str = "${CONFIG_DIR}/table/lift_barrier.yaml"
    """Environment config file."""

    record_dir: str = "./eval_video/pi05_liftbarrier"
    """Directory to save evaluation videos."""

    obs_mode: str = "rgb"
    """Observation mode."""

    control_mode: str = "pd_joint_pos"
    """Control mode."""

    shader: str = "default"
    """Shader for rendering."""

    num_envs: int = 1
    """Number of parallel environments."""


class Panda1Controller:
    """Controls panda-1 using motion planner: grasp barrier, close gripper, and lift.

    Plans a trajectory at episode start using IK + screw motion, then replays
    it step-by-step during evaluation. Trajectory phases:
      1. Move to grasp pose above barrier
      2. Descend to grasp pose
      3. Close gripper
      4. Lift barrier
    """

    def __init__(self, env: BaseEnv, annotation_data: dict):
        self.env = env
        self.annotation_data = annotation_data
        self.base_env: BaseEnv = env.unwrapped
        self.agent = self.base_env.agent.agents[1]  # panda-1
        self.robot = self.agent.robot
        self.control_mode = self.agent.control_mode

        # Setup planner for panda-1 only
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        from robofactory.utils.mplib_utils import FlexiblePlanner
        self.planner = FlexiblePlanner(
            urdf=self.agent.urdf_path,
            srdf=self.agent.urdf_path.replace(".urdf", ".srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="panda_hand_tcp",
            joint_vel_limits=np.ones(7) * 0.9,
            joint_acc_limits=np.ones(7) * 0.9,
        )
        base_pose = to_sapien_pose(self.robot.pose)
        self.planner.set_base_pose(np.hstack([base_pose.p, base_pose.q]))

        self.trajectory = []
        self.step_idx = 0
        self.OPEN = 1.0
        self.CLOSED = -1.0

    def plan(self, env: BaseEnv):
        """Plan panda-1 trajectory after env.reset(). Must be called before step()."""
        self.trajectory = []
        self.step_idx = 0
        env_unwrapped = env.unwrapped
        barrier = env_unwrapped.barrier

        # Get grasp pose for panda-1 (id=2, from the original solve function)
        grasp_pose_arr = self._get_grasp_pose(barrier, id=2)

        current_qpos = self.robot.get_qpos()[0, :-2].cpu().numpy()

        # Phase 1: Move to 5cm above grasp pose (pre-grasp)
        pre_grasp_pose = grasp_pose_arr.copy()
        pre_grasp_pose[2] += 0.05
        result = self._plan_to_pose(pre_grasp_pose, current_qpos)
        if result["status"] != "Success":
            # Try again
            result = self._plan_to_pose(pre_grasp_pose, current_qpos)
        if result["status"] == "Success":
            for i in range(result["position"].shape[0]):
                self.trajectory.append({
                    "qpos": result["position"][i],
                    "gripper": self.OPEN,
                })
            current_qpos = result["position"][-1]
        else:
            print(f"  [panda-1] WARNING: pre-grasp plan failed, using IK fallback")
            ik_result = self._ik(pre_grasp_pose, current_qpos)
            if ik_result is not None:
                for _ in range(30):
                    self.trajectory.append({"qpos": ik_result[:7], "gripper": self.OPEN})
                current_qpos = ik_result[:7]

        # Phase 2: Descend to grasp pose
        result = self._plan_to_pose(grasp_pose_arr, current_qpos)
        if result["status"] == "Success":
            for i in range(result["position"].shape[0]):
                self.trajectory.append({
                    "qpos": result["position"][i],
                    "gripper": self.OPEN,
                })
            current_qpos = result["position"][-1]
        else:
            print(f"  [panda-1] WARNING: descend plan failed, using IK fallback")
            ik_result = self._ik(grasp_pose_arr, current_qpos)
            if ik_result is not None:
                for _ in range(20):
                    self.trajectory.append({"qpos": ik_result[:7], "gripper": self.OPEN})
                current_qpos = ik_result[:7]

        # Phase 3: Close gripper (hold current position, close over 20 steps)
        for i in range(20):
            gripper = self.OPEN + (self.CLOSED - self.OPEN) * (i + 1) / 20
            self.trajectory.append({
                "qpos": current_qpos.copy(),
                "gripper": gripper,
            })

        # Phase 4: Lift (move up 15cm)
        lift_pose = grasp_pose_arr.copy()
        lift_pose[2] += 0.15
        result = self._plan_to_pose(lift_pose, current_qpos)
        if result["status"] == "Success":
            for i in range(result["position"].shape[0]):
                self.trajectory.append({
                    "qpos": result["position"][i],
                    "gripper": self.CLOSED,
                })
        else:
            print(f"  [panda-1] WARNING: lift plan failed, using IK fallback")
            ik_result = self._ik(lift_pose, current_qpos)
            if ik_result is not None:
                for _ in range(30):
                    self.trajectory.append({"qpos": ik_result[:7], "gripper": self.CLOSED})

        print(f"  [panda-1] Planned trajectory: {len(self.trajectory)} steps")

    def step(self) -> np.ndarray:
        """Get next panda-1 action (8-dim: 7 joints + 1 gripper)."""
        if self.step_idx < len(self.trajectory):
            entry = self.trajectory[self.step_idx]
            self.step_idx += 1
        else:
            # Hold last pose
            if self.trajectory:
                entry = self.trajectory[-1]
            else:
                qpos = self.robot.get_qpos()[0, :-2].cpu().numpy()
                entry = {"qpos": qpos, "gripper": self.CLOSED}

        action = np.concatenate([entry["qpos"], [entry["gripper"]]]).astype(np.float32)
        return action

    def _get_grasp_pose(self, actor, id=2):
        """Compute grasp pose from annotated contact points on the barrier."""
        actor_matrix = actor.pose.to_transformation_matrix()
        actor_matrix = actor_matrix[0]
        local_contact_matrix = np.asarray(self.annotation_data['contact_points_pose'][id])
        local_contact_matrix[:3, 3] *= self.annotation_data['scale']
        convert_matrix = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        global_contact_pose_matrix = actor_matrix.cpu().numpy() @ local_contact_matrix @ convert_matrix
        global_contact_pose_matrix_q = global_contact_pose_matrix[:3, :3]
        global_grasp_pose_p = global_contact_pose_matrix[:3, 3]
        global_grasp_pose_q = t3d.quaternions.mat2quat(global_contact_pose_matrix_q)
        return np.array(list(global_grasp_pose_p) + list(global_grasp_pose_q))

    def _plan_to_pose(self, target_pose, current_qpos):
        """Plan screw motion to target pose."""
        return self.planner.plan_screw(
            target_pose,
            current_qpos,
            time_step=self.base_env.control_timestep,
            use_point_cloud=False,
        )

    def _ik(self, target_pose, current_qpos):
        """IK fallback: return joint angles for target pose."""
        base_pose = self.planner.robot.get_base_pose()
        base_tf = np.eye(4)
        base_tf[0:3, 3] = base_pose[:3]
        base_tf[0:3, 0:3] = t3d.quaternions.quat2mat(base_pose[3:])
        goal_tf = np.eye(4)
        goal_tf[0:3, 3] = target_pose[:3]
        goal_tf[0:3, 0:3] = t3d.quaternions.quat2mat(target_pose[3:])
        goal_tf = np.linalg.inv(base_tf).dot(goal_tf)
        goal_pose_local = np.zeros(7)
        goal_pose_local[:3] = goal_tf[0:3, 3]
        goal_pose_local[3:] = t3d.quaternions.mat2quat(goal_tf[0:3, 0:3])

        status, results = self.planner.IK(goal_pose_local, current_qpos)
        if status == "Success" and len(results) > 0:
            return results[0]
        return None


def get_observation(env_obs: dict) -> dict:
    """Convert ManiSkill observation to pi0.5 input format.

    pi0.5 expects:
        "observation/image": (H, W, 3) uint8  - resized to 224x224
        "observation/state": (8,) float32      - 7 joint angles + 1 gripper
    """
    # Get image from head_camera_agent0
    img = env_obs["sensor_data"]["head_camera_agent0"]["rgb"]  # (1, H, W, 3) uint8 tensor
    img = img.squeeze(0).numpy()  # (H, W, 3) uint8
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))

    # Get joint state (7 joints + gripper)
    qpos = env_obs["agent"]["panda-0"]["qpos"].squeeze(0)[:-2].numpy()  # (7,) - remove last 2 (gripper width dims)
    # Gripper: ManiSkill uses width (0~0.04), training data uses [-1, 1] (open=1, close=-1)
    gripper_width = env_obs["agent"]["panda-0"]["qpos"].squeeze(0)[-2:].mean().item()  # average of two finger widths
    gripper = np.array([np.clip(gripper_width / 0.04 * 2 - 1, -1, 1)], dtype=np.float32)
    state = np.concatenate([qpos, gripper]).astype(np.float32)  # (8,)

    return {
        "observation/image": img,
        "observation/state": state,
    }


def main(args: Args):
    # Bypass proxy for localhost websocket connections (proxy breaks WS handshake)
    os.environ["no_proxy"] = "127.0.0.1,localhost"
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"

    # Connect to inference server
    print(f"Connecting to inference server at {args.host}:{args.port}...")
    policy = websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    print(f"Server metadata: {policy.get_server_metadata()}")

    # Wrap with action chunk broker
    policy = action_chunk_broker.ActionChunkBroker(
        policy=policy,
        action_horizon=args.action_horizon,
    )

    # Create environment
    from robofactory import DIR_MAP
    config_path = args.config
    for k, v in DIR_MAP.items():
        config_path = config_path.replace(k, v)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    env_id = config["task_name"] + "-rf"

    env_kwargs = dict(
        config=config_path,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode="rgb_array",
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        num_envs=args.num_envs,
        sim_backend="auto",
        enable_shadow=True,
    )

    env: BaseEnv = gym.make(env_id, **env_kwargs)

    # Create timestamped subdirectories
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_dir = Path("./eval_debug") / ts
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Wrap with video recorder
    if args.record_dir:
        record_dir = Path(args.record_dir) / ts
        record_dir.mkdir(parents=True, exist_ok=True)
        env = RecordEpisodeMA(
            env,
            str(record_dir),
            save_trajectory=False,
            save_video=True,
            info_on_video=True,
        )

    # Run evaluation
    success_count = 0
    total_steps = 0

    # Load annotation data for panda-1 planner
    from robofactory import ASSET_DIR
    annotation_path = os.path.join(ASSET_DIR, "objects/steel_barrier_annotated/model_data.json")
    with open(annotation_path, "r") as f:
        annotation_data = json.load(f)

    # Create panda-1 controller (plans trajectory per episode)
    panda1_controller = Panda1Controller(env, annotation_data)

    for ep_idx in range(args.num_episodes):
        ep_seed = args.seed + ep_idx
        obs, _ = env.reset(seed=ep_seed)
        policy.reset()

        # Plan panda-1 trajectory for this episode (barrier position is now known)
        panda1_controller.plan(env)

        print(f"\nEpisode {ep_idx + 1}/{args.num_episodes} (seed={ep_seed})")
        ep_actions = []
        ep_states = []
        ep_panda1_actions = []
        ep_panda1_states = []
        ep_infos = []

        for step in range(args.max_steps):
            # Get observation in pi0.5 format
            obs_dict = get_observation(obs)
            ep_states.append(obs_dict["observation/state"].copy())

            # Log first step observation details
            if step == 0:
                print(f"  [INIT] state={np.array2string(obs_dict['observation/state'], precision=4, suppress_small=True)}")
                print(f"  [INIT] image shape={obs_dict['observation/image'].shape}, dtype={obs_dict['observation/image'].dtype}")
                print(f"  [INIT] image min/max={obs_dict['observation/image'].min()}/{obs_dict['observation/image'].max()}")
                # Save first frame image
                Image.fromarray(obs_dict["observation/image"]).save(debug_dir / f"ep{ep_idx}_step0.png")

            # Request action from server (ActionChunkBroker handles chunking)
            action = policy.infer(obs_dict)
            action_np = action["actions"]  # (8,) single action from chunk
            ep_actions.append(action_np.copy())

            # Get panda-1 action from motion planner trajectory
            panda1_action = panda1_controller.step()
            ep_panda1_actions.append(panda1_action.copy())
            # Record panda-1 actual state
            panda1_qpos = obs["agent"]["panda-1"]["qpos"].squeeze(0).numpy()
            panda1_grip = np.clip(panda1_qpos[-2:].mean() / 0.04 * 2 - 1, -1, 1)
            panda1_state = np.concatenate([panda1_qpos[:7], [panda1_grip]]).astype(np.float32)
            ep_panda1_states.append(panda1_state.copy())
            action_dict = {"panda-0": action_np, "panda-1": panda1_action}
            obs, reward, terminated, truncated, info = env.step(action_dict)
            env.render()

            total_steps += 1

            # Log action and state periodically
            if step < 5 or step % 50 == 0:
                state_str = np.array2string(obs_dict["observation/state"], precision=4, suppress_small=True)
                action_str = np.array2string(action_np, precision=4, suppress_small=True)
                delta = np.abs(action_np - obs_dict["observation/state"])
                delta_str = np.array2string(delta, precision=4, suppress_small=True)
                panda1_in_traj = panda1_controller.step_idx <= len(panda1_controller.trajectory)
                print(f"  Step {step+1}: state={state_str}")
                print(f"           action={action_str}")
                print(f"           |act-st|={delta_str}")
                print(f"           panda1: grip_action={panda1_action[7]:.4f}, grip_actual={panda1_grip:.4f}, in_traj={panda1_in_traj}")

            # Log success-related info from env
            if step % 50 == 0 or info.get("success", False):
                ep_infos.append({"step": step, "success": info.get("success", False), **{k: v for k, v in info.items() if k != "success"}})

            if info.get("success", False):
                success_count += 1
                print(f"  Step {step + 1}: SUCCESS!")
                break

            if terminated or truncated:
                print(f"  Step {step + 1}: Episode ended (terminated={terminated}, truncated={truncated})")
                break
        else:
            print(f"  Max steps ({args.max_steps}) reached without success.")

        # Save episode debug data
        ep_actions = np.array(ep_actions)
        ep_states = np.array(ep_states)
        ep_panda1_actions = np.array(ep_panda1_actions)
        ep_panda1_states = np.array(ep_panda1_states)
        np.savez(debug_dir / f"ep{ep_idx}_data.npz",
                 actions=ep_actions, states=ep_states,
                 panda1_actions=ep_panda1_actions, panda1_states=ep_panda1_states)
        print(f"  [DEBUG] panda-0 action mean={np.array2string(ep_actions.mean(axis=0), precision=4, suppress_small=True)}")
        print(f"  [DEBUG] panda-0 state  mean={np.array2string(ep_states.mean(axis=0), precision=4, suppress_small=True)}")
        print(f"  [DEBUG] panda-1 action mean={np.array2string(ep_panda1_actions.mean(axis=0), precision=4, suppress_small=True)}")
        print(f"  [DEBUG] panda-1 state  mean={np.array2string(ep_panda1_states.mean(axis=0), precision=4, suppress_small=True)}")
        # Panda-1 gripper analysis
        panda1_grip_actions = ep_panda1_actions[:, 7]
        panda1_grip_states = ep_panda1_states[:, 7]
        traj_len = len(panda1_controller.trajectory)
        print(f"  [DEBUG] panda-1 grip: action_min={panda1_grip_actions.min():.4f}, state_min={panda1_grip_states.min():.4f}, traj_len={traj_len}")
        if traj_len < len(panda1_grip_states):
            hold_actions = panda1_grip_actions[traj_len:]
            hold_states = panda1_grip_states[traj_len:]
            print(f"  [DEBUG] panda-1 hold phase ({len(hold_actions)} steps): action_mean={hold_actions.mean():.4f}, state_mean={hold_states.mean():.4f}, state_min={hold_states.min():.4f}")

    env.close()

    # Print results
    print(f"\n{'=' * 50}")
    print(f"Evaluation Results")
    print(f"{'=' * 50}")
    print(f"Episodes:     {args.num_episodes}")
    print(f"Successes:    {success_count}")
    print(f"Success Rate: {success_count / args.num_episodes:.1%}")
    print(f"Avg Steps:    {total_steps / args.num_episodes:.1f}")
    if args.record_dir:
        print(f"Videos saved: {record_dir}/")
    print(f"Debug data:   {debug_dir}/")


if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)
