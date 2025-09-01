from typing import Any, Dict, Tuple

import numpy as np
import sapien
import torch
import yaml

from mani_skill.agents.robots import Panda
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
import utils.scenes

@register_env("ThreeRobotsPlaceShoes-rf", max_episode_steps=1200)
class ThreeRobotsPlaceShoesEnv(BaseEnv):

    SUPPORTED_ROBOTS = [("panda", "panda", "panda")]
    agent: MultiAgent[Tuple[Panda, Panda, Panda]]
    goal_thresh = 0.025
    box_goal_radius = 0.15

    def __init__(
        self, *args, robot_uids=("panda", "panda", "panda"), **kwargs
    ):
        assert 'config' in kwargs
        with open(kwargs['config'], 'r', encoding='utf-8') as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        del kwargs['config']
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**19,
                max_rigid_contact_count=2**21,
            )
        )

    @property
    def _default_sensor_configs(self):
        camera_cfg = self.cfg.get('cameras', {})
        sensor_cfg = camera_cfg.get('sensor', [])
        all_camera_configs =[]
        for sensor in sensor_cfg:
            pose = sensor['pose']
            if pose['type'] == 'pose':
                sensor['pose'] = sapien.Pose(*pose['params'])
            elif pose['type'] == 'look_at':
                sensor['pose'] = sapien_utils.look_at(*pose['params'])
            all_camera_configs.append(CameraConfig(**sensor))
        return all_camera_configs

    @property
    def _default_human_render_camera_configs(self):
        camera_cfg = self.cfg.get('cameras', {})
        render_cfg = camera_cfg.get('human_render', [])
        all_camera_configs =[]
        for render in render_cfg:
            pose = render['pose']
            if pose['type'] == 'pose':
                render['pose'] = sapien.Pose(*pose['params'])
            elif pose['type'] == 'look_at':
                render['pose'] = sapien_utils.look_at(*pose['params'])
            all_camera_configs.append(CameraConfig(**render))
        return all_camera_configs

    def _load_agent(self, options: dict):
        init_poses = []
        for agent_cfg in self.cfg['agents']:
            init_poses.append(sapien.Pose(p=agent_cfg['pos']['ppos']['p']))
        super()._load_agent(options, init_poses)

    def _load_scene(self, options: dict):
        scene_name = self.cfg['scene']['name']
        scene_builder = getattr(utils.scenes, f'{scene_name}SceneBuilder')
        self.scene_builder = scene_builder(env=self, cfg=self.cfg)
        self.scene_builder.build()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.shoe_box.set_qpos(np.array(self.cfg['objects'][2]['qpos']))
            self.scene_builder.initialize(env_idx)

    @property
    def left_agent(self) -> Panda:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> Panda:
        return self.agent.agents[1]

    @property
    def back_agent(self) -> Panda:
        return self.agent.agents[2]

    def evaluate(self):
        # Check if shoes are inside the box
        shoe_left_in_box = self._check_shoe_in_box(self.shoe_left, self.shoe_box)
        shoe_right_in_box = self._check_shoe_in_box(self.shoe_right, self.shoe_box)
        
        # Check if shoes are still grasped
        is_shoe_left_grasped = self.left_agent.is_grasping(self.shoe_left)
        is_shoe_right_grasped = self.right_agent.is_grasping(self.shoe_right)

        lid_on_box = torch.tensor(self.shoe_box.qpos[0][2] < 0.01)  # Placeholder for lid check

        success = (
            shoe_left_in_box * 
            shoe_right_in_box * 
            lid_on_box * 
            (~is_shoe_left_grasped) * 
            (~is_shoe_right_grasped)
        )
        
        return {
            "shoe_left_in_box": shoe_left_in_box,
            "shoe_right_in_box": shoe_right_in_box,
            "lid_on_box": lid_on_box,
            "is_shoe_left_grasped": is_shoe_left_grasped,
            "is_shoe_right_grasped": is_shoe_right_grasped,
            "success": success.bool(),
        }

    def _check_shoe_in_box(self, shoe, box):
        """Check if the shoe is inside the box."""
        shoe_pos = shoe.pose.p
        box_pos = box.pose.p

        distance = torch.linalg.norm(shoe_pos[:, :2] - box_pos[:, :2], axis=1)
        in_box = (distance < 0.14) & (shoe_pos[:, 2] < 0.08)
        return in_box

    def _check_lid_on_box(self, lid, box):
        """Check if the lid is on top of the box."""
        lid_pos = lid.pose.p
        box_pos = box.pose.p
        
        distance = torch.linalg.norm(lid_pos[:, :2] - box_pos[:, :2], axis=1)
        height_diff = lid_pos[:, 2] - box_pos[:, 2]
        lid_on_box = (distance < 0.05) & (height_diff > 0.12) & (height_diff < 0.18)
        return lid_on_box

    def _get_obs_extra(self, info: Dict):
        return {}

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return {}

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return {}