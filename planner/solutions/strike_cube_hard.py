import numpy as np
import sapien

from tasks import StrikeCubeHardEnv
from planner.motionplanner import PandaArmMotionPlanningSolver

def solve(env: StrikeCubeHardEnv, seed=101, debug=False, vis=False):
    LIFT_HEIGHT = 0.2
    PRE_DIS = 0.05
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=[agent.robot.pose for agent in env.agent.agents],
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        is_multi_agent=True
    )
    env = env.unwrapped
    # Task decomposition
    # use agent 1 to move the cube to the origin
    grasp_poseA = planner.get_grasp_pose_from_obb(env.cube, 1)
    grasp_poseA[2] += 0.3
    planner.move_to_pose_with_screw([grasp_poseA], move_id=[1])
    grasp_poseA[2] -= 0.35
    planner.move_to_pose_with_screw([grasp_poseA], move_id=[1])
    planner.close_gripper([1])
    grasp_poseB = planner.get_grasp_pose_for_stack(grasp_poseA, env.goal_region)
    planner.move_to_pose_with_screw([grasp_poseB], move_id=[1])
    planner.open_gripper([1])
    planner.move_to_pose_with_screw([grasp_poseA], move_id=[1])
    # use agent 0 to strike the cube
    pose1 = planner.get_grasp_pose_w_labeled_direction(actor=env.hammer, actor_data=env.annotation_data['hammer'], pre_dis=-0.05)  # hammer pre grasp pose for agent 0
    pose2 = planner.get_grasp_pose_w_labeled_direction(actor=env.hammer, actor_data=env.annotation_data['hammer'], pre_dis=0)      # hammer grasp pose for agent 0
    planner.move_to_pose_with_screw(pose1, move_id=[0])
    planner.move_to_pose_with_screw(pose2, move_id=[0])
    planner.close_gripper([0])
    pose2[2] += LIFT_HEIGHT
    planner.move_to_pose_with_screw(pose2, move_id=[0])
    pre_strike_block_pose = planner.get_grasp_pose_from_goal_point_and_direction(
                                actor=env.hammer, 
                                actor_data=env.annotation_data['hammer'], 
                                actor_functional_point_id=0, 
                                endpose=env.agent.agents[0].tcp.pose, 
                                target_point=env.cube.pose[:3],
                                pre_dis=PRE_DIS,
                            )
    if pre_strike_block_pose is not None:
        planner.move_to_pose_with_screw(pre_strike_block_pose, move_id=[0])
        pre_strike_block_pose[2] -= PRE_DIS
        planner.move_to_pose_with_screw(pre_strike_block_pose, move_id=[0])
    res = planner.close_gripper([0])
    planner.close()

    return res
