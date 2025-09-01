import numpy as np
import sapien

from tasks import ThreeRobotsPlaceShoesEnv
from planner.motionplanner import PandaArmMotionPlanningSolver

def solve(env: ThreeRobotsPlaceShoesEnv, seed=102, debug=False, vis=False):
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

    # Step 1: Two robots grasp the left and right shoes respectively
    grasp_pose_left_shoe = planner.get_grasp_pose_from_obb(env.shoe_left, 0)
    grasp_pose_right_shoe = planner.get_grasp_pose_from_obb(env.shoe_right, 1)
    grasp_pose_left_shoe[2] += 0.12
    grasp_pose_right_shoe[2] += 0.12
    planner.move_to_pose_with_screw([grasp_pose_left_shoe, grasp_pose_right_shoe], move_id=[0, 1])
    grasp_pose_left_shoe[2] -= 0.12
    grasp_pose_right_shoe[2] -= 0.12
    planner.move_to_pose_with_screw([grasp_pose_left_shoe, grasp_pose_right_shoe], move_id=[0, 1])
    planner.close_gripper([0, 1])
    grasp_pose_left_shoe[2] += 0.25
    grasp_pose_right_shoe[2] += 0.25
    planner.move_to_pose_with_screw([grasp_pose_left_shoe, grasp_pose_right_shoe], move_id=[0, 1])

    box_pose = env.shoe_box.pose.p

    # Move lid to target pose above the box
    target_pose_lid = [box_pose[0, 0] + 0.6, box_pose[0, 1], box_pose[0, 2]]
    target_pose_lid = np.array(target_pose_lid + list([0.7071, 0, -0.7071, 0]))
    planner.move_to_pose_with_screw(target_pose_lid, move_id=2)

    # Move shoes to above the box
    target_pose_left_shoe = [box_pose[0, 0] + 0.01, box_pose[0, 1] - 0.11, box_pose[0, 2] + 0.2]
    target_pose_right_shoe = [box_pose[0, 0] + 0.01, box_pose[0, 1] + 0.11, box_pose[0, 2] + 0.2]
    target_pose_left_shoe = np.array(target_pose_left_shoe + list([0, 1, 0, 0]))
    target_pose_right_shoe = np.array(target_pose_right_shoe + list([0, 1, 0, 0]))

    planner.move_to_pose_with_screw([target_pose_left_shoe, target_pose_right_shoe, target_pose_lid], move_id=[0, 1, 2])

    # Lower shoes and lid
    target_pose_lid = [box_pose[0, 0] + 0.3, box_pose[0, 1], box_pose[0, 2] + 0.1]
    target_pose_lid = np.array(target_pose_lid + list([0.7071, 0, -0.7071, 0]))
    target_pose_left_shoe[2] -= 0.06
    target_pose_right_shoe[2] -= 0.06
    planner.move_to_pose_with_screw([target_pose_left_shoe, target_pose_right_shoe, target_pose_lid], move_id=[0, 1, 2])
    planner.open_gripper([0, 1])

    # Move arms away after releasing shoes
    target_pose_left_shoe[2] += 0.2
    target_pose_right_shoe[2] += 0.2
    target_pose_left_shoe[0] -= 0.1
    target_pose_right_shoe[0] -= 0.1
    planner.move_to_pose_with_screw([target_pose_left_shoe, target_pose_right_shoe], move_id=[0, 1])
    planner.close_gripper([2])

    # Move lid to final position
    target_pose_lid = [box_pose[0, 0] + 0.2, box_pose[0, 1], box_pose[0, 2] + 0.3]
    target_pose_lid = np.array(target_pose_lid + list([0.7071, 0, -0.7071, 0]))
    planner.move_to_pose_with_screw(target_pose_lid, move_id=2)

    target_pose_lid[2] += 0.1
    planner.move_to_pose_with_screw(target_pose_lid, move_id=2)

    res = planner.close_gripper([0, 1, 2])
    planner.close()

    return res