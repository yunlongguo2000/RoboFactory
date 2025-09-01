import numpy as np
import sapien

from tasks import FourRobotsStackCubeEnv
from planner.motionplanner import PandaArmMotionPlanningSolver

def solve(env: FourRobotsStackCubeEnv, seed=102, debug=False, vis=False):
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
    
    grasp_poseA = planner.get_grasp_pose_from_obb(env.cubeA, 0)
    grasp_poseB = planner.get_grasp_pose_from_obb(env.cubeB, 1)
    grasp_poseA[2] += 0.3
    grasp_poseB[2] += 0.3
    planner.move_to_pose_with_screw([grasp_poseA, grasp_poseB], move_id=[0, 1])
    grasp_poseA[2] -= 0.3
    grasp_poseB[2] -= 0.3
    planner.move_to_pose_with_screw([grasp_poseA, grasp_poseB], move_id=[0, 1])
    planner.close_gripper([0, 1])
    grasp_poseA[2] += 0.1
    grasp_poseB[2] += 0.1
    planner.move_to_pose_with_screw([grasp_poseA, grasp_poseB], move_id=[0, 1])
    
    grasp_poseA[1] -= 0.7
    grasp_poseB[1] -= 0.7
    planner.move_to_pose_with_screw([grasp_poseA, grasp_poseB], move_id=[0, 1])
    planner.open_gripper([0, 1])
    grasp_poseA[2] += 0.3
    grasp_poseB[2] += 0.3
    grasp_poseA[1] += 0.3
    grasp_poseB[1] += 0.3
    planner.move_to_pose_with_screw([grasp_poseA, grasp_poseB], move_id=[0, 1])
    
    
    grasp_poseC = planner.get_grasp_pose_from_obb(env.cubeA, 2)
    grasp_poseD = planner.get_grasp_pose_from_obb(env.cubeB, 3)
    grasp_poseC[2] += 0.3
    grasp_poseD[2] += 0.3
    planner.move_to_pose_with_screw([grasp_poseC, grasp_poseD], move_id=[2, 3])
    grasp_poseC[2] -= 0.3
    grasp_poseD[2] -= 0.3
    planner.move_to_pose_with_screw([grasp_poseC, grasp_poseD], move_id=[2, 3])
    planner.close_gripper([2, 3])
    grasp_poseC[2] += 0.1
    grasp_poseD[2] += 0.1
    planner.move_to_pose_with_screw([grasp_poseC, grasp_poseD], move_id=[2, 3])

    
    # Step 2: Stack the cubes
    target_poseA = planner.get_grasp_pose_for_stack(grasp_poseA, env.goal_region)
    target_poseB = target_poseA.copy()
    planner.move_to_pose_with_screw(target_poseA, move_id=2)
    planner.open_gripper([2])
    target_poseA[2] += 0.3
    planner.move_to_pose_with_screw(target_poseA, move_id=2)
    target_poseA[1] += 0.6
    planner.move_to_pose_with_screw(target_poseA, move_id=2)
    
    target_poseB[2] += 0.3
    planner.move_to_pose_with_screw(target_poseB, move_id=3)
    target_poseB[2] -= 0.2
    planner.move_to_pose_with_screw(target_poseB, move_id=3)
    planner.open_gripper([3])

    res = planner.open_gripper([0, 1, 2, 3])
    planner.close()
    
    return res