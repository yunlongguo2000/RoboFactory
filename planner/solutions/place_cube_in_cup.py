import numpy as np
import sapien

from tasks import PlaceCubeInCupEnv
from planner.motionplanner import PandaArmMotionPlanningSolver

def solve(env: PlaceCubeInCupEnv, seed=None, debug=False, vis=False):
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

    cup_grasp_pose = env.cup.pose.p[0].tolist()
    cup_grasp_pose[2] += 0.05
    cup_grasp_pose = np.array(cup_grasp_pose + list([-0.5, 0.5, 0.5, 0.5]))
    cube_grasp_pose = planner.get_grasp_pose_from_obb(env.cube, 1)

    cup_pre_grasp = cup_grasp_pose.copy()
    cube_pre_grasp = cube_grasp_pose.copy()
    cube_pre_grasp[2] += 0.15
    cup_pre_grasp[1] -= 0.1
    planner.move_to_pose_with_screw([cup_pre_grasp, cube_pre_grasp], move_id=[0, 1])

    planner.move_to_pose_with_screw([cup_grasp_pose, cube_grasp_pose], move_id=[0, 1])
    planner.close_gripper(close_id=[0, 1])

    cup_lift = cup_grasp_pose.copy()
    cube_lift = cube_grasp_pose.copy()
    cup_lift[1] += 0.15
    cube_lift[2] += 0.45
    
    planner.move_to_pose_with_screw([cup_lift, cube_lift], move_id=[0, 1])
    
    cup_lift[1] -= 0.2
    planner.move_to_pose_with_screw(cup_lift, move_id=[0])
    
    cube_above_cup = cube_lift.copy()
    cube_above_cup[0] = env.cup.pose.p[0][0]
    cube_above_cup[1] = env.cup.pose.p[0][1]
    cube_above_cup[2] = env.cup.pose.p[0][2] + 0.45
    
    planner.move_to_pose_with_screw(cube_above_cup, move_id=[1])
    
    cube_above_cup[2] -= 0.2
    planner.move_to_pose_with_screw(cube_above_cup, move_id=[1])
    
    res = planner.open_gripper(open_id=[1])

    planner.close()
    return res