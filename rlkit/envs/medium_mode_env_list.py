import numpy as np

from multiworld.envs.mujoco.sawyer_xyz.env_lists import EASY_MODE_LIST
from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place_6dof import SawyerReachPushPickPlace6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_door_6dof import SawyerDoor6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_stack_6dof import SawyerStack6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_hand_insert import SawyerHandInsert6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_assembly_peg_6dof import SawyerNutAssembly6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_sweep import SawyerSweep6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_window_open_6dof import SawyerWindowOpen6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_hammer_6dof import SawyerHammer6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_window_close_6dof import SawyerWindowClose6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_dial_turn_6dof import SawyerDialTurn6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_lever_pull import SawyerLeverPull6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_drawer_open_6dof import SawyerDrawerOpen6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_button_press_topdown_6dof import SawyerButtonPressTopdown6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_drawer_close_6dof import SawyerDrawerClose6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_box_open_6dof import SawyerBoxOpen6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_box_close_6dof import SawyerBoxClose6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_peg_insertion_side_6dof import SawyerPegInsertionSide6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_sweep import SawyerSweep6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_door_close import SawyerDoorClose6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_shelf_place_6dof import SawyerShelfPlace6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_dial_turn_6dof import SawyerDialTurn6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_lever_pull import SawyerLeverPull6DOFEnv


MEDIUM_MODE_DICT = {
    # Train tasks
    'reach': SawyerReachPushPickPlace6DOFEnv,
    'push': SawyerReachPushPickPlace6DOFEnv,
    'pickplace': SawyerReachPushPickPlace6DOFEnv,
    'door_open': SawyerDoor6DOFEnv,
    'drawer_open': SawyerDrawerOpen6DOFEnv,
    'button_press_top_down': SawyerButtonPressTopdown6DOFEnv,
    'peg_insertion_side': SawyerPegInsertionSide6DOFEnv,
    'window_open': SawyerWindowOpen6DOFEnv,
    'sweep': SawyerSweep6DOFEnv,
    # Test
    'drawer_close': SawyerDrawerClose6DOFEnv,
    'door_close': SawyerDoorClose6DOFEnv,
    'shelf_place': SawyerShelfPlace6DOFEnv,
    'dial_turn': SawyerDialTurn6DOFEnv,
    'lever_pull': SawyerLeverPull6DOFEnv
}


MEDIUM_MODE_ARGS_KWARGS = {
    'reach': {
        "args": [],
        "kwargs": {
            'tasks': [{'goal': np.array([-0.1, 0.8, 0.2]),  'obj_init_pos':np.array([0, 0.6, 0.02]), 'obj_init_angle': 0.3, 'type':'reach'}],
            'multitask': False,
            'obs_type': 'plain',
            'if_render': False,
            'random_init': True,
        }
    },
    'push': {
        "args": [],
        "kwargs": {
            'tasks': [{'goal': np.array([0.1, 0.8, 0.02]),  'obj_init_pos':np.array([0, 0.6, 0.02]), 'obj_init_angle': 0.3, 'type':'push'}],
            'multitask': False,
            'obs_type': 'plain',
            'if_render': False,
            'random_init': True,
        }
    },
    'pickplace': {
        "args": [],
        "kwargs": {
            'tasks': [{'goal': np.array([0.1, 0.8, 0.2]),  'obj_init_pos':np.array([0, 0.6, 0.02]), 'obj_init_angle': 0.3, 'type':'pick_place'}],
            'multitask': False,
            'obs_type': 'plain',
            'if_render': False,
            'random_init': True,   
        }
    },
    'door_open': {
        "args": [],
        "kwargs": {
            'tasks': [{'goal': np.array([-0.2, 0.7, 0.15]),  'obj_init_pos':np.array([0.1, 0.95, 0.1]), 'obj_init_angle': 0.3}],
            'multitask': False,
            'obs_type': 'plain',
            'if_render': False,
            'random_init': True,   
        }
    },
    'drawer_open': {
        "args": [],
        "kwargs": {
            'tasks': [{'goal': np.array([0., 0.55, 0.04]),  'obj_init_pos':np.array([0., 0.9, 0.04]), 'obj_init_angle': 0.3}],
            'multitask': False,
            'obs_type': 'plain',
            'if_render': False,
            'random_init': True,   
        }
    },
    'drawer_close': {
        "args": [],
        "kwargs": {
            'tasks': [{'goal': np.array([0., 0.7, 0.04]),  'obj_init_pos':np.array([0., 0.9, 0.04]), 'obj_init_angle': 0.3}],
            'multitask': False,
            'obs_type': 'plain',
            'if_render': False,
            'random_init': True,   
        }
    },
    'button_press_top_down': {
        "args": [],
        "kwargs": {
            'tasks': [{'goal': np.array([0, 0.88, 0.1]), 'obj_init_pos':np.array([0, 0.8, 0.05])}],
            'multitask': False,
            'obs_type': 'plain',
            'if_render': False,
            'random_init': True,   
        }
    },
    'peg_insertion_side': {
        "args": [],
        "kwargs": {
            'tasks': [{'goal': np.array([-0.3, 0.6, 0.05]), 'obj_init_pos':np.array([0, 0.6, 0.02])}],
            'multitask': False,
            'obs_type': 'plain',
            'if_render': False,
            'random_init': True,   
        }
    },
    'window_open': {
        "args": [],
        "kwargs": {
            'tasks': [{'goal': np.array([0.08, 0.785, 0.15]),  'obj_init_pos':np.array([-0.1, 0.785, 0.15]), 'obj_init_angle': 0.3}],
            'multitask': False,
            'obs_type': 'plain',
            'if_render': False,
            'random_init': True,   
        }
    },
    'sweep': {
        "args": [],
        "kwargs": {
            'tasks': [{'goal': np.array([0., 0.95, -0.3]),  'obj_init_pos':np.array([0., 0.6, 0.02]), 'obj_init_angle': 0.3}],
            'multitask': False,
            'obs_type': 'plain',
            'if_render': False,
            'random_init': True,  
        }
    },
    # Test tasks
    'drawer_close': {
        "args": [],
        "kwargs": {
            'tasks': [{'goal': np.array([0., 0.7, 0.04]),  'obj_init_pos':np.array([0., 0.9, 0.04]), 'obj_init_angle': 0.3}],
            'multitask': False,
            'obs_type': 'plain',
            'if_render': False,
            'random_init': False,  
        }
    },
    'door_close': {
        "args": [],
        "kwargs": {
            'tasks': [{'goal': np.array([0.2, 0.8, 0.15]),  'obj_init_pos':np.array([0.1, 0.95, 0.1]), 'obj_init_angle': 0.3}],
            'multitask': False,
            'obs_type': 'plain',
            'if_render': False,
            'random_init': False,  
        }
    },
    'shelf_place': {
        "args": [],
        "kwargs": {
            'tasks': [{'goal': np.array([0., 0.85, 0.001]),  'obj_init_pos':np.array([0, 0.6, 0.02]), 'obj_init_angle': 0.3}],
            'multitask': False,
            'obs_type': 'plain',
            'if_render': False,
            'random_init': False,  
        }
    },
    'dial_turn': {
        "args": [],
        "kwargs": {
            'tasks': [{'goal': np.array([0., 0.73, 0.08]), 'obj_init_pos':np.array([0, 0.7, 0.05])}],
            'multitask': False,
            'obs_type': 'plain',
            'if_render': False,
            'random_init': False,  
        }
    },
    'lever_pull': {
        "args": [],
        "kwargs": {
            'tasks': [{'goal': np.array([0, 0.75, -0.12]), 'obj_init_pos':np.array([0, 0.7, 0.05])}],
            'multitask': False,
            'obs_type': 'plain',
            'if_render': False,
            'random_init': False,  
        }
    },
    # 'window_close': {
    #     "args": [],
    #     "kwargs": {
    #         'tasks': [{'goal': np.array([-0.08, 0.785, 0.15]),  'obj_init_pos':np.array([0.1, 0.785, 0.15]), 'obj_init_angle': 0.3}],
    #         'multitask': False,
    #         'obs_type': 'plain',
    #         'if_render': False,
    #         'random_init': True,   

    #     }
    # },
}