import numpy as np

from metaworld.envs.mujoco.sawyer_xyz.env_lists import EASY_MODE_LIST
from metaworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place import SawyerReachPushPickPlaceEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_door import SawyerDoorEnv
# from metaworld.envs.mujoco.sawyer_xyz.sawyer_stack import SawyerStackEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_hand_insert import SawyerHandInsertEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_assembly_peg import SawyerNutAssemblyEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_sweep import SawyerSweepEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_window_open import SawyerWindowOpenEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_hammer import SawyerHammerEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_window_close import SawyerWindowCloseEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_dial_turn import SawyerDialTurnEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_lever_pull import SawyerLeverPullEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_drawer_open import SawyerDrawerOpenEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_button_press_topdown import SawyerButtonPressTopdownEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_drawer_close import SawyerDrawerCloseEnv
# from metaworld.envs.mujoco.sawyer_xyz.sawyer_box_open import SawyerBoxOpenEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_box_close import SawyerBoxCloseEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_peg_insertion_side import SawyerPegInsertionSideEnv


EASY_MODE_DICT = {
    'reach': SawyerReachPushPickPlaceEnv,
    'push': SawyerReachPushPickPlaceEnv,
    'pickplace': SawyerReachPushPickPlaceEnv,
    'door_open': SawyerDoorEnv,
    'drawer_open': SawyerDrawerOpenEnv,
    'drawer_close': SawyerDrawerCloseEnv,
    'button_press_top_down': SawyerButtonPressTopdownEnv,
    'peg_insertion_side': SawyerPegInsertionSideEnv,
    'window_open': SawyerWindowOpenEnv,
    'window_close': SawyerWindowCloseEnv,
}


EASY_MODE_ARGS_KWARGS = {
    'reach': {
        "args": [],
        "kwargs": {
            # 'tasks': [{'goal': np.array([-0.1, 0.8, 0.2]),  'obj_init_pos':np.array([0, 0.6, 0.02]), 'obj_init_angle': 0.3, 'type':'reach'}],
            'random_init': False,
            'obs_type': 'with_goal',
            # 'if_render': False,
        }
    },
    'push': {
        "args": [],
        "kwargs": {
            # 'tasks': [{'goal': np.array([0.1, 0.8, 0.02]),  'obj_init_pos':np.array([0, 0.6, 0.02]), 'obj_init_angle': 0.3, 'type':'push'}],
            'random_init': False,
            'obs_type': 'with_goal',
            # 'if_render': False,
        }
    },
    'pickplace': {
        "args": [],
        "kwargs": {
            # 'tasks': [{'goal': np.array([0.1, 0.8, 0.2]),  'obj_init_pos':np.array([0, 0.6, 0.02]), 'obj_init_angle': 0.3, 'type':'pick_place'}],
            'random_init': False,
            'obs_type': 'with_goal',
            # 'if_render': False,
        }
    },
    'door_open': {
        "args": [],
        "kwargs": {
            # 'tasks': [{'goal': np.array([-0.2, 0.7, 0.15]),  'obj_init_pos':np.array([0.1, 0.95, 0.1]), 'obj_init_angle': 0.3}],
            'random_init': False,
            'obs_type': 'with_goal',
            # 'if_render': False,
        }
    },
    'drawer_open': {
        "args": [],
        "kwargs": {
            # 'tasks': [{'goal': np.array([0., 0.55, 0.04]),  'obj_init_pos':np.array([0., 0.9, 0.04]), 'obj_init_angle': 0.3}],
            'random_init': False,
            'obs_type': 'with_goal',
            # 'if_render': False,
        }
    },
    'drawer_close': {
        "args": [],
        "kwargs": {
            # 'tasks': [{'goal': np.array([0., 0.7, 0.04]),  'obj_init_pos':np.array([0., 0.9, 0.04]), 'obj_init_angle': 0.3}],
            'random_init': False,
            'obs_type': 'with_goal',
            # 'if_render': False,
        }
    },
    'button_press_top_down': {
        "args": [],
        "kwargs": {
            # 'tasks': [{'goal': np.array([0, 0.88, 0.1]), 'obj_init_pos':np.array([0, 0.8, 0.05])}],
            'random_init': False,
            'obs_type': 'with_goal',
            # 'if_render': False,
        }
    },
    'peg_insertion_side': {
        "args": [],
        "kwargs": {
            # 'tasks': [{'goal': np.array([-0.3, 0.6, 0.05]), 'obj_init_pos':np.array([0, 0.6, 0.02])}],
            'random_init': False,
            'obs_type': 'with_goal',
            # 'if_render': False,
        }
    },
    'window_open': {
        "args": [],
        "kwargs": {
            # 'tasks': [{'goal': np.array([0.08, 0.785, 0.15]),  'obj_init_pos':np.array([-0.1, 0.785, 0.15]), 'obj_init_angle': 0.3}],
            'random_init': False,
            'obs_type': 'with_goal',
            # 'if_render': False,
        }
    },
    'window_close': {
        "args": [],
        "kwargs": {
            # 'tasks': [{'goal': np.array([-0.08, 0.785, 0.15]),  'obj_init_pos':np.array([0.1, 0.785, 0.15]), 'obj_init_angle': 0.3}],
            'random_init': False,
            'obs_type': 'with_goal',
            # 'if_render': False,
        }
    },
}