import os

import seaborn as sns
import numpy as np
import scipy
import pandas as pd
import json
import argparse

import matplotlib.pyplot as plt

# plt.rcParams['pdf.fonttype'] = 42  # Avoid type 3 fonts.
plt.rcParams['text.usetex'] = True

SAVE_PATH = os.path.expanduser(
    '~/research/write-ups/2018-ICML-sac/figures/')

# SMOOTHING_WINDOW = 25  # This needs to be odd number
#SMOOTHING_WINDOW = 5  # This needs to be odd number
#SMOOTHING_WINDOW = 251  # This needs to be odd number
FIG_SCALE = 0.6  # Make figure smaller to increase font size.


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', type=str, default='benchmark-half-cheetah')

    parser.add_argument('--show', dest='show', action='store_true')
    parser.add_argument('--no-show', dest='show', action='store_false')
    parser.add_argument('--save', dest='save', action='store_true')
    parser.add_argument('--no-save', dest='save', action='store_false')
    parser.add_argument('--legend', dest='legend', action='store_true')
    parser.add_argument('--no-legend', dest='legend', action='store_false')
    parser.set_defaults(show=True, save=False, legend=True)

    args = parser.parse_args()
    return args


COMMON_SETTINGS = {
    'threshold': None,
    'legend_loc': None,
    'title_show': True,
    'ylims': None,
    'figsize': [6.4 * FIG_SCALE, 4.8 * FIG_SCALE],
    'xlabel': 'million samples',
    'linewidth': 1,
    # 'err_style': 'unit_traces',
    'err_style': 'ci_band',
    'smoothing_window': 25
}


DATA_ROOT = os.path.expanduser('/mnt/c/Users/Aurick/Desktop/src/rllab/data/s3/')
PATHS = {
    # Swimmer (rllab)
    'sac-hard-swimmer-rllab': 'sac-camera-ready/sac-4step-hardupdates-final-runs/swimmer-rllab/default/hardupdates4stepfinalruns/swimmer-rllab',
    'sac-soft-swimmer-rllab': 'sac-camera-ready/sac-1step-softtau-final-runs/swimmer-rllab/default/softupdates1stepfinalruns/swimmer-rllab',
    'td3-swimmer-rllab': 'sac-camera-ready/td3-final-runs/swimmer-rllab/td3-finals',
    'ppo-swimmer-rllab': 'ppo-baselines-final-runs/swimmer-rllab/longer-runs/',
    'pcl-swimmer-rllab': 'trust-pcl-baselines/trust-pcl-swimmer-rerun-final/',
    'sql-swimmer-rllab': 'sql-min2qf/final-runs/swimmer-rllab/sqlminqffinal/sql-min2qf/',
    'ddpg-swimmer-rllab': 'sac-camera-ready/ddpg/swimmer/ddpg-final-runs/',

    # Hopper-v1
    'ecsac-hopper': 'ecsac-ablations/gym/hopper/default/20181114-ecsac-ablations',
    'sac-hard-hopper': 'sac-camera-ready/sac-4step-hardupdates-final-runs/hopper/default/hardupdates4stepfinalruns/hopper',
    'sac-soft-hopper': 'sac-camera-ready/sac-1step-softtau-final-runs/hopper/default/softupdates1stepfinalruns/hopper',
    'td3-hopper': 'sac-camera-ready/td3-final-runs/hopper/td3-finals',
    'ppo-hopper': 'ppo-baselines-final-runs/hopper/longer-runs/',
    'pcl-hopper': 'trust-pcl-baselines/trust-pcl-hopper-rerun-final/',
    'sql-hopper': 'sql-min2qf/final-runs/hopper/sqlminqffinal/sql-min2qf/',
    'ddpg-hopper': 'sac-camera-ready/ddpg/hopper/ddpg-final-runs/',
    'ddpg-vanilla-hopper': 'sac-camera-ready/vanilla-ddpg/ddpg256-hopper-final/sac-camera-ready/vanilla_ddpg/',
    # HalfCheetah-v1
    'ecsac-half-cheetah': 'ecsac-ablations/gym/half-cheetah/default/20181114-ecsac-albations',
    'sac-hard-half-cheetah': 'sac-camera-ready/sac-4step-hardupdates-final-runs/half-cheetah/default/hardupdates4stepfinalruns/half-cheetah',
    'sac-soft-half-cheetah': 'sac-camera-ready/sac-1step-softtau-final-runs/half-cheetah/default/softupdates1stepfinalruns/half-cheetah',
    'td3-half-cheetah': 'sac-camera-ready/td3-final-runs/half-cheetah/td3-finals',
    'ppo-half-cheetah': 'ppo-baselines-final-runs/half-cheetah/longer-runs/',
    'pcl-half-cheetah': 'trust-pcl-baselines/trust-pcl-cheetah-rerun-final/',
    'sql-half-cheetah': 'sql-min2qf/final-runs/half-cheetah/sqlminqffinal/sql-min2qf/',
    'ddpg-half-cheetah': 'sac-camera-ready/ddpg/half-cheetah/ddpg-final-runs/',
    'ddpg-vanilla-half-cheetah': 'sac-camera-ready/vanilla-ddpg/ddpg256-cheetah-final/sac-camera-ready/vanilla_ddpg/',
    # Walker2d-v1
    'ecsac-walker': 'ecsac-ablations/gym/walker/default/20181114-ecsac-ablations',
    'sac-hard-walker': 'sac-camera-ready/sac-4step-hardupdates-final-runs/walker/default/hardupdates4stepfinalruns/walker',
    'sac-soft-walker': 'sac-camera-ready/sac-1step-softtau-final-runs/walker/default/softupdates1stepfinalruns/walker',
    'td3-walker': 'sac-camera-ready/td3-final-runs/walker/td3-finals',
    'ppo-walker': 'ppo-baselines-final-runs/walker/longer-runs/',
    'pcl-walker': 'trust-pcl-baselines/trust-pcl-walker-rerun-final/',
    'sql-walker': 'sql-min2qf/final-runs/walker/sqlminqffinal/sql-min2qf/',
    'ddpg-walker': 'sac-camera-ready/ddpg/walker/ddpg-final-runs/',
    'ddpg-vanilla-walker': 'sac-camera-ready/vanilla-ddpg/ddpg256-walker-final/sac-camera-ready/vanilla_ddpg/',
    # Ant-v1
    'ecsac-ant': 'ecsac-ablations/gym/ant/default/20181114-ecsac-ablations',
    'sac-hard-ant': 'sac-camera-ready/sac-4step-hardupdates-final-runs/ant/default/hardupdates4stepfinalruns/ant',
    'sac-soft-ant': 'sac-camera-ready/sac-1step-softtau-final-runs/ant/default/softupdates1stepfinalruns/ant',
    'td3-ant': 'sac-camera-ready/td3-final-runs/ant/td3-finals',
    'ppo-ant': 'ppo-baselines-final-runs/ant/longer-runs/',
    'pcl-ant': 'trust-pcl-baselines/trust-pcl-ant-rerun-final/',
    'sql-ant': 'sql-min2qf/final-runs/ant/sqlminqffinal/sql-min2qf/',
    #'ddpg-ant': 'sac-camera-ready/ddpg/ant/ddpg-final-runs/',
    'ddpg-ant': 'sac-camera-ready/ddpg/ant/ddpg-ant-longer',
    'ddpg-vanilla-ant': 'sac-camera-ready/vanilla-ddpg/ddpg256-ant-final/sac-camera-ready/vanilla_ddpg/',
    # Humanoid (rllab)
    'ecsac-humanoid-rllab': 'ecsac-ablations/rllab/humanoid/default/20181115-ecsac-ablations',
    'sac-hard-humanoid-rllab': 'sac-camera-ready/final-runs/humanoid-rllab/default/sac-final-runs/humanoid-rllab',
    'sac-soft-humanoid-rllab': 'sac-camera-ready/sac-1step-softtau-final-runs/humanoid-rllab/default/softupdates1stepfinalruns/humanoid-rllab',
    'td3-humanoid-rllab': 'sac-camera-ready/td3-final-runs/humanoid-rllab/td3-finals',
    'ppo-humanoid-rllab': 'ppo-baselines-final-runs/humanoid-rllab/longer-runs/',
    'pcl-humanoid-rllab': 'trust-pcl-baselines/trust-pcl-humanoid-rerun-final/',
    'sql-humanoid-rllab': 'sql-min2qf/final-runs/humanoid-rllab/sqlminqffinal/sql-min2qf/',
    'ddpg-humanoid-rllab': 'sac-camera-ready/ddpg/humanoid-rllab/ddpg-final-runs/',
    'ddpg-vanilla-humanoid-rllab': 'sac-camera-ready/vanilla-ddpg/ddpg256-humanoid-rllab-final/sac-camera-ready/vanilla_ddpg/',
    # Humanoid-v1
    'ecsac-humanoid-gym': 'ecsac-ablations/gym/humanoid/default/20181114-ecsac-ablations',
    'sac-hard-humanoid-gym': 'sac-camera-ready/final-runs/humanoid-gym/default/sac-final-runs/humanoid-gym',
    'sac-soft-humanoid-gym': 'sac-camera-ready/sac-1step-softtau-final-runs/humanoid-gym/default/softupdates1stepfinalruns/humanoid-gym',
    'td3-humanoid-gym': 'sac-camera-ready/td3-final-runs/humanoid-gym/td3-finals',
    'ppo-humanoid-gym': 'ppo-baselines-final-runs/humanoid-gym/ppo-humanoid-rerun-actual',
    'pcl-humanoid-gym': 'trust-pcl-baselines/trust-pcl-humanoid-rerun-final/',
    'sql-humanoid-gym': 'sql-min2qf/final-runs/humanoid-gym/sqlminqffinal/sql-min2qf/',
    'ddpg-humanoid-gym': 'sac-camera-ready/ddpg/humanoid-gym/ddpg-final-runs/',
    'ddpg-vanilla-humanoid-gym': 'sac-camera-ready/vanilla-ddpg/ddpg256-humanoid-gym-final/sac-camera-ready/vanilla_ddpg/',
    # reward scale sweeps
    'sac-reward-scale-half-cheetah': 'sac-camera-ready/scale-reward-sweeps/half-cheetah/default/scale-reward-sensitivity/half-cheetah',
    'sac-reward-scale-ant': 'sac-camera-ready/scale-reward-sweeps/ant/default/scale-reward-sensitivity/ant',
    'sac-soft-reward-scale-ant': 'sac-camera-ready/softtau-scalerewardfinalsweeps/ant/default/scale-reward-sweeps-final/ant',

    # target sweeps
    'sac-soft-target-ant': 'sac-camera-ready/softtau-finalsweeps/ant/default/softtausweepsant/ant',
    # 'sac-hard-target-ant': 'sac-camera-ready/targetupdateinterval-finalsweeps/ant/default/targetupdateintervalsweepsant/ant',
    'sac-hard-target-ant': 'sac-camera-ready/targetupdateinterval2-finalsweeps/ant/default/targetupdatesweeps/ant',

    # ablations
    'ablation-ant-determ-actor-critic': 'sac-camera-ready/ablations/ant/default/determ-target-update-fixed-variance/with-vf-ablations/ant',
    'ablation-ant-determ-critic': 'sac-camera-ready/ablations/ant/default/determ-target-update-learn-variance/with-vf-ablations/ant',
    'ablation-ant-determ-actor': 'sac-camera-ready/ablations/ant/default/stochastic-target-update-fixed-variance/with-vf-ablations/ant',
    'ablation-ant': 'sac-camera-ready/ablations/ant/default/stochastic-target-update-learn-variance/with-vf-ablations/ant',
    'ablation-ant-determ-actor-critic-no-vf': 'sac-camera-ready/ablations/ant/default/determ-target-update-fixed-variance/no-vf-ablations/ant',
    'ablation-ant-determ-critic-no-vf': 'sac-camera-ready/ablations/ant/default/determ-target-update-learn-variance/no-vf-ablations/ant',
    'ablation-ant-determ-actor-no-vf': 'sac-camera-ready/ablations/ant/default/stochastic-target-update-fixed-variance/no-vf-ablations/ant',
    'ablation-ant-no-vf': 'sac-camera-ready/ablations/ant/default/stochastic-target-update-learn-variance/no-vf-ablations/ant',
}

ALL_COLORS = (
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf'
)

COLORS = {
    'sac-hard': ALL_COLORS[0],
    'sac-soft': ALL_COLORS[1],
    'ddpg-vanilla': ALL_COLORS[2],
    'td3': ALL_COLORS[4],
    'ddpg': ALL_COLORS[3],
    'ppo': ALL_COLORS[5],
    'sql': ALL_COLORS[6],
    'pcl': ALL_COLORS[7],
    'ecsac': ALL_COLORS[8],
}

NAMES = {
    'sac-hard': 'SAC (hard target update)',
    'sac-soft': 'SAC (fixed temperature)',
    'ecsac': 'ECSAC',
    # 'sac-soft': 'SAC (soft target update)',
    'ppo': 'PPO',
    'ddpg-vanilla': 'DDPG',
    'td3': 'TD3 (concurrent)',
    'ddpg': 'SAC (hard target update, deterministic)',
    'pcl': 'Trust-PCL',
}

DATA_FORMATS = {
    'sac-hard': 'sac',
    'sac-soft': 'sac',
    'ecsac': 'ecsac',
    'td3': 'td3',
    'ppo': 'ppo',
    'pcl': 'pcl',
    'ddpg': 'ddpg',
    'ddpg-vanilla': 'ddpg-vanilla',
}

ALGORITHMS = ('sac-hard', 'sac-soft', 'ddpg-vanilla', 'ppo', 'td3', 'ddpg', 'pcl')
# ALGORITHMS = ('sac-hard', 'sac-soft', 'ddpg-vanilla', 'ppo', 'td3', 'ddpg')
ALGORITHMS = ('sac-soft', 'ddpg-vanilla', 'ppo', 'td3', 'ecsac')  # Primary baselines
# ALGORITHMS = ('sac-soft', 'sac-hard', 'ddpg', 'pcl')  # Extra baselines
# ALGORITHMS = ('td3',)

PLOT_CONFIG = {}
PLOT_CONFIG['benchmark-swimmer-rllab'] = {
    'xmax': 1,
    'title': 'Swimmer (rllab)',
    'output': 'benchmarks/swimmer-rllab',
    'experiments': ({
        'name': NAMES[algorithm],
        'path': PATHS[algorithm + '-swimmer-rllab'],
        'color': COLORS[algorithm],
        'data_format': DATA_FORMATS[algorithm]
    } for algorithm in ALGORITHMS),
}
PLOT_CONFIG['benchmark-hopper'] = {
    'xmax': 1,
    'ylim': (-250, 4000),
    'title': 'Hopper-v2',
    'output': 'benchmarks/hopper',
    'experiments': ({
        'name': NAMES[algorithm],
        'path': PATHS[algorithm + '-hopper'],
        'color': COLORS[algorithm],
        'condition': {'use_vf': False, 'use_min_q': True},
        'data_format': DATA_FORMATS[algorithm]
    } for algorithm in ALGORITHMS),
}
PLOT_CONFIG['benchmark-half-cheetah'] = {
    'xmax': 3,
    'ylim': (-1000, 17000),
    'title': 'HalfCheetah-v2',
    'output': 'benchmarks/half-cheetah',
    'experiments': ({
        'name': NAMES[algorithm],
        'path': PATHS[algorithm + '-half-cheetah'],
        'color': COLORS[algorithm],
        'condition': {'use_vf': False, 'use_min_q': True},
        'data_format': DATA_FORMATS[algorithm]
    } for algorithm in ALGORITHMS),
}
PLOT_CONFIG['benchmark-walker'] = {
    'xmax': 3,
    'ylim': (0, 7000),
    'title': 'Walker2d-v2',
    'output': 'benchmarks/walker',
    'experiments': ({
        'name': NAMES[algorithm],
        'path': PATHS[algorithm + '-walker'],
        'color': COLORS[algorithm],
        'condition': {'use_vf': False, 'use_min_q': True},
        'data_format': DATA_FORMATS[algorithm]
    } for algorithm in ALGORITHMS),
}
PLOT_CONFIG['benchmark-ant'] = {
    'xmax': 3,
    'ylim': (-1000, 7000),
    'title': 'Ant-v2',
    'output': 'benchmarks/ant',
    'experiments': ({
        'name': NAMES[algorithm],
        'path': PATHS[algorithm + '-ant'],
        'color': COLORS[algorithm],
        'condition': {'use_vf': False, 'use_min_q': True},
        'data_format': DATA_FORMATS[algorithm]
    } for algorithm in ALGORITHMS),
}
PLOT_CONFIG['benchmark-humanoid-rllab'] = {
    'xmax': 10,
    'ylim': (-250, 7000),
    'title': 'Humanoid (rllab)',
    'output': 'benchmarks/humanoid-rllab',
    # 'smoothing_window': 251,
    'experiments': ({
        'name': NAMES[algorithm],
        'path': PATHS[algorithm + '-humanoid-rllab'],
        'color': COLORS[algorithm],
        'condition': {'use_vf': False, 'use_min_q': True},
        'data_format': DATA_FORMATS[algorithm]
    } for algorithm in ALGORITHMS),
}
PLOT_CONFIG['benchmark-humanoid-gym'] = {
    'xmax': 10,
    'ylim': (-250, 9000),
    'title': 'Humanoid-v2',
    'output': 'benchmarks/humanoid-gym',
    # 'smoothing_window': 251,
    'experiments': ({
        'name': NAMES[algorithm],
        'path': PATHS[algorithm + '-humanoid-gym'],
        'color': COLORS[algorithm],
        'condition': {'use_vf': False, 'use_min_q': True},
        'data_format': DATA_FORMATS[algorithm]
    } for algorithm in ALGORITHMS),
}
NAMES_SEEDS = {'sac-hard': 'stochastic policy', 'ddpg': 'deterministic policy'}
PLOT_CONFIG['seeds-humanoid-rllab'] = {
    'xmax': 10,
    'ylim': (-250, 7000),
    'title': 'Humanoid (rllab)',
    'output': 'benchmarks/seeds-humanoid-rllab',
    'err_style': 'unit_traces',
    'smoothing_window': 251,
    'experiments': ({
        'name': NAMES_SEEDS[algorithm],
        'path': PATHS[algorithm + '-humanoid-rllab'],
        'color': COLORS[algorithm],
        'data_format': DATA_FORMATS[algorithm]
    } for algorithm in ('sac-hard', 'ddpg')),
}
PLOT_CONFIG['ablation-reward-scale-half-cheetah'] = {
    'xmax': 3,
    'ylim': (-1000, 17000),
    'title': 'HalfCheetah-v1',
    'output': 'ablations/reward-scale-half-cheetah',
    'experiments': ({
        'name': str(scale),
        'path': PATHS['sac-reward-scale-half-cheetah'],
        'condition': {'scale_reward': scale},
        'color': ALL_COLORS[index],
        'data_format': 'sac',
    } for index, scale in enumerate((0.3, 1, 3, 10))),
}
PLOT_CONFIG['ablation-reward-scale-ant'] = {
    'xmax': 3,
    'ylim': (-1000, 7000),
    'title': 'Ant-v1',
    'output': 'ablations/reward-scale-ant',
    'experiments': ({
        'name': str(scale),
        'path': PATHS['sac-reward-scale-ant'],
        'condition': {'scale_reward': scale},
        'color': ALL_COLORS[index],
        'data_format': 'sac',
    } for index, scale in enumerate((0.3, 1, 3, 10))),
}
PLOT_CONFIG['sweep-reward-scale-ant'] = {
    'xmax': 3,
    'ylim': (-1000, 7000),
    'title': 'Ant-v1',
    'output': 'sweeps/reward-scale-ant',
    'experiments': ({
        'name': str(scale),
        'path': PATHS['sac-soft-reward-scale-ant'],
        'condition': {'scale_reward': scale},
        'color': ALL_COLORS[index],
        'data_format': 'sac',
    } for index, scale in enumerate((1, 3, 10, 30, 100))),
}
PLOT_CONFIG['sweep-soft-target-ant'] = {
    'xmax': 3,
    # 'ylim': (-1000, 7000),
    'title': 'Ant-v1',
    'output': 'sweeps/soft-target-ant',
    'experiments': ({
        'name': str(tau),
        'path': PATHS['sac-soft-target-ant'],
        'condition': {'tau': tau},
        'color': ALL_COLORS[index],
        'data_format': 'sac',
    } for index, tau in enumerate((0.0001, 0.001, 0.01, 0.1))),
}
PLOT_CONFIG['sweep-hard-target-ant'] = {
    'xmax': 3,
    # 'ylim': (-1000, 7000),
    'title': 'Ant-v1',
    'output': 'sweeps/hard-target-ant',
    'experiments': ({
        'name': str(interval),
        'path': PATHS['sac-hard-target-ant'],
        'condition': {'target_update_interval': interval},
        'color': ALL_COLORS[index],
        'data_format': 'sac',
    } for index, interval in enumerate((125, 250, 500, 1000, 2000))),
    # } for index, interval in enumerate((1, 10, 100, 1000, 10000))),
}

SAC_DDPG_ANT_VARIANTS = (
    'ablation-ant-determ-actor-critic',
    'ablation-ant-determ-critic',
    'ablation-ant-determ-actor',
    'ablation-ant',
)
PLOT_CONFIG['ablation-sac-ddpg-ant'] = {
    'xmax': 3,
    'ylim': (-1000, 7000),
    'title': 'Ant-v1',
    'output': 'ablations/sac-ddpg-ant',
    'experiments': ({
        'name': str(variant),
        'path': PATHS[variant],
        'color': ALL_COLORS[index],
        'data_format': 'sac',
    } for index, variant in enumerate(SAC_DDPG_ANT_VARIANTS)),
}
SAC_DDPG_ANT_NO_VF_VARIANTS = (
    'ablation-ant-determ-actor-critic-no-vf',
    'ablation-ant-determ-critic-no-vf',
    'ablation-ant-determ-actor-no-vf',
    'ablation-ant-no-vf',
)
PLOT_CONFIG['ablation-sac-ddpg-ant-no-vf'] = {
    'xmax': 3,
    'ylim': (-1000, 7000),
    'title': 'Ant-v1',
    'output': 'ablations/sac-ddpg-ant-no-vf',
    'experiments': ({
        'name': str(variant),
        'path': PATHS[variant],
        'color': ALL_COLORS[index],
        'data_format': 'sac',
    } for index, variant in enumerate(SAC_DDPG_ANT_NO_VF_VARIANTS)),
}

EVALUATION_LABELS = ('deterministic evaluation', 'stochastic evaluation')
PLOT_CONFIG['ablation-evaluation-ant'] = {
    'xmax': 3,
    'ylim': (-1000, 7000),
    'title': 'Ant-v1',
    'output': 'ablations/evaluation-ant',
    'experiments': ({
        'name': EVALUATION_LABELS[index],
        'path': PATHS['sac-hard-ant'],
        'color': ALL_COLORS[index],
        'data_format': DATA_FORMATS['sac-hard'],
        'value_field': variant
    } for index, variant in enumerate(('return-average', 'last-path-return'))),
}

PLOT_CONFIG['ablation-evaluation-half-cheetah'] = {
    'xmax': 3,
    'ylim': (-250, 17000),
    'title': 'HalfCheetah-v1',
    'output': 'ablations/evaluation-half-cheetah',
    'experiments': ({
        'name': EVALUATION_LABELS[index],
        'path': PATHS['sac-hard-half-cheetah'],
        'color': ALL_COLORS[index],
        'data_format': DATA_FORMATS['sac-hard'],
        'value_field': variant
    } for index, variant in enumerate(('return-average', 'last-path-return'))),
}


def includes_variant(variant, key, value, default=False):
    if variant.get(key, None) == value:
        return True

    for k, v in variant.items():
        if isinstance(v, dict):
            if includes_variant(v, key, value):
                return True

    return False


def includes_variants(variant, condition, default=False):
    for key, value in condition.items():
        if not includes_variant(variant, key, value, default):
            return False
    return True


def load_rllab(experiment_path, experiment_name, data_format, condition, smoothing_window, value_field=None, sampling_interval=None, 
        default=True):
    trial_paths = [out[0] for out in os.walk(experiment_path)]
    print([out for out in os.walk(experiment_path)])
    trial_paths = trial_paths[1:]  # First is the current directory.
    print(trial_paths)

    trials_list = []

    for trial_path in trial_paths:
        print('hi')
        with open(os.path.join(trial_path, 'variant.json')) as file:
            variant = json.load(file)

        print(experiment_name)
        if not includes_variants(variant, condition, default) and experiment_name == 'ECSAC':
            print('skipped')
            continue

        trial = pd.read_csv(os.path.join(trial_path, 'progress.csv'))
        trial['samples'] = get_num_samples(trial, data_format)
        trial['value'] = get_mean_return(trial, data_format, value_field)

        # Resample every 1000 time steps.
        num_samples = trial['samples'].values[-1]
        interp_samples = range(0, num_samples+1000, 1000)

        interp_fn = scipy.interpolate.interp1d(
            x=trial['samples'].values,
            y=trial['value'].values,
            fill_value='extrapolate')

        interp_value = interp_fn(interp_samples)

        trial = pd.DataFrame({'samples': interp_samples, 'value': interp_value})

        trial['smooth-value'] = np.convolve(
            np.pad(trial['value'].values, int((smoothing_window - 1) / 2), 'edge'),
            np.ones((smoothing_window,)) / smoothing_window,
            mode='valid'
        )

        trial['kilo-samples'] = trial['samples'].values / 1000
        trial['mega-samples'] = trial['kilo-samples'].values / 1000

        trial['experiment-name'] = experiment_name
        trial['trial-id'] = get_trial_id(variant, data_format)

        if sampling_interval:
             # Reduce file size by downampling.
             trial = trial.iloc[::sampling_interval, :]

        trials_list.append(trial)

    trials_list = trials_list[:5]

    try:
        trials = pd.concat(trials_list)
    except:
        import ipdb; ipdb.set_trace()
        pass

    return trials


def get_trial_id(variant, data_format):
    return {
        'sac': lambda: variant['run_params']['seed'],
        'ecsac': lambda: variant['run_params']['seed'],
        'td3': lambda: variant['seed'],
        'ppo': lambda: variant['seed'],
        'pcl': lambda: variant['seed'],
        'sql': lambda: variant['seed'],
        'ddpg': lambda: variant['seed'],
        'ddpg-vanilla': lambda: np.random.randint(0, 99999999),
    }[data_format]()


def get_num_samples(trial, data_format):
    return {
        'sac': lambda: trial['total-samples'],
        'ecsac': lambda: trial['total-samples'],
        'td3': lambda: trial['TimestepsSoFar'],
        'ppo': lambda: trial['TimestepsSoFar'],
        'pcl': lambda: trial['EnvSteps'].values*10,
        'sql': lambda: trial['total-samples'],
        'ddpg': lambda: trial['total-samples'],
        'ddpg-vanilla': lambda: trial['Epoch']*1000,
    }[data_format]()


def get_mean_return(trial, data_format, value_field=None):
    if value_field == None:
        return {
            'sac': lambda: trial['return-average'],
            'ecsac': lambda: trial['return-average'],
            'td3': lambda: trial['EvalAvgReturn'],
            'ppo': lambda: trial['return'],
            'pcl': lambda: trial['AvgEvalReturn'],
            'sql': lambda: trial['return-average'],
            'ddpg': lambda: trial['return-average'],
            'ddpg-vanilla': lambda: trial['AverageReturn'],
        }[data_format]()
    else:
        return trial[value_field]


def initialize_plot():
    sns.set(style='darkgrid')
    fig, ax = plt.subplots(figsize=COMMON_SETTINGS['figsize'])
    return fig, ax


def finalize_plot(ax, title, xmax=None, ylim=None, show_legend=True):
    plt.xlabel('million steps')
    plt.ylabel('average return')

    if show_legend:
        leg = ax.legend()
        leg.set_title('')
    else:
        ax.legend().set_visible(False)

    if True:  # settings['title_show']:
        plt.title(title)

    if xmax:
        plt.xlim(0, xmax)

    if ylim:
        plt.ylim(*ylim)


    # #if settings['ylim'] is not None:
    # #    plt.ylim(*settings['ylim'])

    # if settings['legend_loc'] is not None:
    #     plt.legend(loc=settings['legend_loc'])

    plt.subplots_adjust(bottom=0.14, left=0.16)  # Make labels visible.

    # Fixed x-label position.
    xpos, _ = ax.xaxis.label.get_position()
    ax.xaxis.set_label_coords(xpos, -0.1)

    # Fixed y-label position.
    _, ypos = ax.yaxis.label.get_position()
    ax.yaxis.set_label_coords(-0.15, ypos)


def save_plot(fig, output):
    full_path = os.path.join(SAVE_PATH, output + '.pdf')
    fig.savefig(full_path, format='pdf')


def main():
    args = parse_args()
    fig, ax = initialize_plot()

    for experiment in PLOT_CONFIG[args.plot]['experiments']:
        print('Plotting {exp}'.format(exp=experiment['name']))

        config = COMMON_SETTINGS.copy()
        config.update(PLOT_CONFIG[args.plot])

        sampling_interval = int(config.get('xmax', None))

        experiment_data = load_rllab(
            experiment_path=os.path.join(DATA_ROOT, experiment['path']),
            experiment_name=experiment['name'],
            data_format=experiment['data_format'],
            condition=experiment.get('condition', {}),
            smoothing_window=config['smoothing_window'],
            value_field=experiment.get('value_field', None),
            sampling_interval=sampling_interval)

        sns.tsplot(
            data=experiment_data,
            # time='samples',
            time='mega-samples',
            unit='trial-id',
            condition='experiment-name',
            # value='value',
            value='smooth-value',
            ax=ax,
            color=experiment['color'],
            lw=config['linewidth'],
            ci=100,
            err_style=config['err_style'])

    finalize_plot(
        ax=ax,
        title=PLOT_CONFIG[args.plot]['title'],
        xmax=PLOT_CONFIG[args.plot].get('xmax', None),
        ylim=PLOT_CONFIG[args.plot].get('ylim', None),
        show_legend=args.legend)
    if args.show:
        plt.show()

    if args.save:
        save_plot(fig, PLOT_CONFIG[args.plot]['output'])


if __name__ == '__main__':
    main()
