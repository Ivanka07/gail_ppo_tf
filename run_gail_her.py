#!/usr/bin/python3
import argparse
import gym
import math
import logging
import numpy as np
import datetime
import tensorflow as tf
import sys
import multiprocessing
import os.path as osp
from network_models.discriminator import Discriminator
from collections import defaultdict
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines import her


try:
    from mpi4py import MPI
except ImportError:
    MPI = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env._entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

extra_args = {
        #'demo_file': '/home/ivanna/data_fetch_reach_random_100.npz'
        }

env_id2_obs_shape = {
        'FetchReach-v1': (16,)
        }

batch_size = 100


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory', default='log/train/gail')
    parser.add_argument('--savedir', help='save directory', default='trained_models/gail')
    parser.add_argument('--gamma', default=0.998)
    parser.add_argument('--iteration', default=int(1e4))
    parser.add_argument('--env', default='FetchDrawTriangle-v1')
    parser.add_argument('--expert_file', default='data_fetch_reach_random_100.npz')
    parser.add_argument('--acs', default='actions.csv')
    parser.add_argument('--log_actions', default='log_actions')
    parser.add_argument('--max_episode_length', default=50, type=int)
    parser.add_argument('--render', default=1, choices=[0,1], type=int)
    parser.add_argument('--success_num', default=15, type=int)
    parser.add_argument('--num_timesteps', default=5000, type=int)
    parser.add_argument('--demo_file', default='data_fetch_reach_random_100.npz')
    parser.add_argument('--network', default='mlp')
    parser.add_argument('--seed', default=int(1), type=int)
    parser.add_argument('--num_env', default=int(1), type=int)


    return parser.parse_args()


def train(env, env_type, env_id,seed, num_timesteps, alg_kwargs, old_policy=None):
    print('Training {} on {}:{} with arguments \n{}'.format('her', env_type, env_id, alg_kwargs))
    learn = get_learn_function('her')
    model = learn(
        env=env,
        seed=seed,
        total_timesteps=num_timesteps,
        old_policy=old_policy,
        n_epochs=1,
        **alg_kwargs
    )

    return model


def build_env_sess(args):
    ncpu = multiprocessing.cpu_count()
    print('Number of cpu used', ncpu )
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    seed = args.seed

    env_type, env_id = get_env_type(args.env)
    config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    sess = get_session(config=config)
    env = make_vec_env(env_id, env_type, 1, seed, reward_scale=1.0, flatten_dict_observations=False)

    if env_type == 'mujoco':
        env = VecNormalize(env)

    return env, sess


def get_env_type(env_id):
    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env._entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    for g, e in _game_envs.items():
        if env_id in e:
            env_type = g
            break
    assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())
    return env_type, env_id


def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs

def load_policy(path):
    pass


def main(args):
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        logger.configure()
    else:
        logger.configure(format_strs=[])
        rank = MPI.COMM_WORLD.Get_rank()

    alg_kwargs = {}
    env_type, env_id = get_env_type(args.env)
    env, sess = build_env_sess(args)
    discrim = Discriminator(env, args.env, env_id2_obs_shape)
    
    alg_kwargs = get_learn_function_defaults('her', env_type)
    alg_kwargs.update(extra_args)
    alg_kwargs['network'] = args.network


    expert_data = np.load('expert/' + args.expert_file)
    exp_obs = expert_data['obs']
    exp_act = expert_data['acs']
    print('expert obs shape={}'.format(exp_obs.shape))
    print('expert acts shape={}'.format(exp_obs.shape))
    #training stuff
    saver = tf.train.Saver(max_to_keep=1000)
    writer = tf.summary.FileWriter(args.logdir, sess.graph)
    
    #initial policy training
    her_policy = train(env, env_type, env_id, None, args.num_timesteps, alg_kwargs)
        
    for i in range(args.iteration):
        logging.debug('current iteration={}'.format(i))
        obs_batch = []
        acs_batch = []
        inf_batch = []
        
        obs = env.reset()
        dones = np.zeros((1,))
        state = her_policy.initial_state if hasattr(her_policy, 'initial_state') else None
        policy_steps = 0
        _obs = []
        _acs = []
        _inf = [] #probably, we will need also infos
        while True:
            env.render()
            _obs.append(obs)
            action, _, _, _ = her_policy.step(obs)
            _acs.append(action)
            obs, _, _, info = env.step(action)
            
            policy_steps +=1
            if policy_steps == args.max_episode_length:
                print('len(obs_batch)',len(obs_batch))
                obs_batch.append(_obs)
                acs_batch.append(_acs)
                _obs = []
                _acs = []
                policy_steps=0
                env.reset()

            if len(obs_batch) == 100:
                break
  #  her_policy = train(env, env_type, env_id, None, args.num_timesteps, alg_kwargs, old_policy=her_policy)


    #disriminator training


    #policy training



    
    env.close()
    




if __name__ == '__main__':
    logging.basicConfig(filename='training_gail_her.log', level=logging.DEBUG)
    logging.debug('~~~~~~~~~~~starting training gail with her ~~~~~~~~~~~~~~')
    args = argparser()
    main(args)
