#!/usr/bin/env python3
import argparse
from baselines.common.cmd_util import mujoco_arg_parser
from baselines import bench, logger

def train(env_id, seed, config):
    from baselines.common import set_global_seeds
    from baselines.common.randseedfac import seeder
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo_valmod import ppo_valmod

    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 1
    tf_config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf_config.gpu_options.allow_growth = True
    # use single thread. on such a small problem, multithreading gives you a slowdown
    # this way, we can better use multiple cores for different experiments
    tf.Session(config=tf_config).__enter__()

    set_global_seeds(seed)
    def make_env():
        env = gym.make(env_id)
        env.seed(seeder.get_seed())
        env = bench.Monitor(env, logger.get_dir())
        return env
    env = DummyVecEnv([make_env])
    #env = VecNormalize(env)

    ppo_valmod.learn(env, config)
    #ppo_valmod.learn(policy=policy, env=env, nsteps=2048, nminibatches=32,
    #    lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
    #    ent_coef=0.0,
    #    lr=3e-4,
    #    cliprange=0.2,
    #    total_timesteps=num_timesteps)


def main():
    import shutil
    args = arg_parser_config()
    logger.configure(args.log_directory)
    # copy the config
    shutil.copy2(args.config_name. args.log_directory)
    train(args.env, seed=args.seed, config=args)

def arg_parser_config():
    import argparse
    import imp
    from baselines.ppo_valmod.config import Config
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, help='The config you want to run in PPO.', default=None)
    args = parser.parse_args()
    try:
        config_file = args.config
        config = imp.load_source('config', config_file).config
    except Exception as e:
        config = dict()
        print(str(e) + '\n'
                 +'config import error. File not exist or map_config not specified. Using default settings.')
    return Config(config)

if __name__ == '__main__':
    main()
