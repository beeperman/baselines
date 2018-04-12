import random
import os.path as osp
import datetime
from baselines.ppo_valmod.policies import MlpPolicy

__file_path = osp.abspath(__file__)

MUJOCO_BASE = {
# train parameters
        'env': 'Walker2d-v2',  # openai gym envid
        'seed': random.randint(0, 10000000),
        'log_directory': osp.join(osp.split(__file__)[0], 'logs',
                datetime.datetime.now().strftime("MUJOCO_BASE-%Y-%m-%d-%H-%M-%S-%f")),
        'config_file': __file_path,
# learn parameters
        'policy': MlpPolicy,
        'nsteps': 2048,  # steps in each iteration
        'total_timesteps': 1e6,  # total steps
        'ent_coef': 0.0,  # entropy coefficient
        'lr': 3e-4,  # learning rate
        'vf_coef': 0.5,  # value function coefficient
        'max_grad_norm': 0.5,  # grad clipping
        'gamma': 0.99,
        'lam': 0.95,  # lambda
        'log_interval': 1,  # log every ? iterations
        'nminibatches': 32,  # number of minibatches
        'noptepochs': 10,  # epoches in each iteration
        'cliprange': 0.2,  # see PPO clipping ver.
        'save_interval': 0,  # save model every ? iterations. 0 means no model saving
# model-based parameters
        'img_switch': False,
        'img_trials': 10,
        'img_n_steps': 10,
}

class Config(object):
    def __init__(self, config_dict):
        self.set_attributes(config_dict)
        self.config_dict = config_dict

    def set_attributes(self, config_dict):
        base_dict = config_dict['base_dict'] if 'base_dict' in config_dict else MUJOCO_BASE
        self._set_attributes(base_dict)
        self._set_attributes(config_dict)

    def _set_attributes(self, dict):
        for key in dict:
            self.__setattr__(key, dict[key])
