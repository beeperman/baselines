import os.path as osp
import datetime

__file_path = osp.abspath(__file__)
__config_name = str.split(__file_path, '/')[-1].replace('.py', '')
__logs_dir = '/'.join(str.split(__file_path, '/')[:-2]) + '/logs'
__log_dir = osp.join(__logs_dir, datetime.datetime.now().strftime(__config_name + "-%Y-%m-%d-%H-%M-%S-%f"))

config = {
    'log_directory': __log_dir,
    'config_file': __file_path,
    'seed': 0,
    'total_timesteps': 1e7,
    'vf_coef': 0.5,
}