#########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

##########################################################################################
# Path Config

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "../../../..")
sys.path.insert(0, "../../../../..")

##########################################################################################
# import

import logging

from utils import create_logger

from mPDTSP_Tester import mPDTSPTester as Tester

##########################################################################################
# parameters

env_params = {
    'customer_size': 10, # 10 for mPDTSP20, 25 for mPDTSP50, 50 for mPDSTP100
    'node_size': 21, # 21 for mPDTSP20, 51 for mPDTSP50, 101 for mPDSTP100
    'problem_gen_params': {
        'scaler': 1.0,
        'capacity':20, # 20 for mPDTSP20, 30 for mPDTSP50, 40 for mPDSTP100
        'demand_min':1,
        'demand_max':10,
    },
    'pomo_size': 10, # >=10 for PDTSP20, >=25 for PDTSP50, >=50 for PDSTP100
}

model_params = {
    'embedding_dim': 256,
    'sqrt_embedding_dim': 256 ** 0.5,
    'encoder_layer_num': 5,
    'qkv_dim': 16,
    'sqrt_qkv_dim': 16 ** 0.5,
    'head_num': 16,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'ms_hidden_dim': 16,
    'ms_layer1_init': (1 / 2) ** 0.5,
    'ms_layer2_init': (1 / 16) ** 0.5,
    'eval_type': 'argmax',
    'one_hot_seed_num': 150,
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': './result/train20',  # directory path of pre-trained model and log files saved.
        'epoch': 2000,  # epoch version of pre-trained model to laod.
    },
    'test_episodes':2000,
    'test_batch_size': 1000,
    'augmentation_enable': True,
    'aug_factor': 50,
    'aug_batch_size': 5,
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']

logger_params = {
    'log_file': {
        'desc': 'POMO_mPDTSP20_test',
        'filename': 'log.txt'
    }
}


##########################################################################################
# main

def main():
    create_logger(**logger_params)
    _print_config()

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    tester.run()


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(f"{g_key} = {globals()[g_key]}") for g_key in globals() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    main()