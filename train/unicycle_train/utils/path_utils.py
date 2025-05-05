# train/unicycle_train/utils/path_utils.py

import os

def get_project_root():
    current_file = os.path.abspath(__file__)
    return os.path.abspath(os.path.join(current_file, '..', '..', '..', '..'))

def get_data_dir():
    return os.path.join(get_project_root(), 'data')

def get_pre_trained_models_dir():
    return os.path.join(get_project_root(), 'pre_trained_models')

def get_logs_dir():
    return os.path.join(get_data_dir(), 'logs')

def get_models_dir():
    return os.path.join(get_data_dir(), 'models')

def get_untrained_dir():
    return os.path.join(get_models_dir(), 'untrained_model')

def get_pretest_logs_dir():
    return os.path.join(get_logs_dir(), "pre_test_logs")

def get_posttest_logs_dir():
    return os.path.join(get_logs_dir(), "post_test_logs")

def get_train_logs_dir():
    return os.path.join(get_logs_dir(), "train_logs")