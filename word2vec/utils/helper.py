import yaml

import os.path as osp

import tensorflow as tf

import word2vec.models.word2vec_models as w





def get_model_class(type_model):
    
    return w.Word2VecSkipGram if type_model == 'skip_gram' else w.Word2VecCBOW



    
def get_lr_scheduler_fn(lr_scheduler, initial_lr = 1e-3, step_size = 10):
    
    if lr_scheduler == 'linear_decrease':
        
        def schedule(epoch):
            return initial_lr - epoch * initial_lr / step_size
        
        return tf.keras.callbacks.LearningRateScheduler(schedule)
    



def log_config(config, model_dir):
    
    config_path = osp.join(model_dir, 'config.yml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    