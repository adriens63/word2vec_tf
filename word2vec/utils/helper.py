import yaml

import word2vec.utils.trainer as t
import word2vec.models.word2vec_models as w
import word2vec.archs.data_loader as dl
import word2vec.archs.constants as c




def get_model_class(type_model):
    
    return w.Word2VecSkipGram if type_model == 'skip_gram' else w.Word2VecCBOW



def get_lr_scheduler_fn(lr_scheduler):
    
    if lr_scheduler == 'linear_decrease':
        
        return t.linear_decrease
    

def get_config(config):
    