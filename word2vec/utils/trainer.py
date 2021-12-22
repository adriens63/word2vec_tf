#TODO : create a class Trainer to wrap up these functions
import tensorflow as tf
from tensorboard.plugins import projector

import os

from ..archs.constants import PENTE, LR_INI, VOCAB_SIZE
import word2vec.utils.trainer as t
import word2vec.models.word2vec_models as w
import word2vec.archs.data_loader as dl
import word2vec.archs.constants as c





#********************* trainer *******************

class Trainer:
    
    def __init__(
        self,
        device,
        model,
        epochs,
        loss_fn,
        optimizer,
        lr_scheduler,
        train_data_loader,
        train_steps,
        val_data_loader,
        val_steps,
        checkpoint_frequency,
        model_dir,
        model_name
        ):
        
        self.device = device
        self.model = model
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_data_loader = train_data_loader
        self.train_steps = train_steps
        self.val_data_loader = val_data_loader
        self.val_steps = val_steps
        self.checkpoint_frequency = checkpoint_frequency
        self.model_dir = model_dir
        self.model_name = model_name
        
        self.loss = {'train' : [], 'val' : []}
    
    
    def launch_training(self):
        
        model.fit()
    
    










def linear_decrease(epoch, _):
    
    return PENTE * epoch + LR_INI


def log_metadata(log_dir, inverse_vocab):
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(os.path.join(log_dir, 'metadata.tsv'), 'w') as f:
        for word in inverse_vocab:
            f.write('{}\n'.format(word))
        for unknown in range(1, VOCAB_SIZE - len(inverse_vocab)):
            f.write('unknown #{}\n'.format(unknown))
            

def log_embeddings(log_dir, model):
    
    weights = tf.Variable(model.layers[0].get_weights()[0][1:])
    # Create a checkpoint from embedding, the filename and key are the
    # name of the tensor.
    checkpoint = tf.train.Checkpoint(embedding=weights)
    checkpoint.save(os.path.join(log_dir, 'embedding.ckpt'))


def config_projector(log_dir):
    
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()

    embedding.tensor_name = 'embedding/.att/value'
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(log_dir, config)