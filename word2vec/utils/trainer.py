#TODO : create a class Trainer to wrap up these functions
import tensorflow as tf
from tensorboard.plugins import projector

import os

from ..archs.constants import PENTE, LR_INI, VOCAB_SIZE





# ******************** global constants ******************

AUTOTUNE = tf.data.AUTOTUNE





#********************* trainer *******************

class Trainer:
    
    def __init__(
        self,
        device,
        model,
        epochs,
        batch_size,
        buffer_size,
        loss_fn,
        optimizer,
        lr_scheduler,
        train_data_loader,
        train_steps,
        val_data_loader,
        val_steps,
        checkpoint_frequency,
        model_name,
        model_dir,
        log_dir
        ):
        
        self.device = device
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_data_loader = train_data_loader
        self.train_steps = train_steps
        self.val_data_loader = val_data_loader
        self.val_steps = val_steps
        self.checkpoint_frequency = checkpoint_frequency
        self.model_name = model_name
        self.model_dir = model_dir
        self.log_dir = log_dir
        
        self.loss = {'train' : [], 'val' : []}
    
    
    def launch_training(self):
        
        with tf.device(self.device):
            self.model.fit(self.train_ds,
                    epochs = self.epochs,
                    #callbacks = [tensorboard],
                    callbacks = [self.lr_scheduler],
                    verbose = 1)
        
    
    def get_ds(self):
        
        self.train_ds = self.train_data_loader.get_ds_ready()
        self.train_ds = self.train_ds.shuffle(self.buffer_size).batch(self.batch_size, drop_remainder = True).cache().prefetch(buffer_size = AUTOTUNE)
        self.inverse_vocab = self.train_data_loader.get_vocab()
    
    
    def save_weights(self):
        
        self.model.save_weights(self.model_dir + self.model_name + '.h5')


    def log_metadata(self):
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        with open(os.path.join(self.log_dir, 'metadata.tsv'), 'w') as f:
            for word in self.inverse_vocab:
                f.write('{}\n'.format(word))
            for unknown in range(1, VOCAB_SIZE - len(self.inverse_vocab)):
                f.write('unknown #{}\n'.format(unknown))

    
    def log_embeddings(self):
        
        weights = tf.Variable(self.model.layers[0].get_weights()[0][1:])
        # Create a checkpoint from embedding, the filename and key are the
        # name of the tensor.
        checkpoint = tf.train.Checkpoint(embedding = weights)
        checkpoint.save(os.path.join(self.log_dir, 'embedding.ckpt'))


    def config_projector(self):
    
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()

        embedding.tensor_name = 'embedding/.att/value'
        embedding.metadata_path = 'metadata.tsv'
        projector.visualize_embeddings(self.log_dir, config)



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
    checkpoint = tf.train.Checkpoint(embedding = weights)
    checkpoint.save(os.path.join(log_dir, 'embedding.ckpt'))


def config_projector(log_dir):
    
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()

    embedding.tensor_name = 'embedding/.att/value'
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(log_dir, config)