#TODO : create a class Trainer to wrap up these functions
import tensorflow as tf
from tensorboard.plugins import projector

import os

from ..archs.constants import PENTE, LR_INI, VOCAB_SIZE





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