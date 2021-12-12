#TODO : create a class Trainer to wrap up these functions
from ..archs.constants import PENTE, LR_INI, VOCAB_SIZE

import os





def linear_decrease(epoch, _):
    
    return PENTE * epoch + LR_INI


def log_metadata(log_dir, inverse_vocab):
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
        for word in inverse_vocab:
            f.write("{}\n".format(word))
        for unknown in range(1, VOCAB_SIZE - len(inverse_vocab)):
            f.write("unknown #{}\n".format(unknown))
            
