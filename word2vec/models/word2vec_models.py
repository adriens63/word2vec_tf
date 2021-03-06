import sys
sys.path.insert(0, "./")

import tensorflow as tf
import tensorflow.keras.layers as tfnn

from ..archs.constants import EMBEDDING_DIM, VOCAB_SIZE





class Word2VecCBOW(tf.keras.Model):
    
    def __init__(self):
        
        super(Word2VecCBOW, self).__init__()
        
        self.E = tfnn.Embedding(
                                input_dim = VOCAB_SIZE, 
                                output_dim = EMBEDDING_DIM
                                )
        self.G = tfnn.GlobalAveragePooling1D()
        self.D = tfnn.Dense(
                            units = VOCAB_SIZE, 
                            activation = 'softmax'
                            )
        
    def call(self, x):
        
        out = self.E(x)
        out = self.G(out)
        out = self.D(out)
    
        return out




class Word2VecSkipGram(tf.keras.Model):
    
    def __init__(self):
        
        super(Word2VecSkipGram, self).__init__()
        
        self.E = tfnn.Embedding(
                                input_dim = VOCAB_SIZE, 
                                output_dim = EMBEDDING_DIM
                                )
        self.D = tfnn.Dense(
                            units = VOCAB_SIZE, 
                            activation = 'softmax'
                            )
        
    def call(self, x):
        
        out = self.E(x)
        out = self.D(out)
    
        return out
    
