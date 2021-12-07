import tensorflow as tf
from tensorflow.python.keras.backend import sparse_categorical_crossentropy

import word2vec.models.word2vec_models as w
import word2vec.archs.data_loader as dl
import word2vec.archs.constants as c

get_ds = dl.GetDataset('skip_gram', c.PATH)

ds = get_ds.get_ds_ready()

ds = ds.batch(c.BATCH_SIZE)

name = 'w2v_1'

w2v = w.Word2VecSkipGram()

w2v.compile(optimizer='adam',
                 loss=sparse_categorical_crossentropy,
                 metrics=['accuracy'])

with tf.device("/device:CPU:0"):
        w2v.fit(ds,
                epochs = 30,
                #callbacks = [tensorboard],
                verbose = 1)

w2v.save_weights(name + '.h5')