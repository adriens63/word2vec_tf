import tensorflow as tf
from tensorflow.python.keras.backend import sparse_categorical_crossentropy

import word2vec.utils.trainer as t
import word2vec.models.word2vec_models as w
import word2vec.archs.data_loader as dl
import word2vec.archs.constants as c





AUTOTUNE = tf.data.AUTOTUNE

get_ds = dl.GetDataset('skip_gram', c.PATH)

ds = get_ds.get_ds_ready()

ds = ds.shuffle(c.BUFFER_SIZE).batch(c.BATCH_SIZE, drop_remainder = True).cache().prefetch(buffer_size = AUTOTUNE)

inverse_vocab = get_ds.get_vocab()

name = 'w2v_1'

w2v = w.Word2VecSkipGram()
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.025)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(t.linear_decrease)

w2v.compile(optimizer = optimizer,
                loss = 'sparse_categorical_crossentropy',
                metrics = ['accuracy'])

with tf.device("/device:GPU:0"):
        w2v.fit(ds,
                epochs = 30,
                #callbacks = [tensorboard],
                callbacks = [lr_scheduler]
                verbose = 1)

w2v.save_weights('./weights/' + name + '.h5')