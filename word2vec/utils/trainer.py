import tensorflow as tf
from tensorboard.plugins import projector

import pickle

import os

from ..archs.constants import VOCAB_SIZE, MEMORY_GPU





# ******************** global constants ******************

AUTOTUNE = tf.data.AUTOTUNE





# ************************* Checkpoint *******************

class Checkpoint(tf.keras.callbacks.ModelCheckpoint):

    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)\

        # Also save the optimizer state
        filepath = self._get_file_path(epoch, logs)
        filepath = filepath.rsplit(".", 1)[0]
        filepath += ".pkl"

        with open(filepath, 'wb') as fp:
            pickle.dump(
            {
            'opt': self.model.optimizer.get_config(),
            'epoch': epoch+1
            # Add additional keys if you need to store more values
            }, fp, protocol = pickle.HIGHEST_PROTOCOL)
        print('Epoch %05d: saving optimizer to %s' % (epoch + 1, filepath))
        print()





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
        weights_path,
        ) -> None:
        
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
        
        self.mod_dir = weights_path + model_name + '/'
        self.log_dir = weights_path + model_name + '/log_dir/'
        self.ckp_dir = weights_path + model_name + '/ckp_dir/'
        self.compiled = False
        #TODO add val_steps, train_steps, into code

        #self.checkpoint = Checkpoint(model, self.ckp_dir + 'model-{epoch:02d}-{loss:.2f}.hdf5', monitor = 'val_loss', verbose = 1)
        #self.checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = self.ckp_dir + 'model-{epoch:02d}-{loss:.2f}', monitor = 'loss', save_freq = 'epoch', period = 1, verbose = 1)
        # self.tnsorboard = tf.keras.callbacks.TensorBoard(
        #                                                 log_dir = 'logs', histogram_freq = 0, write_graph = True,
        #                                                 write_images = False, write_steps_per_second = False, update_freq = 'epoch')                   
    
    def launch_training(self) -> None:
        
        if not hasattr(self, 'train_ds'):
            
            self.get_ds()
            
        if not self.compiled:
            
            self.compile()
        
        print('.... Start training')
        print(f'device : {self.device}')
        
        with tf.device(self.device):
            self.model.fit(self.train_ds,
                            epochs = self.epochs,
                            callbacks = [self.lr_scheduler, 
                                         #self.checkpoint, 
                                         #self.tnsorboard
                                         ],
                            validation_data = self.val_ds,
                            verbose = 1,
                            shuffle = False # the data is already shuffled when loaded
                            )
        print('done;')
        print()
        
    
    def compile(self) -> None:
        
        self.model.compile(optimizer = self.optimizer,
                            loss = self.loss_fn,
                            metrics = ['accuracy'],
                            )
        
        self.compiled = True
                
    
    def get_ds(self) -> None:
        
        self.train_ds = self.train_data_loader.get_ds_ready()
        self.train_ds = self.train_ds.shuffle(self.buffer_size).batch(self.batch_size, drop_remainder = True).cache().prefetch(buffer_size = AUTOTUNE)
        self.inverse_vocab = self.train_data_loader.get_vocab()
        
        if self.val_data_loader:
            
            self.val_ds = self.val_data_loader.get_ds_ready()
            self.val_ds = self.val_ds.shuffle(self.buffer_size).batch(self.batch_size, drop_remainder = True).cache().prefetch(buffer_size = AUTOTUNE)

        else:
            
            self.val_ds = None
    
    
    def save_weights(self) -> None:
        
        if not os.path.exists(self.mod_dir):
            os.makedirs(self.mod_dir)
        
        self.model.save_weights(self.mod_dir + self.model_name + '.h5')


    def log_metadata(self) -> None:
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        with open(os.path.join(self.log_dir, 'metadata.tsv'), 'w') as f:
            for word in self.inverse_vocab:
                f.write('{}\n'.format(word))
            for unknown in range(1, VOCAB_SIZE - len(self.inverse_vocab)):
                f.write('unknown #{}\n'.format(unknown))

    
    def log_embeddings(self) -> None:
        
        weights = tf.Variable(self.model.layers[0].get_weights()[0][1:])
        # Create a checkpoint from embedding, the filename and key are the
        # name of the tensor.
        checkpoint = tf.train.Checkpoint(embedding = weights)
        checkpoint.save(os.path.join(self.log_dir, 'embedding.ckpt'))


    def config_projector(self) -> None:
    
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()

        embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
        embedding.metadata_path = 'metadata.tsv'
        projector.visualize_embeddings(self.log_dir, config)




