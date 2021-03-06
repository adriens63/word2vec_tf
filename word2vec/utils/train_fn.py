#TODO formating
import word2vec.utils.trainer as t
import word2vec.utils.helper as h
import word2vec.archs.data_loader as dl
import word2vec.archs.data_loader_small_wiki as dls
import word2vec.archs.constants as c





def train(config) -> None:
    
    model_class = h.get_model_class(config['type_model'])
    w2v = model_class()
    
    lr_scheduler = h.get_lr_scheduler_fn(lr_scheduler = config['lr_scheduler'], initial_lr = config['lr'], step_size = config['epochs'])
    
    if config['small_wiki']:
        train_data_loader = dls.GetDataset(config['type_model'], config['train_path'])
        
    else:
        train_data_loader = dl.GetDataset(config['type_model'], config['train_path'])
        
    val_data_loader = None
    
    trainer = t.Trainer(
                        device = config['device'],
                        model = w2v,
                        epochs = config['epochs'],
                        batch_size = c.BATCH_SIZE,
                        buffer_size = c.BUFFER_SIZE,
                        loss_fn = config['loss_fn'],
                        optimizer = config['optimizer'],
                        lr_scheduler = lr_scheduler,
                        train_data_loader = train_data_loader,
                        train_steps = config['train_steps'],
                        val_data_loader = val_data_loader,
                        val_steps = config['val_steps'],
                        checkpoint_frequency = config['checkpoint_frequency'],
                        model_name = config['model_name'],
                        weights_path = config['weights_path'],
                        )
    
    trainer.get_ds()
    trainer.compile()
    trainer.launch_training()
    
    trainer.save_weights()
    trainer.log_metadata()
    trainer.log_embeddings()
    trainer.config_projector()
    
    h.log_config(config, config['weights_path'] + config['model_name'])
    print('w2v saved to directory: ', config['weights_path'])