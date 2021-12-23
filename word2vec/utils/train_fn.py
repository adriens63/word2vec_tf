import word2vec.utils.trainer as t
import word2vec.utils.helper as h
import word2vec.models.word2vec_models as w
import word2vec.archs.data_loader as dl
import word2vec.archs.constants as c





def train(config):
    
    model_class = h.get_model_class(config['type_model'])
    w2v = model_class()
    
    lr_scheduler = h.get_lr_scheduler_fn(config['lr_scheduler'])
    
    train_data_loader = dl.GetDataset(config['type_model'], c.PATH)

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
                        train_steps = config['train_step'],
                        val_data_loader = None,
                        val_steps = config['val_step'],
                        checkpoint_frequency = config['checkpoint_frequency'],
                        model_name = config['model_name'],
                        model_dir = c.WEIGHTS_PATH,
                        log_dir = c.LOG_PATH
                        )
    