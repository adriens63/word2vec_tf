import yaml

import argparse

import word2vec.utils.train_fn as t_fn





# ********************* launch training ***********************
# cmd to launch : python -m word2vec.train --config ./word2vec/config/config.yml
# cmd to visualize : tensorboard --logdir=./word2vec/weights/w2v_skip_gram_1/log_dir/ --port=8012

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser(description = 'word2vec training')
    parser.add_argument('--config', type = str, required = True, help = 'path to yaml config')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    t_fn.train(config)