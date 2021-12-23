import tensorflow as tf

import yaml

import argparse

import word2vec.utils.trainer as t
import word2vec.models.word2vec_models as w
import word2vec.archs.data_loader as dl
import word2vec.archs.constants as c





# ********************* launch training ***********************
# cmd to launch : python -m word2vec.train --config '.word2vec/config/config.yaml'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'word2vec training')
    parser.add_argument('--config', type=str, required=True, help='path to yaml config')
    args = parser.parse_args()
    
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    train(config)