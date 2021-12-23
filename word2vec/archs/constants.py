CBOW_N_WORDS = 3
SKIPGRAM_N_WORDS = 3 # context = 3 words before, 3 words after the target word

MIN_WORD_FREQUENCY = 30
MAX_SEQ_LENGTH = 256 # define it to 0 for no truncature

EMBEDDING_DIM = 256

VOCAB_SIZE = 4096
BATCH_SIZE = 64

LANGUAGE = 'english'

BUFFER_SIZE = 8192

#/!\ only useful for train_old, otherwise modify it in the config.yaml file 
PATH = '/coding_linux20/programming/datasets/wikitext-2-raw' + '/wiki.train.raw'
LOG_PATH = './logs' # path from word2vec_tf, because we launch with 'python -m word2vec.train'
WEIGHTS_PATH = './word2vec/weights/'

NUM_EPOCHS = 5
LR_INI = 0.025
PENTE = - LR_INI / NUM_EPOCHS