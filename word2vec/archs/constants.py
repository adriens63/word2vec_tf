N_ELEMNT = 10000

CBOW_N_WORDS = 3
SKIPGRAM_N_WORDS = 3 # context = 3 words before, 3 words after the target word

MIN_WORD_FREQUENCY = 30
MAX_SEQ_LENGTH = 256 # define it to 0 for no truncature

EMBEDDING_DIM = 256

VOCAB_SIZE = 4096
BATCH_SIZE = 64

LANGUAGE = 'french'

BUFFER_SIZE = 8192

NUM_EPOCHS = 4
LR_INI = 0.025
PENTE = - LR_INI / NUM_EPOCHS